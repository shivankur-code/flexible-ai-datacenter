"""SLO-aware LLM inference replica simulator.

One *replica* is an 8-GPU server serving a 70B-class model with tensor
parallelism. It runs vLLM-style continuous batching at iteration level. Each
iteration serves one decode token for every running sequence, plus prompt *chunks* that fill the
rest of the per-iteration token budget (chunked prefill, the vLLM V1 default).

Iteration time comes from a roofline model. Prefill is compute-bound
(FLOPs / (peak * MFU)). Decode is bandwidth-bound: weights plus KV cache read
once per step, plus a compute term that grows with batch size.

The power model matches the time model. Each phase demands a GPU power of
``idle + (TDP - idle) * activity``. A power cap ``kappa * TDP`` that sits below
that demand slows the phase by ``((cap-idle)/(demand-idle))**gamma``. Capping
therefore trades latency for power at a rate set by how compute-bound the phase
is. This trade-off is what turns SLO headroom into grid flexibility.

All parameters live in ``configs/fleet_default.yaml``, with sources or
ASSUMPTION markers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fleet_default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@dataclass
class Replica:
    """Time and power model of one serving replica at power cap ``kappa`` (fraction of TDP)."""

    cfg: dict
    kappa: float = 1.0

    def __post_init__(self):
        r, m = self.cfg["replica"], self.cfg["model"]
        self.r, self.m = r, m
        self.n = r["n_gpus"]
        self.flops_per_token = 2.0 * m["params"]
        self.weight_bytes = m["params"] * m["bytes_per_param"]
        self.kv_capacity_tokens = int((self.n * r["gpu_mem_bytes"] * r["kv_mem_fraction"] - self.weight_bytes) / m["kv_bytes_per_token"])
        if self.kv_capacity_tokens <= 0:
            raise ValueError("model does not fit in GPU memory")

    # ---- power-cap response
    def _phase(self, activity: float, gamma: float) -> tuple[float, float]:
        """Return (per-GPU power W, performance factor <= 1) for a phase demanding ``activity``."""
        idle, tdp = self.r["gpu_idle_w"], self.r["gpu_tdp_w"]
        demand = idle + (tdp - idle) * activity
        cap = self.kappa * tdp
        if demand <= cap:
            return demand, 1.0
        return cap, ((cap - idle) / (demand - idle)) ** gamma

    def _host_w(self, gpu_w: float) -> float:
        r = self.r
        frac = (gpu_w - r["gpu_idle_w"]) / (r["gpu_tdp_w"] - r["gpu_idle_w"])
        return r["host_idle_w"] + (r["host_active_w"] - r["host_idle_w"]) * max(frac, 0.0)

    def prefill(self, tokens: int) -> tuple[float, float]:
        """(seconds, server watts) for a prefill iteration over ``tokens`` prompt tokens."""
        r = self.r
        gpu_w, perf = self._phase(r["activity_prefill"], r["gamma_prefill"])
        t = r["iter_overhead_s"] + tokens * self.flops_per_token / (self.n * r["peak_flops"] * r["mfu_prefill"] * perf)
        return t, self.n * gpu_w + self._host_w(gpu_w)

    def decode(self, batch: int, kv_tokens: int) -> tuple[float, float]:
        """(seconds, server watts) for one decode step of ``batch`` sequences holding ``kv_tokens``."""
        r, m = self.r, self.m
        a = r["activity_decode_min"] + (r["activity_decode_max"] - r["activity_decode_min"]) * min(batch / r["decode_batch_sat"], 1.0)
        gpu_w, perf = self._phase(a, r["gamma_decode"])
        mem = (self.weight_bytes + kv_tokens * m["kv_bytes_per_token"]) / (self.n * r["mem_bw"] * r["mbu_decode"])
        comp = batch * self.flops_per_token / (self.n * r["peak_flops"] * r["mfu_decode"])
        t = r["iter_overhead_s"] + (mem + comp) / perf
        return t, self.n * gpu_w + self._host_w(gpu_w)

    @property
    def idle_w(self) -> float:
        return self.n * self.r["gpu_idle_w"] + self.r["host_idle_w"]

    @property
    def max_w(self) -> float:
        return self.n * self.r["gpu_tdp_w"] + self.r["host_active_w"]


@dataclass
class SimResult:
    rate: float              # offered req/s
    n_done: int
    ttft_p99: float
    tpot_p99: float
    ttft_p50: float
    tpot_p50: float
    avg_power_w: float       # server power, including idle gaps
    tokens_per_s: float      # generated tokens/s
    joules_per_token: float  # server energy per generated token (output)
    backlog: int             # requests still queued at the end (instability signal)


def simulate(replica: Replica, rate: float, ctx: np.ndarray, gen: np.ndarray, duration_s: float = 600.0,
             warmup_s: float = 60.0, seed: int = 0, arrivals: np.ndarray | None = None) -> SimResult:
    """Simulate one replica with chunked-prefill continuous batching (vLLM V1 style).

    Each iteration serves one decode token for every running sequence, then
    fills the rest of the ``max_prefill_tokens`` budget with prompt chunks,
    FIFO. A request is admitted (its KV for prompt + output reserved) when its
    first chunk is scheduled. It emits its first token at the end of the
    iteration that finishes its prompt. Iteration time is a prefill sub-phase
    for the chunk plus a decode sub-phase for the running batch, each with its
    own power under the cap.

    Request sizes are drawn with replacement from ``(ctx, gen)``. Metrics cover
    requests that arrive after ``warmup_s``. The simulation drains every request,
    so overload shows up as unbounded TTFT. ``backlog`` is the queue length at
    ``duration_s``.
    """
    rng = np.random.default_rng(seed)
    if arrivals is None:
        n = rng.poisson(rate * duration_s)
        arrivals = np.sort(rng.uniform(0, duration_s, n))
    n = len(arrivals)
    pick = rng.integers(0, len(ctx), n)
    c_in = np.maximum(np.minimum(ctx[pick].astype(np.int64), replica.kv_capacity_tokens // 2), 1)
    g_out = np.maximum(gen[pick].astype(np.int64), 1)

    max_batch = replica.r["max_batch"]
    budget = replica.r["max_prefill_tokens"]
    kv_cap = replica.kv_capacity_tokens
    overhead = replica.r["iter_overhead_s"]
    first = np.full(n, np.nan)
    done = np.full(n, np.nan)

    run_id = np.empty(0, np.int64)     # decoding sequences
    run_left = np.empty(0, np.int64)
    run_kv = np.empty(0, np.int64)
    pre = []                           # admitted, prompt partly processed: [id, prompt tokens left]
    kv_reserved = 0

    t, nxt, energy, gen_tokens = 0.0, 0, 0.0, 0
    backlog = None
    idle_w = replica.idle_w
    while True:
        if backlog is None and t >= duration_s:
            backlog = int(np.searchsorted(arrivals, t, side="right") - nxt + len(pre))
        arrived_end = np.searchsorted(arrivals, t, side="right")
        n_dec = len(run_id)
        room = budget - n_dec
        # admit new requests into the prefill queue while KV and batch slots allow
        while nxt < arrived_end and n_dec + len(pre) < max_batch and kv_reserved + c_in[nxt] + g_out[nxt] <= kv_cap:
            pre.append([nxt, int(c_in[nxt])])
            kv_reserved += int(c_in[nxt] + g_out[nxt])
            nxt += 1
        chunk, finished = 0, []
        for item in pre:
            if room - chunk <= 0:
                break
            take = min(item[1], room - chunk)
            item[1] -= take
            chunk += take
            if item[1] == 0:
                finished.append(item[0])
        if chunk == 0 and n_dec == 0:
            if nxt >= n:
                break
            t_next = max(arrivals[nxt], t)
            energy += idle_w * (t_next - t)
            t = t_next
            continue
        dt, e = 0.0, 0.0
        if chunk:
            tp, wp = replica.prefill(chunk)
            dt += tp
            e += wp * tp
        if n_dec:
            td, wd = replica.decode(n_dec, int(run_kv.sum()))
            if chunk:
                td -= overhead
            dt += td
            e += wd * td
        t += dt
        energy += e
        # decode progress
        if n_dec:
            gen_tokens += n_dec
            run_left -= 1
            run_kv += 1
            fin = run_left <= 0
            if fin.any():
                done[run_id[fin]] = t
                kv_reserved -= int((run_kv[fin] + run_left[fin]).sum())
                run_id, run_left, run_kv = run_id[~fin], run_left[~fin], run_kv[~fin]
        # prompts that completed this iteration emit their first token
        if finished:
            pre = [it for it in pre if it[1] > 0]
            ids = np.array(finished)
            first[ids] = t
            gen_tokens += len(ids)
            left = g_out[ids] - 1
            fin = left <= 0
            done[ids[fin]] = t
            kv_reserved -= int((c_in[ids[fin]] + g_out[ids[fin]]).sum())
            run_id = np.concatenate([run_id, ids[~fin]])
            run_left = np.concatenate([run_left, left[~fin]])
            run_kv = np.concatenate([run_kv, c_in[ids[~fin]] + 1])

    span = max(t, duration_s)
    if t < duration_s:
        energy += idle_w * (duration_s - t)
    m = arrivals >= warmup_s
    ttft = first[m] - arrivals[m]
    multi = m & (g_out > 1)
    tpot = (done[multi] - first[multi]) / (g_out[multi] - 1)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else 0.0
    return SimResult(
        rate=rate,
        n_done=int(m.sum()),
        ttft_p99=q(ttft, 99), tpot_p99=q(tpot, 99), ttft_p50=q(ttft, 50), tpot_p50=q(tpot, 50),
        avg_power_w=float(energy / span),
        tokens_per_s=float(gen_tokens / span),
        joules_per_token=float(energy / max(gen_tokens, 1)),
        backlog=backlog or 0,
    )


def meets(res: SimResult, ttft_s: float, tpot_s: float) -> bool:
    return res.ttft_p99 <= ttft_s and res.tpot_p99 <= tpot_s


def capacity(replica: Replica, ctx, gen, ttft_s: float, tpot_s: float, lo: float = 0.05, hi: float = 64.0,
             iters: int = 12, **sim_kw) -> tuple[float, SimResult]:
    """Largest Poisson rate (req/s) at which one replica meets the p99 SLOs (bisection on a log scale)."""
    best = None
    if not meets(simulate(replica, lo, ctx, gen, **sim_kw), ttft_s, tpot_s):
        return 0.0, simulate(replica, lo, ctx, gen, **sim_kw)
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        r = simulate(replica, mid, ctx, gen, **sim_kw)
        if meets(r, ttft_s, tpot_s):
            lo, best = mid, r
        else:
            hi = mid
    if best is None:
        best = simulate(replica, lo, ctx, gen, **sim_kw)
    return lo, best


def saturation(replica: Replica, ctx, gen, n_requests: int = 3000, seed: int = 0) -> SimResult:
    """Throughput-optimal (batch) operation: every request is queued at t=0, latency is ignored."""
    return simulate(replica, 0.0, ctx, gen, duration_s=1e-9, warmup_s=0.0, seed=seed, arrivals=np.zeros(n_requests))


def build_table(cfg: dict, tokens, services=("conv", "code"), rate_fracs=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                duration_s: float = 600.0, seed: int = 0):
    """Per service and power cap: SLO capacity (req/s per replica) and a power-vs-rate curve.

    Returns ``(interactive, batch)`` DataFrames.

    ``interactive`` has one row per (service, kappa, rate) with ``mu`` (SLO
    capacity at that cap) and ``power_w``. Rates are fractions of the full-power
    capacity ``mu_1``, so the power at any load can be interpolated.

    ``batch`` has one row per kappa with saturation throughput (req/s), server
    power and joules per request, using the ``batch.token_mix`` sizes.
    """
    import pandas as pd

    rows, brow = [], []
    for svc in services:
        d = tokens[tokens.service == svc]
        ctx, gen = d.ctx_tokens.to_numpy(), d.gen_tokens.to_numpy()
        slo = cfg["slo"][svc]
        mu1 = None
        for kappa in cfg["power_caps"]:
            rep = Replica(cfg, kappa)
            mu, _ = capacity(rep, ctx, gen, slo["ttft_s"], slo["tpot_s"], duration_s=duration_s, seed=seed)
            if mu1 is None:
                mu1 = mu
            for f in rate_fracs:
                res = simulate(rep, f * mu1, ctx, gen, duration_s=duration_s, seed=seed)
                rows.append(dict(service=svc, kappa=kappa, mu=mu, mu1=mu1, rate=f * mu1, power_w=res.avg_power_w,
                                 ttft_p99=res.ttft_p99, tpot_p99=res.tpot_p99, joules_per_token=res.joules_per_token))
    d = tokens[tokens.service == cfg["batch"]["token_mix"]]
    ctx, gen = d.ctx_tokens.to_numpy(), d.gen_tokens.to_numpy()
    for kappa in cfg["power_caps"]:
        rep = Replica(cfg, kappa)
        n_req = 3000
        res = saturation(rep, ctx, gen, n_requests=n_req, seed=seed)
        brow.append(dict(kappa=kappa, power_w=res.avg_power_w, tokens_per_s=res.tokens_per_s,
                         joules_per_token=res.joules_per_token, idle_w=rep.idle_w, max_w=rep.max_w))
    return pd.DataFrame(rows), pd.DataFrame(brow)


def cached_table(cfg: dict, tokens, cache_dir: Path | str | None = None, **kw):
    """:func:`build_table`, cached on disk under a hash of the config."""
    import hashlib
    import json

    import pandas as pd

    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode() + str(sorted(kw.items())).encode()).hexdigest()[:12]
    d = Path(cache_dir or REPO_ROOT / "results" / "fleet_tables")
    fi, fb = d / f"{key}_interactive.csv", d / f"{key}_batch.csv"
    if fi.exists() and fb.exists():
        return pd.read_csv(fi), pd.read_csv(fb)
    it, bt = build_table(cfg, tokens, **kw)
    d.mkdir(parents=True, exist_ok=True)
    it.to_csv(fi, index=False)
    bt.to_csv(fb, index=False)
    return it, bt

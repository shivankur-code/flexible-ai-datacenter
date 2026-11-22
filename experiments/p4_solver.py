"""Phase 4: feasibility-guaranteed learned solver for joint fleet-grid dispatch (RQ4).

Family: 24 h DC-OPF on RTS-GMLC (0.6x ratings) with one LLM-inference facility
at bus 313 (5,472 decision variables per instance after completion).
Instances: each day of 2020 x 4 random envelopes (facility 150/300/450 MW,
batch fill 0.25/0.5/0.75). Days with day % 5 == 0 are held out (test). OOD
test: a 600 MW facility at 0.9 fill on the test days.

Exact methods (optimal and feasible by construction), timed one instance at a time:
  HiGHS on the full LP; line-constraint generation (no learning); *learned
  screening* (ours): an MLP predicts binding (hour, line) limits, and the LP
  starts from that set, adding back any violated limit; and oracle screening
  (the true binding set, the best case for any screen).
End-to-end learned solvers (batched): a pure gauge map around a closed-form
interior anchor (feasible), DC3-style completion + correction (approximately
feasible), and predict-correct-project (DC3 + closed-form ray projection, feasible).

    python experiments/p4_solver.py [--quick]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p4"
SITE = 313
SIZES, FILLS = (150.0, 300.0, 450.0), (0.25, 0.5, 0.75)


def _setup():
    from gridflex.fleet import cached_table, load_config
    from gridflex.grid import load_rts_gmlc
    from gridflex.solvers.problem import DispatchFamily
    from gridflex.workload import load_tokens, weekly_profile, year_profile

    cfg = load_config()
    it, bt = cached_table(cfg, load_tokens())
    net = load_rts_gmlc().with_ratings(0.6)
    yp = year_profile(net.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    return cfg, it, bt, net, yp, DispatchFamily(net, SITE)


def _gen(args):
    days, variants, seed = args
    from gridflex.envelope import build_envelope
    from gridflex.solvers.data import make_dataset
    from gridflex.solvers.problem import Batch

    torch.set_num_threads(1)
    cfg, it, bt, net, yp, fam = _setup()
    envs = {v: build_envelope(cfg, SITE, v[0], net.timestamps, yp, it, bt, batch_fill=v[1]) for v in set(variants)}
    rng = np.random.default_rng(seed)
    parts = []
    for d in days:
        vs = variants if len(variants) == 1 else [variants[i] for i in rng.choice(len(variants), 4, replace=False)]
        parts.append(make_dataset(net, fam, [envs[v] for v in vs], [d]))
    return dict(batch=Batch.cat([p["batch"] for p in parts]), s_star=torch.cat([p["s_star"] for p in parts]),
                obj_star=torch.cat([p["obj_star"] for p in parts]),
                highs_total_s=np.concatenate([p["highs_total_s"] for p in parts]),
                highs_solve_s=np.concatenate([p["highs_solve_s"] for p in parts]),
                day=np.concatenate([p["meta"][:, 0] for p in parts]))


def generate(days, variants, workers, tag):
    path = OUT / f"data_{tag}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from gridflex.solvers.problem import Batch

    chunks = [(list(c), variants, i) for i, c in enumerate(np.array_split(np.array(days), workers * 4)) if len(c)]
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_gen, chunks))
    D = dict(batch=Batch.cat([p["batch"] for p in parts]), s_star=torch.cat([p["s_star"] for p in parts]),
             obj_star=torch.cat([p["obj_star"] for p in parts]),
             highs_total_s=np.concatenate([p["highs_total_s"] for p in parts]),
             highs_solve_s=np.concatenate([p["highs_solve_s"] for p in parts]),
             day=np.concatenate([p["day"] for p in parts]))
    print(f"generated {tag}: {len(D['batch'])} instances in {time.time() - t:.0f}s", flush=True)
    torch.save(D, path)
    return D


def _dc_from(b, i):
    from gridflex.opf import DataCenter

    n = lambda a: a[i].numpy()
    return DataCenter(SITE, firm=n(b.firm), shed_max=n(b.shedmax), shed_budget=float(b.budget[i]), flex_max=n(b.flexmax),
                      cum_min=n(b.cummin), cum_max=n(b.cummax), late_cost=500.0)


def time_exact(fam, D, masks):
    """Mean wall time of screened exact solves for the given initial masks (calibration)."""
    from gridflex.opf import solve_opf_screened

    ts = []
    for i in range(len(D["batch"])):
        tic = time.perf_counter()
        solve_opf_screened(fam.net, 24 * int(D["day"][i]), 24, [_dc_from(D["batch"], i)], initial=np.array(masks[i]), compute_lme=False)
        ts.append(time.perf_counter() - tic)
    return float(np.mean(ts))


def exact_benchmark(fam, screen, freq_mask, D, split, log):
    """Time the exact methods one instance at a time, in this process (HiGHS single run each)."""
    from gridflex.opf import solve_opf, solve_opf_screened
    from gridflex.solvers.models import binding_labels

    net, b = fam.net, D["batch"]
    tic = time.perf_counter()
    pred = screen.predict(b).numpy()
    nn_ms = 1e3 * (time.perf_counter() - tic) / len(b)       # batched inference, amortised per instance
    oracle = binding_labels(fam, D["s_star"], b).numpy() > 0.5
    rec = dict(full=[], cg=[], freq=[], learned=[], oracle=[])
    rounds = dict(cg=[], freq=[], learned=[], oracle=[])
    gaps, rows_kept = [], []
    for i in range(len(b)):
        t0 = 24 * int(D["day"][i])
        dc = [_dc_from(b, i)]
        tic = time.perf_counter(); full = solve_opf(net, t0, 24, dc, compute_lme=False); rec["full"].append(time.perf_counter() - tic)
        for key, init in (("cg", None), ("freq", freq_mask), ("learned", pred[i]), ("oracle", oracle[i])):
            tic = time.perf_counter()
            r = solve_opf_screened(net, t0, 24, dc, initial=init, compute_lme=False)
            rec[key].append(time.perf_counter() - tic)
            rounds[key].append(r.info["rounds"])
            if key == "learned":
                gaps.append(abs(r.objective - full.objective) / abs(full.objective))
                rows_kept.append(r.info["monitored"])
                viol = float(np.maximum(np.abs(r.flow) - net.line_rating[:, None], 0).max())
                assert viol < 1e-5, viol
    full_ms = 1e3 * np.mean(rec["full"])
    out = [dict(method="HiGHS (full LP)", split=split, n=len(b), gap_mean=0.0, gap_max=0.0, viol_max=0.0, feasible_frac=1.0,
                ms_per_instance=full_ms, speedup=1.0, rounds_mean=1.0)]
    names = dict(cg="Constraint generation (no learning)", freq="Frequency screening + CG (no learning)",
                 learned="Learned screening + CG", oracle="Oracle screening (bound)")
    for key in ("cg", "freq", "learned", "oracle"):
        ms = 1e3 * np.mean(rec[key]) + (nn_ms if key == "learned" else 0.0)
        out.append(dict(method=names[key], split=split, n=len(b), gap_mean=float(np.mean(gaps)) if key == "learned" else 0.0,
                        gap_max=float(np.max(gaps)) if key == "learned" else 0.0, viol_max=0.0, feasible_frac=1.0,
                        ms_per_instance=ms, speedup=full_ms / ms, rounds_mean=float(np.mean(rounds[key])),
                        rows_monitored_mean=float(np.mean(rows_kept)) if key == "learned" else np.nan,
                        nn_ms=nn_ms if key == "learned" else np.nan))
    p, y = pred.reshape(len(b), -1), oracle.reshape(len(b), -1)
    log(f"  {split}: screen recall {float((p & y).sum() / max(y.sum(), 1)):.4f}, monitored {p.sum(1).mean():.0f} of {p.shape[1]} rows "
        f"(binding {y.sum(1).mean():.0f}); one-round instances {np.mean(np.array(rounds['learned']) == 1):.2%}")
    return out


def figure(df, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURFACE, BLUE, RED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#e34948"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    d = df[df.split == "test"].copy()
    full = float(d[d.method == "HiGHS (full LP)"].ms_per_instance.iloc[0])
    d["speedup"] = full / d.ms_per_instance
    d = d.sort_values("speedup")
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(len(d))
    exact = d.gap_mean.abs() < 1e-9
    feas = d.feasible_frac >= 1.0
    ax.barh(y, d.speedup, color=[BLUE if e else (RED if not f else "#b9b8b3") for e, f in zip(exact, feas)], height=0.55)
    for yi, (_, r) in zip(y, d.iterrows()):
        if abs(r.gap_mean) < 1e-9:
            note = "exact optimum, zero violation"
        elif r.feasible_frac >= 1.0:
            note = f"feasible, gap {100 * r.gap_mean:,.0f}%"
        else:
            note = f"gap {100 * r.gap_mean:+.1f}%, violates up to {r.viol_max:,.0f} MW"
        ax.annotate(f"{r.speedup:,.1f}x  {note}", (r.speedup, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d.method)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xlim(right=d.speedup.max() * 60)
    ax.set_xlabel("speed-up over HiGHS on the full LP (per 24 h instance, log scale)")
    ax.set_title("Joint fleet-grid dispatch, 5.5k variables: speed vs guarantees (held-out days)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "fig5_solver_benchmark.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "figures" / "fig5_solver_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from gridflex.solvers.models import (DC3Solver, GaugeSolver, ProjectedSolver, ScreenNet, binding_labels, evaluate,
                                         train_dc3, train_gauge, train_screen)
    from gridflex.solvers.problem import max_violation

    torch.manual_seed(0)
    variants = [(s, f) for s in SIZES for f in FILLS]
    days = list(range(0, 366, 3 if args.quick else 1))
    test_days = [d for d in days if d % 5 == 0]
    train_days = [d for d in days if d % 5 != 0]
    tag = "quick" if args.quick else "full"
    tr = generate(train_days, variants, args.workers, f"train_{tag}")
    te = generate(test_days, variants, args.workers, f"test_{tag}")
    ood = generate(test_days, [(600.0, 0.9)], args.workers, f"ood_{tag}")
    _, _, _, _, _, fam = _setup()
    torch.set_num_threads(4)
    ep = dict(epochs_sup=40, epochs_ssl=20) if args.quick else dict(epochs_sup=150, epochs_ssl=100)

    log = open(OUT / f"train_{tag}.log", "w")
    pr = lambda m: (print(m, flush=True), log.write(m + "\n"), log.flush())
    t = time.time()
    gauge = GaugeSolver(fam, tr["batch"])
    train_gauge(gauge, tr["batch"], tr["s_star"], tr["obj_star"], log=pr, **ep)
    t_gauge = time.time() - t
    t = time.time()
    dc3 = DC3Solver(fam, tr["batch"])
    train_dc3(dc3, tr["batch"], tr["s_star"], tr["obj_star"], log=pr, epochs_sup=ep["epochs_sup"], epochs_ssl=ep["epochs_ssl"] // 2)
    t_dc3 = time.time() - t

    # screen: fit on most training instances; choose its threshold (and the frequency baseline's) by *timed*
    # exact solves on held-out validation instances, minimising mean wall time
    t = time.time()
    n_tr = len(tr["batch"])
    # validation = whole held-out *days* (a day's other envelopes must not sit in the fit set)
    tr_days = np.unique(tr["day"])
    rng = np.random.default_rng(1)
    val_days = rng.choice(tr_days, size=max(len(tr_days) // 12, 4), replace=False)
    is_val = np.isin(tr["day"], val_days)
    vi, fi = torch.tensor(np.flatnonzero(is_val)), torch.tensor(np.flatnonzero(~is_val))
    fit = {k: (v[fi] if isinstance(v, torch.Tensor) else (v[fi.numpy()] if isinstance(v, np.ndarray) else v)) for k, v in tr.items() if k != "batch"}
    fit["batch"] = tr["batch"].index(fi)
    val = {k: (v[vi] if isinstance(v, torch.Tensor) else (v[vi.numpy()] if isinstance(v, np.ndarray) else v)) for k, v in tr.items() if k != "batch"}
    val["batch"] = tr["batch"].index(vi)
    y_fit = binding_labels(fam, fit["s_star"], fit["batch"])
    screen = ScreenNet(fam, fit["batch"])
    train_screen(screen, fit["batch"], y_fit, log=pr, epochs=40 if args.quick else 150, recall_target=0.99)
    with torch.no_grad():
        p_val = torch.sigmoid(screen.logits(val["batch"])).reshape(len(val["batch"]), fam.T, fam.L).numpy()
    freq = y_fit.mean(0).numpy()                                   # (T,L) binding frequency in training
    best = {}
    for kind, cands in (("learned", (0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)), ("frequency", (0.01, 0.03, 0.1, 0.2, 0.35, 0.5))):
        res = []
        for c in cands:
            masks = (p_val >= c) if kind == "learned" else np.broadcast_to(freq >= c, p_val.shape)
            res.append((time_exact(fam, val, masks), c))
            pr(f"  calibrate {kind:9s} threshold {c:<6} mean {1e3 * res[-1][0]:.1f} ms, rows {masks.reshape(len(masks), -1).sum(1).mean():.0f}")
        best[kind] = min(res)[1]
    screen.threshold = best["learned"]
    freq_mask = freq >= best["frequency"]
    t_screen = time.time() - t
    thr = screen.threshold
    pr(f"screen threshold {thr} | frequency-screen threshold {best['frequency']} ({int(freq_mask.sum())} rows)")
    rows = []
    for split, D in (("test", te), ("ood_600MW", ood)):
        rows += exact_benchmark(fam, screen, freq_mask, D, split, pr)
        b = D["batch"]
        fam.anchor(b)
        anchor_ok = float((~torch.isnan(fam.last_anchor_shed)).float().mean())
        for model, name in ((gauge, "Gauge map (pure)"), (dc3, "DC3-style"), (ProjectedSolver(dc3), "Predict-correct-project")):
            r, s = evaluate(model, b, D["obj_star"], name)
            full_ms = [x for x in rows if x["split"] == split and x["method"] == "HiGHS (full LP)"][0]["ms_per_instance"]
            r.update(split=split, anchor_interior_frac=anchor_ok, speedup=full_ms / r["ms_per_instance"], gap_max=float("nan"))
            rows.append(r)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"benchmark_{tag}.csv", index=False)
    meta = dict(n_train=len(tr["batch"]), n_test=len(te["batch"]), n_ood=len(ood["batch"]), n_vars=fam.n,
                train_s_gauge=t_gauge, train_s_dc3=t_dc3, train_s_screen=t_screen, screen_threshold=thr,
                threads=torch.get_num_threads())
    json.dump(meta, open(OUT / f"meta_{tag}.json", "w"), indent=1)
    torch.save(dict(gauge=gauge.state_dict(), dc3=dc3.state_dict(), screen=screen.state_dict()), OUT / f"models_{tag}.pt")
    figure(df, tag)
    pd.set_option("display.width", 250)
    print(json.dumps(meta))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

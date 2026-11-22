"""Data-center operating modes and the rolling-horizon year simulation (RQ2, RQ3).

Modes:

* ``none``: no data center (grid baseline).
* ``inflexible``: fixed load. Interactive runs at full power and batch work is
  served on arrival.
* ``price``: the DC best-responds to the day-ahead LMP at its bus, taken from
  the ``inflexible`` run (it ignores its own price impact). The grid then
  re-dispatches around the resulting load.
* ``avg_ci``: the same, but following the system *average* emission rate (the
  signal most carbon-aware schedulers use).
* ``lme``: the same, but following the *locational marginal emission* rate at
  the DC bus.
* ``coopt``: the DC envelope is co-optimised inside the OPF (min system cost).

``carbon_price`` ($/t) is a grid-wide policy scenario. It adds to every
generator's cost in *all* modes, so flexible and inflexible DCs are compared on
the same carbon-priced grid.

Every mode uses the same envelope: shed within the SLO, shift batch work within
deadlines. Only the signal differs. The *true* emissions of every mode come
from the grid's own re-dispatch, not from the signal.

Rolling horizon: each day we solve a ``horizon``-hour window (48 h by default,
with perfect day-ahead foresight of load and renewables), commit the first
24 h, and carry generator ramp state and the actual batch backlog into the next
day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import highspy
import numpy as np
import pandas as pd

from .envelope import Envelope
from .grid import Network
from .opf import DataCenter, OPFResult, _LPBuilder, solve_opf

MODES = ("none", "inflexible", "price", "avg_ci", "lme", "coopt")


# ---------------------------------------------------------------- DC-side best response
def best_response(dc: DataCenter, signal: np.ndarray, tie_eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Schedule one DC to minimise ``sum(signal * d)`` within its envelope.

    Returns ``(d, shed, flex, late)``. Ties, which flat signals make common, are
    broken toward serving work early and shedding late (``tie_eps``), which
    matches the inflexible operator.
    """
    T = dc.T
    s = np.asarray(signal, dtype=float)
    scale = max(np.abs(s).max(), 1e-9)
    s = s / scale  # normalise so the tie-break is relative
    inf = highspy.kHighsInf
    lp = _LPBuilder()
    early = tie_eps * np.arange(T)
    sh = lp.add_vars(T, 0.0, dc.shed_max, -s + dc.shed_cost / scale - early[::-1])
    fx = lp.add_vars(T, 0.0, dc.flex_max, s + early)
    late = None
    if dc.cum_min is not None and dc.late_cost is not None:
        late = lp.add_vars(T, 0.0, inf, dc.late_cost / scale)
    if np.isfinite(dc.shed_budget) and dc.shed_max.any():
        blocks = np.arange(T) // 24
        lp.add_rows(blocks, sh, np.ones(T), -inf, dc.shed_budget, int(blocks[-1]) + 1)
    if dc.cum_min is not None or dc.cum_max is not None:
        lo = dc.cum_min if dc.cum_min is not None else np.full(T, -inf)
        hi = dc.cum_max if dc.cum_max is not None else np.full(T, inf)
        r = np.concatenate([np.full(t + 1, t) for t in range(T)])
        c = np.concatenate([fx[: t + 1] for t in range(T)])
        if late is not None:
            lp.add_rows(np.r_[r, np.arange(T)], np.r_[c, late], np.ones(len(c) + T), lo, np.full(T, inf), T)
            lp.add_rows(r, c, np.ones(len(c)), np.full(T, -inf), hi, T)
        else:
            lp.add_rows(r, c, np.ones(len(c)), lo, hi, T)
    if lp.n_row == 0:  # no coupling: bounds alone decide
        lp.add_rows([0], [fx[0]], [0.0], -inf, inf, 1)
    A, c, lb, ub, rlo, rhi = lp.matrix()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    m = highspy.HighsLp()
    m.num_col_, m.num_row_ = A.shape[1], A.shape[0]
    m.col_cost_, m.col_lower_, m.col_upper_ = c, lb, ub
    m.row_lower_, m.row_upper_ = rlo, rhi
    m.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    m.a_matrix_.start_, m.a_matrix_.index_, m.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(m)
    h.run()
    if h.modelStatusToString(h.getModelStatus()) != "Optimal":
        raise RuntimeError("DC best response not optimal")
    x = np.asarray(h.getSolution().col_value)
    shed, flex = x[sh], x[fx]
    lt = x[late] if late is not None else np.zeros(T)
    return dc.firm - shed + flex, shed, flex, lt


# ---------------------------------------------------------------- year simulation
@dataclass
class YearRun:
    mode: str
    hourly: pd.DataFrame                      # system and DC metrics per committed hour
    lmp: np.ndarray                           # (N, H) committed LMPs
    lme: np.ndarray                           # (N, H)
    info: dict = field(default_factory=dict)

    def totals(self) -> dict:
        h = self.hourly
        out = dict(mode=self.mode, **self.info)
        for col in ("gen_cost", "emissions", "curtailment", "congestion_rent", "grid_shed", "dc_energy", "dc_shed", "dc_flex", "dc_late"):
            if col in h:
                out[col] = float(h[col].sum())
        return out


def _commit_metrics(net: Network, r: OPFResult, n: int) -> dict:
    p = r.p[:, :n]
    avail = net.gen_avail[:, r.t0:r.t0 + n]
    return dict(
        gen_cost=net.gen_cost @ p,
        emissions=net.gen_co2 @ p,
        curtailment=(avail - p)[net.is_renewable].sum(0),
        congestion_rent=-(r.lmp[:, :n] * r.inj[:, :n]).sum(0),
        grid_shed=r.shed[:, :n].sum(0),
        binding_lines=r.binding_lines[:, :n].sum(0),
    )


def run_year(net: Network, envs: list[Envelope], mode: str, signals: dict | None = None, days: int | None = None,
             horizon: int = 48, carbon_price: float = 0.0, t_start: int = 0, levers: str = "flexible",
             fixed_loads: list[np.ndarray] | None = None) -> YearRun:
    """Rolling-horizon simulation of ``days`` days for one operating mode.

    ``signals[k]`` is the hourly signal for envelope k (from the inflexible
    run) and is needed for ``price``, ``avg_ci`` and ``lme``. ``levers``
    restricts the flexible modes to ``shift`` (deferral only) or ``shed``
    (power capping only). ``mode="fixed"`` replays the given DC load profiles,
    for example on a copper-plate network to measure congestion cost.
    """
    if mode not in MODES + ("fixed",):
        raise ValueError(mode)
    H = net.n_hours
    days = days or (H - t_start) // 24
    served = [float(e.cum_due[t_start - 1]) if t_start > 0 else 0.0 for e in envs]
    p_prev = None
    rows, lmps, lmes = [], [], []
    for d in range(days):
        t0 = t_start + 24 * d
        T = min(horizon, H - t0)
        n = min(24, T)
        dcs, plan = [], None
        if mode == "inflexible":
            dcs = [e.window(t0, T, "inflexible") for e in envs]
        elif mode == "fixed":
            dcs = [DataCenter.inflexible(e.bus, fixed_loads[k][t0:t0 + T]) for k, e in enumerate(envs)]
        elif mode == "coopt":
            dcs = [e.window(t0, T, levers, served_before=served[k]) for k, e in enumerate(envs)]
        elif mode in ("price", "avg_ci", "lme"):
            plan = []
            for k, e in enumerate(envs):
                w = e.window(t0, T, levers, served_before=served[k])
                dd, sh, fx, lt = best_response(w, signals[k][t0:t0 + T])
                plan.append((sh, fx, lt))
                dcs.append(DataCenter.inflexible(e.bus, dd, name=w.name))
        r = solve_opf(net, t0, T, dcs, carbon_price=carbon_price, p_init=p_prev)
        p_prev = r.p[:, n - 1]
        m = _commit_metrics(net, r, n)
        row = pd.DataFrame(m)
        row.insert(0, "hour", np.arange(t0, t0 + n))
        if envs:
            row["dc_energy"] = r.d[:, :n].sum(0)
            if plan is None:
                sh, fx, lt = r.dc_shed[:, :n], r.dc_flex[:, :n], r.dc_late[:, :n]
            else:
                sh = np.stack([p[0][:n] for p in plan])
                fx = np.stack([p[1][:n] for p in plan])
                lt = np.stack([p[2][:n] for p in plan])
            if mode in ("inflexible", "fixed") or levers == "shed":
                fx = np.stack([e.inflexible_batch[t0:t0 + n] for e in envs])
            row["dc_shed"], row["dc_flex"], row["dc_late"] = sh.sum(0), fx.sum(0), lt.sum(0)
            for k in range(len(envs)):
                served[k] += float(fx[k].sum())
                row[f"dc{k}_mw"] = r.d[k, :n]
                b = net.bus_index(envs[k].bus)
                row[f"dc{k}_lmp"] = r.lmp[b, :n]
                row[f"dc{k}_lme"] = r.lme[b, :n]
        rows.append(row)
        lmps.append(r.lmp[:, :n])
        lmes.append(r.lme[:, :n])
    hourly = pd.concat(rows, ignore_index=True)
    end = t_start + 24 * days
    backlog = sum(float(e.cum_arrived[end - 1]) - served[k] for k, e in enumerate(envs))
    info = dict(carbon_price=carbon_price, horizon=horizon, days=days, levers=levers,
                sites=",".join(str(e.bus) for e in envs), backlog_end_mwh=backlog)
    return YearRun(mode, hourly, np.concatenate(lmps, 1), np.concatenate(lmes, 1), info)


def signals_from(run: YearRun, net: Network, envs: list[Envelope], kind: str) -> list[np.ndarray]:
    """Day-ahead signals derived from the inflexible run, one per envelope, over the full year.

    Hours past the simulated span repeat the last value (they only feed lookahead).
    """
    H = net.n_hours
    h0 = int(run.hourly.hour.iloc[0])

    def full(x):
        out = np.full(H, x[-1])
        out[h0:h0 + len(x)] = x
        return out

    sigs = []
    for k, e in enumerate(envs):
        b = net.bus_index(e.bus)
        if kind == "price":
            sigs.append(full(run.lmp[b]))
        elif kind == "lme":
            sigs.append(full(run.lme[b]))
        elif kind == "avg_ci":
            load = net.load[:, h0:h0 + len(run.hourly)].sum(0) + run.hourly.dc_energy.to_numpy()
            sigs.append(full(run.hourly.emissions.to_numpy() / load))
        else:
            raise ValueError(kind)
    return sigs

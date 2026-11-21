"""Multi-period DC-OPF in PTDF form with flexible data-center loads, solved with HiGHS.

For each hour ``t`` in the horizon, the decision variables are:

* ``p[g,t]``   generator output (MW), ``pmin <= p <= avail``
* ``shed[n,t]`` involuntary load shedding at bus n (MW), priced at VOLL
* ``d[k,t]``   power drawn by data center k (MW), inside its envelope
* ``inj[n,t]`` net injection at bus n (free; defined by the bus-balance rows)

The constraints are:

* bus balance   ``sum_{g at n} p - inj + shed - sum_{k at n} d = load[n,t]``.
  The dual of this row is the LMP at bus n.
* system balance ``sum_n inj = 0``
* line limits   ``-F <= PTDF @ inj <= F``
* generator ramps ``|p[t] - p[t-1]| <= R`` (also against ``p_init`` when given)
* data-center ramps, and cumulative-energy bounds on ``sum_{tau<=t} d[k,tau]``.
  The cumulative bounds encode deferrable work with deadlines; see
  :mod:`gridflex.envelope`.

The objective is ``sum (cost + carbon_price * co2) p + VOLL * shed``.

Marginal prices come from the optimal basis (see :func:`basis_marginals`). The
same basis with emission rates in place of costs gives locational marginal
emissions (LMEs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .grid import Network

VOLL = 10_000.0  # $/MWh


@dataclass
class DataCenter:
    """A data-center load with a linear flexibility envelope over the horizon (arrays have length T).

    Power drawn is ``d = firm - shed + flex``, where

    * ``firm``: power at nominal operation, excluding deferrable work (MW)
    * ``shed`` in ``[0, shed_max]``: power-capping that spends SLO headroom (MW).
      Total ``sum(shed)`` is limited to ``shed_budget`` (MWh) and costs
      ``shed_cost`` $/MWh.
    * ``flex`` in ``[0, flex_max]``: deferrable (batch) work. Its cumulative
      energy ``sum_{tau<=t} flex`` must lie in ``[cum_min[t], cum_max[t]]``:
      no earlier than arrival, no later than the deadline. With ``late_cost``
      set, missing ``cum_min`` is allowed at ``late_cost`` $ per MWh-hour of
      lateness, and the lateness is reported.
    """

    bus: int                       # external bus id
    firm: np.ndarray
    shed_max: np.ndarray | None = None
    shed_budget: float = np.inf
    shed_cost: float = 0.0
    flex_max: np.ndarray | None = None
    cum_min: np.ndarray | None = None
    cum_max: np.ndarray | None = None
    late_cost: float | None = None
    ramp: float = np.inf           # MW/h on total power d
    name: str = "dc"

    def __post_init__(self):
        self.firm = np.asarray(self.firm, dtype=float)
        T = len(self.firm)
        z = np.zeros(T)
        self.shed_max = z if self.shed_max is None else np.asarray(self.shed_max, dtype=float)
        self.flex_max = z if self.flex_max is None else np.asarray(self.flex_max, dtype=float)

    @property
    def T(self) -> int:
        return len(self.firm)

    @classmethod
    def inflexible(cls, bus: int, load, name: str = "dc") -> "DataCenter":
        return cls(bus=bus, firm=np.asarray(load, dtype=float), name=name)


@dataclass
class OPFResult:
    status: str
    t0: int
    T: int
    objective: float
    p: np.ndarray            # (G, T)
    shed: np.ndarray         # (N, T)
    d: np.ndarray            # (K, T) total data-center power
    dc_shed: np.ndarray      # (K, T)
    dc_flex: np.ndarray      # (K, T)
    dc_late: np.ndarray      # (K, T) MWh of deferrable work behind its deadline
    inj: np.ndarray          # (N, T)
    flow: np.ndarray         # (L, T)
    lmp: np.ndarray          # (N, T) $/MWh
    lme: np.ndarray | None   # (N, T) tCO2/MWh
    gen_cost: float          # $ (excludes the carbon price and VOLL)
    emissions: float         # tCO2
    curtailment: float       # MWh of renewable energy available but not used
    congestion_rent: float   # $
    binding_lines: np.ndarray  # (L, T) bool
    info: dict = field(default_factory=dict)


class _LPBuilder:
    """Collects sparse rows and columns, then hands the model to HiGHS."""

    def __init__(self):
        self.n_col = 0
        self.lb, self.ub, self.c = [], [], []
        self.rows, self.cols, self.vals = [], [], []
        self.rlo, self.rhi = [], []
        self.n_row = 0

    def add_vars(self, n, lb, ub, c=0.0):
        idx = np.arange(self.n_col, self.n_col + n)
        self.lb.append(np.broadcast_to(lb, (n,)).astype(float))
        self.ub.append(np.broadcast_to(ub, (n,)).astype(float))
        self.c.append(np.broadcast_to(c, (n,)).astype(float))
        self.n_col += n
        return idx

    def add_rows(self, r, c, v, lo, hi, n_rows):
        """Add ``n_rows`` rows. ``r`` holds row offsets in ``[0, n_rows)``."""
        base = self.n_row
        self.rows.append(np.asarray(r) + base)
        self.cols.append(np.asarray(c))
        self.vals.append(np.asarray(v, dtype=float))
        self.rlo.append(np.broadcast_to(lo, (n_rows,)).astype(float))
        self.rhi.append(np.broadcast_to(hi, (n_rows,)).astype(float))
        self.n_row += n_rows
        return np.arange(base, base + n_rows)

    def matrix(self):
        A = sp.coo_matrix(
            (np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
            shape=(self.n_row, self.n_col),
        ).tocsc()
        A.sum_duplicates()
        return A, np.concatenate(self.c), np.concatenate(self.lb), np.concatenate(self.ub), np.concatenate(self.rlo), np.concatenate(self.rhi)


def solve_opf(
    net: Network,
    t0: int,
    T: int = 24,
    datacenters: list[DataCenter] | tuple = (),
    carbon_price: float = 0.0,
    p_init: np.ndarray | None = None,
    compute_lme: bool = True,
    allow_shed: bool = True,
) -> OPFResult:
    """Solve the multi-period DC-OPF for hours ``[t0, t0+T)``."""
    G, N, L, K = net.n_gen, net.n_bus, net.n_line, len(datacenters)
    hours = np.arange(t0, t0 + T)
    load = net.load[:, hours]
    avail = net.gen_avail[:, hours]
    inf = highspy.kHighsInf

    lp = _LPBuilder()
    gcost = net.gen_cost + carbon_price * net.gen_co2
    # Variables are laid out hour-major: [p(:,t) for t], then shed, d, inj.
    p_idx = lp.add_vars(G * T, np.minimum(np.repeat(net.gen_pmin[:, None], T, 1), avail).T.ravel(), avail.T.ravel(), np.tile(gcost, T)).reshape(T, G)
    shed_ub = load.T.ravel() if allow_shed else 0.0
    s_idx = lp.add_vars(N * T, 0.0, shed_ub, VOLL).reshape(T, N)
    if K:
        d_idx = lp.add_vars(K * T, 0.0, inf, 0.0).reshape(T, K)
        sh_idx = lp.add_vars(K * T, 0.0, np.stack([dc.shed_max for dc in datacenters], 1).ravel(),
                             np.tile([dc.shed_cost for dc in datacenters], T)).reshape(T, K)
        fx_idx = lp.add_vars(K * T, 0.0, np.stack([dc.flex_max for dc in datacenters], 1).ravel(), 0.0).reshape(T, K)
    else:
        d_idx = sh_idx = fx_idx = np.zeros((T, 0), dtype=int)
    late_idx = {}
    for k, dc in enumerate(datacenters):
        if dc.late_cost is not None and dc.cum_min is not None:
            late_idx[k] = lp.add_vars(T, 0.0, inf, dc.late_cost)
    i_idx = lp.add_vars(N * T, -inf, inf, 0.0).reshape(T, N)

    dc_bus = np.array([net.bus_index(dc.bus) for dc in datacenters], dtype=int)
    gb, gi = net.gen_bus, np.arange(G)

    # bus balance rows
    bus_rows = np.zeros((T, N), dtype=int)
    for t in range(T):
        r = np.concatenate([gb, np.arange(N), np.arange(N), dc_bus])
        c = np.concatenate([p_idx[t, gi], i_idx[t], s_idx[t], d_idx[t]])
        v = np.concatenate([np.ones(G), -np.ones(N), np.ones(N), -np.ones(K)])
        bus_rows[t] = lp.add_rows(r, c, v, load[:, t], load[:, t], N)
    # system balance
    for t in range(T):
        lp.add_rows(np.zeros(N, int), i_idx[t], np.ones(N), 0.0, 0.0, 1)
    # line limits (dense PTDF rows over injection vars)
    ptdf = net.ptdf
    nzr, nzc = np.nonzero(np.abs(ptdf) > 1e-10)
    line_rows = np.zeros((T, L), dtype=int)
    for t in range(T):
        line_rows[t] = lp.add_rows(nzr, i_idx[t, nzc], ptdf[nzr, nzc], -net.line_rating, net.line_rating, L)
    # generator ramps
    rg = np.flatnonzero(np.isfinite(net.gen_ramp))
    if len(rg):
        R = net.gen_ramp[rg]
        n = len(rg)
        for t in range(1, T):
            lp.add_rows(np.r_[np.arange(n), np.arange(n)], np.r_[p_idx[t, rg], p_idx[t - 1, rg]], np.r_[np.ones(n), -np.ones(n)], -R, R, n)
        if p_init is not None:
            lp.add_rows(np.arange(n), p_idx[0, rg], np.ones(n), p_init[rg] - R, p_init[rg] + R, n)
    # data centers: d = firm - shed + flex, ramps, shed budget, cumulative deferrable energy
    for k, dc in enumerate(datacenters):
        r = np.r_[np.arange(T), np.arange(T), np.arange(T)]
        lp.add_rows(r, np.r_[d_idx[:, k], sh_idx[:, k], fx_idx[:, k]], np.r_[np.ones(T), np.ones(T), -np.ones(T)], dc.firm, dc.firm, T)
        if np.isfinite(dc.ramp):
            for t in range(1, T):
                lp.add_rows([0, 0], [d_idx[t, k], d_idx[t - 1, k]], [1.0, -1.0], -dc.ramp, dc.ramp, 1)
        if np.isfinite(dc.shed_budget) and dc.shed_max.any():
            lp.add_rows(np.zeros(T, int), sh_idx[:, k], np.ones(T), -inf, dc.shed_budget, 1)
        if dc.cum_min is not None or dc.cum_max is not None:
            lo = dc.cum_min if dc.cum_min is not None else np.full(T, -inf)
            hi = dc.cum_max if dc.cum_max is not None else np.full(T, inf)
            r = np.concatenate([np.full(t + 1, t) for t in range(T)])
            c = np.concatenate([fx_idx[: t + 1, k] for t in range(T)])
            v = np.ones(len(c))
            if k in late_idx:  # sum flex + late[t] >= cum_min[t]; the upper bound applies to flex alone
                lp.add_rows(np.r_[r, np.arange(T)], np.r_[c, late_idx[k]], np.r_[v, np.ones(T)], lo, np.full(T, inf), T)
                lp.add_rows(r, c, v, np.full(T, -inf), hi, T)
            else:
                lp.add_rows(r, c, v, lo, hi, T)

    A, c, lb, ub, rlo, rhi = lp.matrix()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = A.shape[1], A.shape[0]
    model.col_cost_, model.col_lower_, model.col_upper_ = c, lb, ub
    model.row_lower_, model.row_upper_ = rlo, rhi
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = A.indptr
    model.a_matrix_.index_ = A.indices
    model.a_matrix_.value_ = A.data
    h.passModel(model)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status != "Optimal":
        raise RuntimeError(f"OPF not optimal: {status} (t0={t0})")
    sol = h.getSolution()
    x = np.asarray(sol.col_value)

    # marginals from the optimal basis
    basis = h.getBasis()
    col_basic = np.array([s == highspy.HighsBasisStatus.kBasic for s in basis.col_status])
    row_basic = np.array([s == highspy.HighsBasisStatus.kBasic for s in basis.row_status])
    lmp_all = basis_marginals(A, c, col_basic, row_basic)
    lme = None
    if compute_lme:
        e = np.zeros_like(c)
        e[p_idx.ravel()] = np.tile(net.gen_co2, T)
        lme = basis_marginals(A, e, col_basic, row_basic)[bus_rows].T

    p = x[p_idx].T
    shed = x[s_idx].T
    d = x[d_idx].T if K else np.zeros((0, T))
    dc_shed = x[sh_idx].T if K else np.zeros((0, T))
    dc_flex = x[fx_idx].T if K else np.zeros((0, T))
    dc_late = np.zeros((K, T))
    for k, idx in late_idx.items():
        dc_late[k] = x[idx]
    inj = x[i_idx].T
    flow = ptdf @ inj
    lmp = lmp_all[bus_rows].T
    # Congestion rent = what loads pay minus what generators are paid = -sum(LMP * net injection).
    congestion_rent = float(-(lmp * inj).sum())
    return OPFResult(
        status=status,
        t0=t0,
        T=T,
        objective=float(h.getInfo().objective_function_value),
        p=p,
        shed=shed,
        d=d,
        dc_shed=dc_shed,
        dc_flex=dc_flex,
        dc_late=dc_late,
        inj=inj,
        flow=flow,
        lmp=lmp,
        lme=lme,
        gen_cost=float(net.gen_cost @ p.sum(1)),
        emissions=float(net.gen_co2 @ p.sum(1)),
        curtailment=float((avail - p)[net.is_renewable].sum()),
        congestion_rent=congestion_rent,
        binding_lines=np.abs(flow) >= net.line_rating[:, None] * (1 - 1e-6),
        info=dict(n_col=A.shape[1], n_row=A.shape[0], nnz=A.nnz, highs_row_dual=np.asarray(sol.row_dual)),
    )


def basis_marginals(A: sp.csc_matrix, c: np.ndarray, col_basic: np.ndarray, row_basic: np.ndarray) -> np.ndarray:
    """Row marginals ``y`` (∂objective/∂rhs) for objective ``c`` at a fixed optimal basis.

    With the basis held fixed, the basic variables satisfy ``B x_B = b - N x_N``,
    so ``∂(c_B^T x_B)/∂b = y`` with ``B^T y = c_B``. Rows whose slack is basic
    are not binding, so their ``y`` is 0. Over the binding rows, ``y`` solves
    ``A[binding, basic_cols]^T y = c[basic_cols]``.

    With ``c`` set to cost this reproduces the LMPs. With ``c`` set to CO2
    rates it gives LMEs: the marginal tCO2 per MWh of extra load at each bus,
    given the current set of marginal units and binding constraints. Under
    degeneracy LMPs/LMEs are not unique. We return the one implied by HiGHS's
    final basis.
    """
    y = np.zeros(A.shape[0])
    R = np.flatnonzero(~row_basic)
    C = np.flatnonzero(col_basic)
    if len(R) == 0:
        return y
    if len(R) != len(C):
        raise RuntimeError(f"basis is not square: {len(R)} binding rows vs {len(C)} basic columns")
    M = A[R][:, C].T.tocsc()
    y[R] = spsolve(M, c[C])
    return y


def run_horizon(net: Network, t0: int, days: int, datacenters_fn=None, carbon_price: float = 0.0, compute_lme: bool = True):
    """Roll day-ahead (24 h) OPFs over ``days`` days, carrying ramp state across days.

    ``datacenters_fn(t0) -> list[DataCenter]`` builds each day's envelopes.
    """
    out, p_prev = [], None
    for d in range(days):
        s = t0 + 24 * d
        dcs = datacenters_fn(s) if datacenters_fn else []
        r = solve_opf(net, s, 24, dcs, carbon_price=carbon_price, p_init=p_prev, compute_lme=compute_lme)
        p_prev = r.p[:, -1]
        out.append(r)
    return out

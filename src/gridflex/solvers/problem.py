"""The joint fleet-grid dispatch problem as a differentiable, batched LP family (PyTorch, float64).

Every instance is a 24 h DC-OPF (see :mod:`gridflex.opf`) with one data
center at a fixed bus. Load, renewable availability and the DC envelope
vary between instances; the network and constraint structure do not.

Parametrisation (completion, in the DC3 sense):

* *Free* variables are rescaled to boxes ``s in [0,1]``: a generator schedule
  ``p_hat``, load shedding at every bus, DC power capping ``dsh``, DC
  deferrable power ``fx``, and DC lateness ``late``. A variable whose box has
  zero width (solar at night, say) just ignores its ``s``.
* *Dependent* quantities are completed from the equalities, so power balance
  holds by construction. DC power is ``d = firm - dsh + fx``. Each hour's
  system balance is closed by a **distributed slack**: every generator moves by
  the same fraction ``delta_t`` of its range, ``p = p_hat + width * delta_t``.
  Net injections then follow from bus balance. (Completing with one pivot
  variable instead concentrates the prediction error of ~5,000 variables in one
  bus; we tried it and it fails badly.) Every feasible dispatch is
  representable, with ``delta = 0``.
* All remaining constraints are affine in ``s``: generator bounds on ``p``,
  line limits, generator ramps, ``d >= 0``, deadline and arrival bounds on
  cumulative deferrable energy, and the daily shed budget. :meth:`DispatchFamily.constraints`
  returns them as ``f(s) <= 0``.

:meth:`DispatchFamily.anchor` builds a *strictly* interior point in closed
form (no LP solve). From it, :func:`gauge_map` maps any ``z`` in the open cube
exactly into the feasible set (Tabas & Zhang 2022; Li et al. 2023). Because
``f`` is affine, ``G y = f(s0 + y) - f(s0)``, so no constraint matrix is ever
formed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..grid import Network
from ..opf import VOLL

DT = torch.float64


@dataclass
class Batch:
    """Instance data, batched (B instances, T hours)."""

    load: torch.Tensor      # (B,T,N)
    avail: torch.Tensor     # (B,T,G)
    firm: torch.Tensor      # (B,T)
    shedmax: torch.Tensor   # (B,T)
    flexmax: torch.Tensor   # (B,T)
    cummin: torch.Tensor    # (B,T)
    cummax: torch.Tensor    # (B,T)
    budget: torch.Tensor    # (B,)
    lateub: torch.Tensor    # (B,)

    def __len__(self):
        return self.load.shape[0]

    def index(self, i) -> "Batch":
        return Batch(*(getattr(self, f)[i] for f in self.__dataclass_fields__))

    @staticmethod
    def cat(bs: list["Batch"]) -> "Batch":
        return Batch(*(torch.cat([getattr(b, f) for b in bs]) for f in Batch.__dataclass_fields__))


class DispatchFamily:
    def __init__(self, net: Network, dc_bus: int, T: int = 24, late_cost: float = 500.0, shed_cost: float = 0.0):
        self.net, self.T = net, T
        self.G, self.N, self.L = net.n_gen, net.n_bus, net.n_line
        self.k = net.bus_index(dc_bus)
        self.ptdf = torch.tensor(net.ptdf, dtype=DT)
        self.F = torch.tensor(net.line_rating, dtype=DT)
        self.cost = torch.tensor(net.gen_cost, dtype=DT)
        self.pmin = torch.tensor(net.gen_pmin, dtype=DT)
        self.gen_bus = torch.tensor(net.gen_bus)
        self.ramp_idx = torch.tensor(np.flatnonzero(np.isfinite(net.gen_ramp)))
        self.R = torch.tensor(net.gen_ramp[np.isfinite(net.gen_ramp)], dtype=DT)
        self.late_cost, self.shed_cost = late_cost, shed_cost
        # per-hour layout of s: [p_hat (G) | shed (N) | dsh | fx | late]
        self.n_h = self.G + self.N + 3
        self.n = self.T * self.n_h

    # ---------------------------------------------------------------- s <-> x
    def _split(self, s):
        B = s.shape[0]
        s = s.view(B, self.T, self.n_h)
        G, N = self.G, self.N
        return s[..., :G], s[..., G:G + N], s[..., G + N], s[..., G + N + 1], s[..., G + N + 2]

    def primal(self, s: torch.Tensor, b: Batch) -> dict:
        """Map ``s`` to every physical variable (MW), completing the equalities."""
        sp, ssh, sd, sf, sl = self._split(s)
        pmin = torch.minimum(self.pmin.expand_as(b.avail), b.avail)
        width = b.avail - pmin
        p_hat = pmin + width * sp
        shed = b.load * ssh
        dsh = b.shedmax * sd
        fx = b.flexmax * sf
        late = b.lateub[:, None] * sl
        d = b.firm - dsh + fx
        delta = (b.load.sum(-1) + d - p_hat.sum(-1) - shed.sum(-1)) / width.sum(-1)
        p = p_hat + width * delta[..., None]
        gen_at_bus = torch.zeros_like(b.load).index_add_(-1, self.gen_bus, p)
        inj = gen_at_bus + shed - b.load
        inj[..., self.k] -= d
        flow = inj @ self.ptdf.T
        return dict(p=p, pmin=pmin, width=width, shed=shed, dsh=dsh, fx=fx, late=late, d=d, delta=delta, inj=inj, flow=flow)

    def constraints(self, s: torch.Tensor, b: Batch, x: dict | None = None) -> torch.Tensor:
        """Affine inequality residuals ``f(s)`` (B, R). Feasible iff all are <= 0 (and s in [0,1])."""
        x = x or self.primal(s, b)
        B = s.shape[0]
        pr = x["p"][..., self.ramp_idx]
        dr = pr[:, 1:] - pr[:, :-1]
        cfx = x["fx"].cumsum(1)
        dead = (x["width"] <= 1e-9).to(DT)          # zero-width units: rows are identically 0; mark slack
        parts = [
            (x["pmin"] - x["p"] - dead).reshape(B, -1), (x["p"] - x["pmin"] - x["width"] - dead).reshape(B, -1),
            -x["d"],                                       # DC power >= 0
            (x["flow"] - self.F).reshape(B, -1), (-x["flow"] - self.F).reshape(B, -1),
            (dr - self.R).reshape(B, -1), (-dr - self.R).reshape(B, -1),
            cfx - b.cummax, b.cummin - cfx - x["late"],   # arrivals / deadlines (soft via late)
            (x["dsh"].sum(1) - b.budget)[:, None],
        ]
        return torch.cat(parts, 1)

    def objective(self, s: torch.Tensor, b: Batch, x: dict | None = None) -> torch.Tensor:
        x = x or self.primal(s, b)
        return ((x["p"] * self.cost).sum((1, 2)) + VOLL * x["shed"].sum((1, 2))
                + self.late_cost * x["late"].sum(1) + self.shed_cost * x["dsh"].sum(1))

    def from_opf(self, res, b: Batch) -> torch.Tensor:
        """Encode a (single-instance) :class:`gridflex.opf.OPFResult` as ``s`` (1, n)."""
        t = lambda a: torch.tensor(np.asarray(a), dtype=DT)
        pmin = torch.minimum(self.pmin.expand_as(b.avail), b.avail)[0]
        w = b.avail[0] - pmin
        sp = torch.where(w > 1e-9, (t(res.p.T) - pmin) / w.clamp_min(1e-12), torch.full_like(w, 0.5))
        ld = b.load[0]
        ssh = torch.where(ld > 1e-9, t(res.shed.T) / ld.clamp_min(1e-12), torch.full_like(ld, 0.5))
        div = lambda a, m: torch.where(m > 1e-9, a / m.clamp_min(1e-12), torch.full_like(m, 0.5))
        sd = div(t(res.dc_shed[0]), b.shedmax[0])
        sf = div(t(res.dc_flex[0]), b.flexmax[0])
        sl = t(res.dc_late[0]) / b.lateub[0]
        s = torch.cat([sp, ssh, sd[:, None], sf[:, None], sl[:, None]], 1)
        return s.reshape(1, -1).clamp(0.0, 1.0)

    # ---------------------------------------------------------------- interior anchor
    SHED_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)

    def _anchor_at(self, b: Batch, shed_frac: torch.Tensor) -> torch.Tensor:
        B, T = len(b), self.T
        pmin = torch.minimum(self.pmin.expand_as(b.avail), b.avail)
        width = b.avail - pmin
        dsh = 0.5 * b.shedmax * torch.clamp(b.budget / (b.shedmax.sum(1) + 1e-9), max=1.0)[:, None]
        dcm = torch.diff(b.cummax, dim=1, prepend=torch.zeros(B, 1, dtype=DT))
        fx = torch.minimum(0.5 * dcm, 0.5 * b.flexmax).clamp_min(0.0)
        d = b.firm - dsh + fx
        target = (1 - shed_frac[:, None]) * b.load.sum(-1) + d
        alpha = (target - pmin.sum(-1)) / width.sum(-1)
        sp = alpha[..., None].expand(B, T, self.G)
        ssh = shed_frac[:, None, None].expand(B, T, self.N)
        sd = torch.where(b.shedmax > 1e-9, dsh / b.shedmax.clamp_min(1e-12), torch.full_like(dsh, 0.5))
        sf = torch.where(b.flexmax > 1e-9, fx / b.flexmax.clamp_min(1e-12), torch.full_like(fx, 0.5))
        # lateness mid-box: far from both bounds, and the deadline rows get a large slack
        late = torch.maximum(0.5 * b.lateub[:, None].expand_as(fx), torch.relu(b.cummin - fx.cumsum(1)) + 1.0)
        sl = late / b.lateub[:, None]
        s0 = torch.cat([sp, ssh, sd[..., None], sf[..., None], sl[..., None]], -1).reshape(B, -1)
        wid = self.box_width(b)
        return torch.where(wid > 1e-9, s0, torch.full_like(s0, 0.5))

    def anchor(self, b: Batch, margin: float = 1e-6) -> torch.Tensor:
        """A strictly interior point, in closed form: the lowest-shedding member of a one-parameter family.

        For shed fraction ``phi``: shed ``phi`` of every bus's load, and serve the
        rest plus the DC with all generators at the same fraction of their range.
        This keeps net injections, and so line flows, small. The DC caps at half
        its budget, defers at half the arrival rate, and puts lateness mid-box.
        We try ``phi`` from small to large and keep, per instance, the smallest
        one that is strictly interior. Any pull toward the anchor is billed at
        the value of lost load, so a low-shedding anchor matters.
        """
        B = len(b)
        best = None
        chosen = torch.full((B,), float("nan"), dtype=DT)
        for phi in self.SHED_GRID:
            s0 = self._anchor_at(b, torch.full((B,), phi, dtype=DT))
            ok = (self.constraints(s0, b).amax(1) < -margin) & (torch.minimum(s0, 1 - s0).amin(1) > margin)
            take = ok & torch.isnan(chosen)
            best = s0 if best is None else torch.where(take[:, None], s0, best)
            chosen = torch.where(take, torch.full_like(chosen, phi), chosen)
            if not torch.isnan(chosen).any():
                break
        self.last_anchor_shed = chosen          # NaN = no interior member found (reported, never silent)
        return best

    def box_width(self, b: Batch) -> torch.Tensor:
        B, T = len(b), self.T
        pmin = torch.minimum(self.pmin.expand_as(b.avail), b.avail)
        w = torch.cat([b.avail - pmin, b.load, b.shedmax[..., None], b.flexmax[..., None],
                       b.lateub[:, None, None].expand(B, T, 1)], -1)
        return w.reshape(B, -1)


def gauge_map(family: DispatchFamily, z: torch.Tensor, s0: torch.Tensor, b: Batch, f0: torch.Tensor | None = None) -> torch.Tensor:
    """Map ``z`` in [-1,1]^n into the feasible polytope around the interior point ``s0``.

    ``s = s0 + (||z||_inf / gamma(z)) z``, where ``gamma`` is the gauge (Minkowski
    functional) of the polytope recentred at ``s0``. It includes the box
    ``0 <= s <= 1`` and all affine constraints. The output is feasible for every
    ``z``, which is the guarantee.
    """
    f0 = family.constraints(s0, b) if f0 is None else f0
    r = -f0                                                       # slack at the anchor, > 0
    Gz = family.constraints(s0 + z, b) - f0                       # affine: G z
    g_aff = (Gz / r).amax(1)
    g_box = torch.maximum(z / (1 - s0), -z / s0).amax(1)
    gamma = torch.maximum(g_aff, g_box).clamp_min(1e-12)
    scale = z.abs().amax(1) / gamma
    return s0 + scale[:, None] * z


def ray_project(family: DispatchFamily, s_hat: torch.Tensor, s0: torch.Tensor, b: Batch, f0: torch.Tensor | None = None,
                shrink: float = 1.0) -> torch.Tensor:
    """Exact feasibility by a closed-form line search toward an interior point.

    ``s = s0 + t (s_hat - s0)`` with the largest ``t <= 1`` that keeps every
    affine row and the box feasible: ``t = min(1, 1 / gauge(s_hat - s0))``. A
    feasible prediction is returned unchanged. An infeasible one is pulled back
    toward ``s0`` just enough. This is the gauge's ray form (the homeomorphic /
    bisection projection of Liang et al. 2023, solved in closed form because
    the constraints are affine).
    """
    f0 = family.constraints(s0, b) if f0 is None else f0
    y = s_hat - s0
    g_aff = ((family.constraints(s0 + y, b) - f0) / -f0).amax(1)
    g_box = torch.maximum(y / (1 - s0), -y / s0).amax(1)
    gamma = torch.maximum(g_aff, g_box)
    t = torch.clamp(1.0 / gamma.clamp_min(1e-300), max=1.0) * shrink
    return s0 + t[:, None] * y


def inverse_gauge(family: DispatchFamily, s: torch.Tensor, s0: torch.Tensor, b: Batch, f0: torch.Tensor | None = None) -> torch.Tensor:
    """The ``z`` that :func:`gauge_map` sends to a feasible ``s`` (training targets)."""
    f0 = family.constraints(s0, b) if f0 is None else f0
    y = s - s0
    g_aff = ((family.constraints(s0 + y, b) - f0) / -f0).amax(1)
    g_box = torch.maximum(y / (1 - s0), -y / s0).amax(1)
    gamma = torch.maximum(g_aff, g_box).clamp_min(1e-12)
    return y * (gamma / y.abs().amax(1).clamp_min(1e-12))[:, None]


def max_violation(family: DispatchFamily, s: torch.Tensor, b: Batch) -> torch.Tensor:
    """Largest constraint violation per instance, in the constraint's own units (MW / MWh)."""
    f = family.constraints(s, b)
    box = torch.maximum(s - 1, -s) * family.box_width(b)
    return torch.maximum(f.amax(1), box.amax(1)).clamp_min(0.0)


def closed_form_repair(family: DispatchFamily, s: torch.Tensor, b: Batch) -> torch.Tensor:
    """Exact repairs for rows with a one-variable fix (no effect on other rows).

    * lateness appears only in its own deadline row: ``late >= cummin - cumsum(fx)``
    * the daily shed budget: scale DC capping down uniformly if it is over budget
    """
    B, T, nh = s.shape[0], family.T, family.n_h
    s = s.reshape(B, T, nh).clone()
    G, N = family.G, family.N
    fx = b.flexmax * s[..., G + N + 1]
    need = torch.relu(b.cummin - fx.cumsum(1)) / b.lateub[:, None]
    s[..., G + N + 2] = torch.maximum(s[..., G + N + 2], need).clamp(max=1.0)
    dsh = (b.shedmax * s[..., G + N]).sum(1)
    scale = torch.clamp(b.budget / dsh.clamp_min(1e-12), max=1.0)
    s[..., G + N] = s[..., G + N] * scale[:, None]
    return s.reshape(B, -1)


def row_scale(family: DispatchFamily, b: Batch) -> torch.Tensor:
    """Per-row normalisation for violation penalties (MW rows by line rating / unit range, etc.)."""
    B, T = len(b), family.T
    pmin = torch.minimum(family.pmin.expand_as(b.avail), b.avail)
    w = (b.avail - pmin).clamp_min(1.0).reshape(B, -1)
    F = family.F.expand(B, T, -1).reshape(B, -1)
    R = family.R.expand(B, T - 1, -1).reshape(B, -1)
    one = torch.ones(B, T, dtype=DT) * 100.0
    return torch.cat([w, w, one, F, F, R, R, one, one, torch.full((B, 1), 100.0, dtype=DT)], 1)


def feasibility_seek(family: DispatchFamily, s: torch.Tensor, b: Batch, iters: int = 30, lr: float = 1.0,
                     anchor_weight: float = 0.0, restarts: int = 1) -> torch.Tensor:
    """Drive ``s`` toward the feasible set: L-BFGS on the normalised squared violation (FSNet-style).

    The box is kept by a clipped map with a pull-back penalty. L-BFGS's line
    search stalls at the hinge and clip kinks, so we restart it with fresh
    memory ``restarts`` times, each with ``iters`` iterations. The result is approximately feasible; follow it with
    :func:`ray_project` for exactness.
    """
    sc = row_scale(family, b)
    s_ref = s.detach()
    w = family.box_width(b).clamp_min(1e-6)     # optimise in MW units: v = s * width (much better conditioned)
    u = (s_ref * w).clone().requires_grad_(True)

    def closure():
        opt.zero_grad()
        sv = (u / w).clamp(0.0, 1.0)
        v = (torch.relu(family.constraints(sv, b) / sc) ** 2).sum()
        v = v + (((u / w) - sv) * w / 100.0).pow(2).sum()      # pull back into the box (MW/100)
        if anchor_weight:
            v = v + anchor_weight * (((sv - s_ref) * w / 100.0) ** 2).sum()
        v.backward()
        return v

    with torch.enable_grad():
        for _ in range(restarts):
            opt = torch.optim.LBFGS([u], lr=lr, max_iter=iters, history_size=50, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-14, tolerance_change=1e-16)
            opt.step(closure)
    return (u.detach() / w).clamp(0.0, 1.0)

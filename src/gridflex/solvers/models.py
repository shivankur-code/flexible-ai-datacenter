"""Learned solvers for :class:`~gridflex.solvers.problem.DispatchFamily`.

* :class:`GaugeSolver`: an MLP predicts ``z in (-1,1)^n``, and
  :func:`~gridflex.solvers.problem.gauge_map` sends it to a point that is
  feasible by construction. Training is supervised on the inverse-gauge
  targets of the HiGHS optima, then fine-tuned self-supervised on the LP
  objective (no labels needed).
* :class:`DC3Solver`: DC3-style (Donti, Rolnick & Kolter, ICLR 2021). The MLP
  predicts ``s``, equalities hold by completion, and ``k`` unrolled gradient
  steps on the squared inequality violation correct it. Feasibility is
  approximate. Repeated correction at test time drives violations down but
  does not guarantee zero.
* :class:`ProjectedSolver` (ours): the DC3 prediction and correction, followed
  by :func:`~gridflex.solvers.problem.ray_project` onto the segment toward a
  closed-form interior anchor. It is exactly feasible, and because the
  correction leaves little violation, the final retraction is tiny.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .problem import DT, Batch, DispatchFamily, gauge_map, inverse_gauge, max_violation, ray_project


def features(fam: DispatchFamily, b: Batch) -> torch.Tensor:
    ren = torch.tensor(fam.net.is_renewable)
    pmax = torch.tensor(fam.net.gen_pmax, dtype=DT)[ren]
    B = len(b)
    return torch.cat([
        b.load.reshape(B, -1) / 100.0,
        (b.avail[..., ren] / pmax).reshape(B, -1),
        torch.cat([b.firm, b.shedmax, b.flexmax, b.cummin, b.cummax], 1) / 100.0,
        b.budget[:, None] / 100.0,
    ], 1)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 1024, depth: int = 2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers).to(DT)

    def forward(self, x):
        return self.net(x)


class _Base(nn.Module):
    def __init__(self, fam: DispatchFamily, train_b: Batch, hidden: int = 1024, depth: int = 2):
        super().__init__()
        self.fam = fam
        x = features(fam, train_b)
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0).clamp_min(1e-6))
        self.mlp = MLP(x.shape[1], fam.n, hidden, depth)

    def raw(self, b: Batch) -> torch.Tensor:
        return self.mlp((features(self.fam, b) - self.mu) / self.sd)


class GaugeSolver(_Base):
    def forward(self, b: Batch, s0=None, f0=None) -> torch.Tensor:
        s0 = self.fam.anchor(b) if s0 is None else s0
        z = torch.tanh(self.raw(b))
        return gauge_map(self.fam, z, s0, b, f0)


def to_box(raw: torch.Tensor, pad: float = 0.05) -> torch.Tensor:
    """Map logits to [0,1] so that exact 0 and 1 are reachable. LP optima sit on bounds,
    and a plain sigmoid's ~0.5% floor on shedding costs ~VOLL x 0.5% of load."""
    return ((1 + 2 * pad) * torch.sigmoid(raw) - pad).clamp(0.0, 1.0)


class DC3Solver(_Base):
    def __init__(self, *a, k_train: int = 5, k_test: int = 50, lr_corr: float = 1e-3, **kw):
        super().__init__(*a, **kw)
        self.k_train, self.k_test, self.lr_corr = k_train, k_test, lr_corr

    def correct(self, s: torch.Tensor, b: Batch, k: int) -> torch.Tensor:
        """Unrolled gradient steps on the squared violation (in MW units), then clip to the box."""
        w = self.fam.box_width(b)
        for _ in range(k):
            v = (torch.relu(self.fam.constraints(s, b)) ** 2).sum()
            g, = torch.autograd.grad(v, s, create_graph=self.training)
            # precondition by box width: a step in s moves x by width * step
            s = (s - self.lr_corr * g / (w ** 2 + 1.0)).clamp(0.0, 1.0)
        return s

    def forward(self, b: Batch) -> torch.Tensor:
        s = to_box(self.raw(b))
        if not s.requires_grad:
            s = s.detach().requires_grad_(True)
        with torch.enable_grad():
            return self.correct(s, b, self.k_train if self.training else self.k_test)


class ProjectedSolver(nn.Module):
    """Wraps a trained :class:`DC3Solver`: predict, correct, then ray-project for exact feasibility."""

    def __init__(self, dc3: DC3Solver):
        super().__init__()
        self.dc3, self.fam = dc3, dc3.fam

    def forward(self, b: Batch) -> torch.Tensor:
        s = self.dc3(b)
        s0 = self.fam.anchor(b)
        return ray_project(self.fam, s.detach(), s0, b)


def _batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_gauge(model: GaugeSolver, b: Batch, s_star: torch.Tensor, obj_star: torch.Tensor, epochs_sup: int = 150,
                epochs_ssl: int = 100, bs: int = 64, lr: float = 1e-3, seed: int = 0, log=print):
    fam = model.fam
    gen = torch.Generator().manual_seed(seed)
    s0 = fam.anchor(b)
    f0 = fam.constraints(s0, b)
    z_star = inverse_gauge(fam, s_star, s0, b, f0).clamp(-0.999, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for ep in range(epochs_sup + epochs_ssl):
        sup = ep < epochs_sup
        if ep == epochs_sup:
            opt = torch.optim.AdamW(model.parameters(), lr=lr / 5, weight_decay=1e-5)
        tot = 0.0
        for i in _batches(len(b), bs, gen):
            bi = b.index(i)
            if sup:
                z = torch.tanh(model.raw(bi))
                loss = ((z - z_star[i]) ** 2).mean()
            else:
                s = gauge_map(fam, torch.tanh(model.raw(bi)), s0[i], bi, f0[i])
                loss = (fam.objective(s, bi) / obj_star[i]).mean() - 1.0
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(i)
        hist.append(tot / len(b))
        if ep % 25 == 0 or ep == epochs_sup + epochs_ssl - 1:
            log(f"  gauge ep {ep:4d} {'sup' if sup else 'ssl'} loss {hist[-1]:.5f}")
    return hist


def train_dc3(model: DC3Solver, b: Batch, s_star: torch.Tensor, obj_star: torch.Tensor, epochs_sup: int = 150,
              epochs_ssl: int = 50, bs: int = 64, lr: float = 1e-3, rho: float = 10.0, seed: int = 0, log=print):
    """Supervised on s*, then DC3's self-supervised loss: objective + rho * squared violation, through the unrolled correction."""
    fam = model.fam
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for ep in range(epochs_sup + epochs_ssl):
        sup = ep < epochs_sup
        if ep == epochs_sup:
            opt = torch.optim.AdamW(model.parameters(), lr=lr / 5, weight_decay=1e-5)
        tot = 0.0
        model.train()
        for i in _batches(len(b), bs, gen):
            bi = b.index(i)
            if sup:
                loss = ((to_box(model.raw(bi)) - s_star[i]) ** 2).mean()
            else:
                s = model(bi)
                viol = (torch.relu(fam.constraints(s, bi)) ** 2).sum(1)
                loss = (fam.objective(s, bi) / obj_star[i]).mean() - 1.0 + rho * (viol / obj_star[i]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(i)
        hist.append(tot / len(b))
        if ep % 25 == 0 or ep == epochs_sup + epochs_ssl - 1:
            log(f"  dc3   ep {ep:4d} {'sup' if sup else 'ssl'} loss {hist[-1]:.5f}")
    model.eval()
    return hist


@torch.no_grad()
def evaluate(model, b: Batch, obj_star: torch.Tensor, name: str, repeats: int = 3) -> dict:
    """Gap, violation and wall time per instance (the whole set in one batch, best of ``repeats``)."""
    model.eval()
    fam = model.fam
    times = []
    for _ in range(repeats):
        tic = time.perf_counter()
        s = model(b)
        times.append(time.perf_counter() - tic)
    obj = fam.objective(s, b)
    x = fam.primal(s, b)
    gap = (obj - obj_star) / obj_star
    viol = max_violation(fam, s, b)
    return dict(method=name, n=len(b), gap_mean=float(gap.mean()), gap_median=float(gap.median()),
                gap_p95=float(gap.quantile(0.95)), viol_max=float(viol.max()), viol_mean=float(viol.mean()),
                feasible_frac=float((viol <= 1e-6).float().mean()), shed_mwh_mean=float(x["shed"].sum((1, 2)).mean()),
                ms_per_instance=1e3 * min(times) / len(b)), s


# ---------------------------------------------------------------- learned constraint screening
class ScreenNet(_Base):
    """Predicts which (hour, line) limits bind. Its output decides which line rows the exact LP starts with.

    Exactness never depends on it: :func:`gridflex.opf.solve_opf_screened`
    adds back any violated limit and re-solves. A good screen only removes
    rows and rounds.
    """

    def __init__(self, fam: DispatchFamily, train_b: Batch, hidden: int = 512, depth: int = 2):
        nn.Module.__init__(self)
        self.fam = fam
        x = features(fam, train_b)
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sd", x.std(0).clamp_min(1e-6))
        self.mlp = MLP(x.shape[1], fam.T * fam.L, hidden, depth)
        self.threshold = 0.5

    def logits(self, b: Batch) -> torch.Tensor:
        return self.raw(b)

    @torch.no_grad()
    def predict(self, b: Batch) -> torch.Tensor:
        return (torch.sigmoid(self.logits(b)) >= self.threshold).reshape(len(b), self.fam.T, self.fam.L)


def binding_labels(fam: DispatchFamily, s_star: torch.Tensor, b: Batch, tol: float = 1e-4) -> torch.Tensor:
    flow = fam.primal(s_star, b)["flow"]
    return (flow.abs() >= fam.F - tol).to(DT)            # (B,T,L)


def train_screen(model: ScreenNet, b: Batch, y: torch.Tensor, epochs: int = 150, bs: int = 64, lr: float = 1e-3,
                 recall_target: float = 0.999, val_frac: float = 0.15, seed: int = 0, log=print):
    """BCE with positive up-weighting on most of ``b``. Then calibrate the threshold on a held-out slice for ``recall_target``.

    Missing a binding row costs a whole extra LP round, while monitoring an
    extra row costs ~0.07 ms, so the calibration aims for very high recall on
    unseen instances.
    """
    gen = torch.Generator().manual_seed(seed)
    n = len(b)
    perm = torch.randperm(n, generator=gen)
    n_val = max(int(val_frac * n), 1)
    vi, ti = perm[:n_val], perm[n_val:]
    bt, bv = b.index(ti), b.index(vi)
    yf = y.reshape(n, -1)
    yt, yv = yf[ti], yf[vi]
    pos = yt.mean().clamp_min(1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).sqrt())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        tot = 0.0
        for i in _batches(len(bt), bs, gen):
            loss = loss_fn(model.logits(bt.index(i)), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(i)
        if ep % 25 == 0 or ep == epochs - 1:
            log(f"  screen ep {ep:4d} bce {tot / len(bt):.5f}")
    with torch.no_grad():
        p = torch.sigmoid(model.logits(bv)).reshape(-1)
    pos_p = p[yv.reshape(-1) > 0.5].sort().values
    k = int((1 - recall_target) * len(pos_p))
    model.threshold = float(pos_p[k]) if len(pos_p) else 0.5
    kept = float((p >= model.threshold).float().sum() / len(bv))
    log(f"  screen threshold {model.threshold:.2e}: validation recall >= {recall_target}, {kept:.0f} rows monitored per instance")
    return model.threshold

"""Instance generation for the learned solvers: days x data-center envelopes, labelled by HiGHS."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..envelope import Envelope
from ..grid import Network
from ..opf import DataCenter, solve_opf
from .problem import DT, Batch, DispatchFamily


def batch_from(net: Network, dc: DataCenter, t0: int, T: int = 24) -> Batch:
    h = slice(t0, t0 + T)
    t = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=DT)[None]
    cummax = dc.cum_max if dc.cum_max is not None else np.full(T, 1e6)
    cummin = dc.cum_min if dc.cum_min is not None else np.zeros(T)
    return Batch(
        load=t(net.load[:, h].T), avail=t(net.gen_avail[:, h].T), firm=t(dc.firm), shedmax=t(dc.shed_max),
        flexmax=t(dc.flex_max), cummin=t(cummin), cummax=t(cummax),
        budget=torch.tensor([min(dc.shed_budget, 1e9)], dtype=DT),
        lateub=torch.tensor([2.0 * float(np.max(cummax)) + 100.0], dtype=DT),
    )


def make_dataset(net: Network, family: DispatchFamily, envs: list[Envelope], days: list[int], T: int = 24) -> dict:
    """Solve every (day, envelope) pair with HiGHS. Return batched instances, optimal ``s``, objectives and timings."""
    bs, ss, objs, t_total, t_solve, meta = [], [], [], [], [], []
    for d in days:
        t0 = 24 * d
        for j, env in enumerate(envs):
            dc = env.window(t0, T, "flexible")
            b = batch_from(net, dc, t0, T)
            tic = time.perf_counter()
            r = solve_opf(net, t0, T, [dc], compute_lme=False)
            t_total.append(time.perf_counter() - tic)
            t_solve.append(r.info.get("solve_s", np.nan))
            bs.append(b)
            ss.append(family.from_opf(r, b))
            objs.append(r.objective)
            meta.append((d, j))
    return dict(batch=Batch.cat(bs), s_star=torch.cat(ss), obj_star=torch.tensor(objs, dtype=DT),
                highs_total_s=np.array(t_total), highs_solve_s=np.array(t_solve), meta=np.array(meta))

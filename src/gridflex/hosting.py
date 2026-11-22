"""Flexible-interconnection map (RQ5): the minimum SLO-feasible flexibility for hosting an AI data center at a bus.

A facility of size S at bus b is *hostable at level L* if, on every evaluated
day, the grid can serve it with no involuntary load shedding anywhere and
without missing any batch deadline, when the operator co-optimises the
facility's level-L envelope. Each stress day is solved as a 48 h window (the
day and the next), starting from the backlog an as-soon-as-possible operator
would carry, and the whole window is scored, so deferred work cannot vanish
past midnight. Solves are exact: HiGHS with line-constraint generation,
warm-started from the previous level's binding set.

Flexibility ladder. Energy is the same at every level; only what may move
changes:

    L0  inflexible
    L1  power capping within the SLO (shed, 4 h/day budget)
    L2  L1 + batch deferral, 4 h deadline
    L3  L1 + batch deferral, 12 h deadline
    L4  L1 + batch deferral, 24 h deadline
    L5  L4, plus capping allowed down to a 2x-relaxed latency SLO (same fleet,
        traffic and energy; only the capping headroom grows)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .envelope import build_envelope
from .grid import Network
from .opf import solve_opf_screened

LEVELS = (
    dict(name="L0 inflexible", mode="inflexible", deadline_h=24, slo_scale=1.0),
    dict(name="L1 power-cap", mode="shed", deadline_h=24, slo_scale=1.0),
    dict(name="L2 + defer 4 h", mode="flexible", deadline_h=4, slo_scale=1.0),
    dict(name="L3 + defer 12 h", mode="flexible", deadline_h=12, slo_scale=1.0),
    dict(name="L4 + defer 24 h", mode="flexible", deadline_h=24, slo_scale=1.0),
    dict(name="L5 + SLO x2", mode="flexible", deadline_h=24, slo_scale=2.0),
)
NOT_HOSTABLE = len(LEVELS)


def stress_days(net: Network, per_month: int = 4) -> list[int]:
    """Each month's ``per_month`` most stressful days (alternating highest load and highest net load), most stressful first.

    Validation (``experiments/p5_map.py --validate``) showed 2 per month misses
    multi-day stress spells for marginal flexible sites, so the default is 4.
    """
    load = net.load.sum(0)
    ren = net.gen_avail[net.is_renewable].sum(0)
    days = net.n_hours // 24
    dl = load[: days * 24].reshape(days, 24).max(1)
    dn = (load - ren)[: days * 24].reshape(days, 24).max(1)
    month = np.array([net.timestamps[24 * d].month for d in range(days)])
    picks = []
    for m in range(1, 13):
        idx = np.flatnonzero(month == m)
        idx = idx[idx < days - 1]  # need a lookahead day
        by_l, by_n = list(idx[np.argsort(-dl[idx])]), list(idx[np.argsort(-dn[idx])])
        chosen = []
        while len(chosen) < per_month and (by_l or by_n):
            for lst in (by_l, by_n):
                while lst and lst[0] in chosen:
                    lst.pop(0)
                if lst and len(chosen) < per_month:
                    chosen.append(lst.pop(0))
        picks += [int(c) for c in chosen]
    picks = list(dict.fromkeys(picks))
    return sorted(picks, key=lambda d: -dn[d])


def scaled_slo(cfg: dict, scale: float) -> dict:
    c = copy.deepcopy(cfg)
    for svc in c["slo"]:
        c["slo"][svc] = {k: v * scale for k, v in c["slo"][svc].items()}
    return c


@dataclass
class DayCheck:
    day: int
    grid_shed_mwh: float
    late_mwh_h: float
    dc_shed_mwh: float
    ok: bool


def check_level(net: Network, env, level: dict, days: list[int], tol: float = 0.1, masks: dict | None = None,
                stop_early: bool = True) -> list[DayCheck]:
    """Solve each day (48 h window, first 24 h scored) for one envelope and level."""
    out = []
    masks = {} if masks is None else masks
    for d in days:
        t0 = 24 * d
        T = min(48, net.n_hours - t0)
        # the stress day starts with the backlog an as-soon-as-possible operator would carry (not a
        # worst-case "everything held to its deadline" backlog, which would penalise long deadlines)
        dc = env.window(t0, T, level["mode"], served_before=float(env.inflexible_batch[:t0].sum()))
        r = solve_opf_screened(net, t0, T, [dc], initial=masks.get(d), compute_lme=False)
        masks[d] = r.binding_lines.T.copy() | (masks[d] if d in masks else False)
        shed = float(r.shed.sum())                 # score the whole 48 h: deferred work must land somewhere
        late = float(r.dc_late.sum())
        ok = shed <= tol and late <= tol
        out.append(DayCheck(d, shed, late, float(r.dc_shed.sum()), ok))
        if stop_early and not ok:
            break
    return out


def min_level(net: Network, envs_by_level: list, days: list[int], start: int = 0, tol: float = 0.1) -> tuple[int, list]:
    """Lowest level, at or above ``start``, whose envelope is hostable on every day. Returns (level, per-level checks)."""
    masks: dict = {}
    log = []
    for L in range(start, len(LEVELS)):
        checks = check_level(net, envs_by_level[L], LEVELS[L], days, tol, masks)
        log.append((L, checks))
        if all(c.ok for c in checks) and len(checks) == len(days):
            return L, log
    return NOT_HOSTABLE, log


def envelopes_for(cfg: dict, tables: dict, bus: int, size_mw: float, net: Network, yp: dict, batch_fill: float = 0.5) -> list:
    """One envelope per ladder level (sharing the facility and workload; only deadline / SLO differ)."""
    it, bt = tables[1.0]
    envs = []
    for lv in LEVELS:
        cap = tables[lv["slo_scale"]][0] if lv["slo_scale"] != 1.0 else None
        envs.append(build_envelope(cfg, bus, size_mw, net.timestamps, yp, it, bt, batch_fill=batch_fill,
                                   deadline_h=lv["deadline_h"], cap_table=cap))
    return envs

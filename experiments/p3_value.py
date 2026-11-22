"""Phase 3: the grid value of SLO-grounded data-center flexibility (RQ2) and the price-following paradox (RQ3).

For each site (one 300 MW LLM-inference facility at a time) we roll a full
year of 48 h-lookahead DC-OPFs on RTS-GMLC with lines derated to 0.6x, under:
none / inflexible / price / avg_ci / lme / coopt. At the main site we add
lever decompositions (shift-only, shed-only) and co-optimisation with a carbon
price. Congestion cost = cost on the real network minus cost of the same loads
on a copper plate (ratings x1000).

    python experiments/p3_value.py                     # full year, 5 sites, 4 workers (~45 min)
    python experiments/p3_value.py --days 14 --sites 313 --no-copper   # smoke test

Writes results/p3/runs/*.csv.gz (hourly) and results/p3/totals.csv.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p3"
RUNS = OUT / "runs"


@lru_cache(maxsize=None)
def _setup(scale: float, facility_mw: float, sites: tuple):
    from gridflex.envelope import build_envelope
    from gridflex.fleet import cached_table, load_config
    from gridflex.grid import load_rts_gmlc
    from gridflex.workload import load_tokens, weekly_profile, year_profile

    cfg = load_config()
    it, bt = cached_table(cfg, load_tokens())
    base = load_rts_gmlc()
    yp = year_profile(base.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    envs = {s: build_envelope(cfg, s, facility_mw, base.timestamps, yp, it, bt) for s in sites}
    return base, envs


def run_task(task: dict) -> dict:
    """One year run. ``task`` keys: name, site, mode, [levers, carbon_price, copper, signal_from, fixed_from]."""
    from gridflex.modes import run_year

    base, envs = _setup(task["scale"], task["facility_mw"], tuple(task["all_sites"]))
    net = base.with_ratings(1000.0 if task.get("copper") else task["scale"])
    env = [envs[task["site"]]] if task["site"] is not None else []
    kw = dict(days=task["days"], t_start=task["t_start"], horizon=task["horizon"])
    mode = task["mode"]
    t = time.time()
    if task.get("fixed_from"):
        if env:
            h = pd.read_csv(RUNS / f"{task['fixed_from']}.csv.gz")
            load = np.zeros(net.n_hours)
            load[h.hour.to_numpy()] = h.dc0_mw.to_numpy()
            r = run_year(net, env, "fixed", fixed_loads=[load], **kw)
        else:
            r = run_year(net, [], "none", **kw)
        r.mode = task["name"]
    else:
        signals = None
        if mode in ("price", "avg_ci", "lme"):
            h = pd.read_csv(RUNS / f"{task['signal_from']}.csv.gz")
            col = {"price": "dc0_lmp", "lme": "dc0_lme"}.get(mode)
            sig = np.zeros(net.n_hours)
            if col:
                x = h[col].to_numpy()
            else:  # system average emission rate, including the DC load
                x = h.emissions.to_numpy() / (base.load[:, h.hour.to_numpy()].sum(0) + h.dc_energy.to_numpy())
            sig[:] = x[-1]
            sig[h.hour.to_numpy()] = x
            signals = [sig]
        r = run_year(net, env, mode, signals=signals, levers=task.get("levers", "flexible"),
                     carbon_price=task.get("carbon_price", 0.0), **kw)
    r.hourly.to_csv(RUNS / f"{task['name']}.csv.gz", index=False)
    tot = r.totals()
    tot.update(name=task["name"], site=task["site"], copper=bool(task.get("copper")), wall_s=time.time() - t,
               base_mode=mode, scale=task["scale"])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, nargs="+", default=[101, 122, 207, 303, 313])
    ap.add_argument("--main-site", type=int, default=313)
    ap.add_argument("--facility-mw", type=float, default=300.0)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--days", type=int, default=364)
    ap.add_argument("--t-start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--carbon-price", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-copper", action="store_true")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    common = dict(scale=args.scale, facility_mw=args.facility_mw, all_sites=args.sites, days=args.days,
                  t_start=args.t_start, horizon=args.horizon)
    T = lambda **k: {**common, **k}

    stage1 = [T(name="none", site=None, mode="none")] + [T(name=f"{s}_inflexible", site=s, mode="inflexible") for s in args.sites]
    stage2 = []
    for s in args.sites:
        for m in ("price", "avg_ci", "lme", "coopt"):
            stage2.append(T(name=f"{s}_{m}", site=s, mode=m, signal_from=f"{s}_inflexible"))
    s = args.main_site
    stage2 += [
        T(name=f"{s}_coopt_shift", site=s, mode="coopt", levers="shift"),
        T(name=f"{s}_coopt_shed", site=s, mode="coopt", levers="shed"),
        T(name=f"{s}_price_shift", site=s, mode="price", levers="shift", signal_from=f"{s}_inflexible"),
        T(name=f"{s}_avg_ci_shift", site=s, mode="avg_ci", levers="shift", signal_from=f"{s}_inflexible"),
    ]
    cp = args.carbon_price
    stage2b = [T(name=f"{s}_inflexible_cp{cp:g}", site=s, mode="inflexible", carbon_price=cp)]
    stage2c = [T(name=f"{s}_{m}_cp{cp:g}", site=s, mode=m, carbon_price=cp, signal_from=f"{s}_inflexible_cp{cp:g}")
               for m in ("price", "lme", "coopt")]
    stage3 = []
    if not args.no_copper:
        stage3 = [T(name="none_copper", site=None, mode="none", copper=True, fixed_from="none")]
        for s in args.sites:
            for m in ("inflexible", "price", "avg_ci", "lme", "coopt"):
                stage3.append(T(name=f"{s}_{m}_copper", site=s, mode="fixed", copper=True, fixed_from=f"{s}_{m}"))

    totals = []
    with ProcessPoolExecutor(args.workers) as ex:
        for stage, tasks in (("1", stage1), ("2", stage2 + stage2b), ("2c", stage2c), ("3", stage3)):
            t = time.time()
            for tot in ex.map(run_task, tasks):
                totals.append(tot)
                print(f"  stage {stage}: {tot['name']:<24} {tot['wall_s']:6.0f}s", flush=True)
            print(f"stage {stage} done in {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(totals)
    df.to_csv(OUT / "totals.csv", index=False)
    print(df.drop(columns=["sites"], errors="ignore").round(1).to_string())


if __name__ == "__main__":
    main()

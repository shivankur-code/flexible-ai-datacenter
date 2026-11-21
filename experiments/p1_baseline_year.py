"""Phase 1: the no-data-center baseline year on RTS-GMLC.

Rolls 366 day-ahead DC-OPFs at each line-rating scale. Writes daily metrics
and per-bus annual LMP/LME to results/p1/, and prints an annual summary.

    python experiments/p1_baseline_year.py [--scales 1.0 0.6] [--days 366]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridflex.grid import load_rts_gmlc
from gridflex.opf import run_horizon

OUT = Path(__file__).resolve().parents[1] / "results" / "p1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.6])
    ap.add_argument("--days", type=int, default=366)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_rts_gmlc()
    summary = []
    for s in args.scales:
        net = base.with_ratings(s)
        t = time.time()
        res = run_horizon(net, 0, args.days)
        wall = time.time() - t
        daily = pd.DataFrame(
            dict(
                date=[net.timestamps[r.t0].date() for r in res],
                gen_cost=[r.gen_cost for r in res],
                emissions_t=[r.emissions for r in res],
                curtailment_mwh=[r.curtailment for r in res],
                congestion_rent=[r.congestion_rent for r in res],
                binding_line_hours=[int(r.binding_lines.sum()) for r in res],
                shed_mwh=[r.shed.sum() for r in res],
            )
        )
        daily.to_csv(OUT / f"daily_ratings{s:g}.csv", index=False)
        lmp = np.concatenate([r.lmp for r in res], 1)
        lme = np.concatenate([r.lme for r in res], 1)
        pd.DataFrame(
            dict(bus=net.bus_ids, area=net.bus_area, lmp_mean=lmp.mean(1), lmp_std=lmp.std(1),
                 lme_mean=lme.mean(1), lme_std=lme.std(1))
        ).to_csv(OUT / f"bus_marginals_ratings{s:g}.csv", index=False)
        load = net.load[:, : 24 * args.days].sum()
        summary.append(
            dict(
                ratings=s,
                days=args.days,
                load_twh=load / 1e6,
                gen_cost_musd=daily.gen_cost.sum() / 1e6,
                emissions_mt=daily.emissions_t.sum() / 1e6,
                avg_ci_t_per_mwh=daily.emissions_t.sum() / load,
                curtailment_twh=daily.curtailment_mwh.sum() / 1e6,
                congestion_rent_musd=daily.congestion_rent.sum() / 1e6,
                binding_line_hours=int(daily.binding_line_hours.sum()),
                shed_mwh=daily.shed_mwh.sum(),
                lmp_spread_mean=float((lmp.max(0) - lmp.min(0)).mean()),
                lme_mean=float(lme.mean()),
                lme_spread_mean=float((lme.max(0) - lme.min(0)).mean()),
                wall_s=wall,
            )
        )
    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()

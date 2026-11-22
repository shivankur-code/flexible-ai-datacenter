"""Phase 5: the flexible-interconnection map (RQ5, Figure 1).

For every bus of RTS-GMLC and facility sizes 100/250/500/1000 MW: the minimum
rung of the SLO-feasible flexibility ladder (see gridflex.hosting) at which an
LLM-inference campus can connect without any involuntary load shedding or
missed batch deadline, on the year's stress days (lines at 0.6x by default).
Every solve is exact (HiGHS with line-constraint generation).

    python experiments/p5_map.py [--ratings 0.6] [--workers 4]
    python experiments/p5_map.py --verify        # stage 2: full-year verification of every flexible entry (Fig. 1)
    python experiments/p5_map.py --validate      # full-year check of selected map entries

Writes results/p5/map_ratings*.csv and paper/figures/fig1_flex_interconnection_map.{png,pdf}.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p5"
SIZES = (100.0, 250.0, 500.0, 1000.0)


@lru_cache(maxsize=None)
def _setup(ratings: float):
    from gridflex.fleet import cached_table, load_config
    from gridflex.grid import load_rts_gmlc
    from gridflex.hosting import scaled_slo, stress_days
    from gridflex.workload import load_tokens, weekly_profile, year_profile

    cfg = load_config()
    tok = load_tokens()
    tables = {1.0: cached_table(cfg, tok), 2.0: cached_table(scaled_slo(cfg, 2.0), tok)}
    net = load_rts_gmlc().with_ratings(ratings)
    yp = year_profile(net.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    return cfg, tables, net, yp, stress_days(net)


def map_bus(args):
    bus, ratings = args
    from gridflex.hosting import LEVELS, NOT_HOSTABLE, envelopes_for, min_level

    cfg, tables, net, yp, days = _setup(ratings)
    rows, start = [], 0
    t = time.time()
    for size in SIZES:
        if start == NOT_HOSTABLE:  # monotone in size: a bigger campus is not hostable either
            rows.append(dict(bus=bus, size_mw=size, level=NOT_HOSTABLE, inferred=True))
            continue
        envs = envelopes_for(cfg, tables, bus, size, net, yp)
        L, log = min_level(net, envs, days, start=start)
        worst = {l: max((c.grid_shed_mwh for c in ch), default=0.0) for l, ch in log}
        rows.append(dict(bus=bus, size_mw=size, level=L, inferred=False,
                         shed_at_start_mwh=worst.get(start, np.nan),
                         levels_tried=len(log), days_solved=sum(len(ch) for _, ch in log)))
        start = L
    for r in rows:
        r["level_name"] = LEVELS[r["level"]]["name"] if r["level"] < len(LEVELS) else "not hostable"
        r["wall_s"] = time.time() - t
    return rows


def _validate_one(args):
    ratings, bus, size, map_level, L = args
    from gridflex.hosting import LEVELS, check_level, envelopes_for

    cfg, tables, net, yp, _ = _setup(ratings)
    envs = envelopes_for(cfg, tables, bus, size, net, yp)
    all_days = list(range(net.n_hours // 24 - 1))
    ch = check_level(net, envs[L], LEVELS[L], all_days, stop_early=False)
    fails = [c.day for c in ch if not c.ok]
    return dict(bus=bus, size_mw=size, map_level=map_level, checked_level=L, days=len(ch), failing_days=len(fails),
                shed_mwh=sum(c.grid_shed_mwh for c in ch), late_mwh_h=sum(c.late_mwh_h for c in ch),
                first_failing_days=",".join(map(str, fails[:5])))


def _verify_one(args):
    """Full-year verification of one flexible map entry: raise the level until every day passes."""
    ratings, bus, size, level = args
    from gridflex.hosting import LEVELS, NOT_HOSTABLE, check_level, envelopes_for

    cfg, tables, net, yp, _ = _setup(ratings)
    envs = envelopes_for(cfg, tables, bus, size, net, yp)
    all_days = list(range(net.n_hours // 24 - 1))
    L = level
    while L < NOT_HOSTABLE:
        ch = check_level(net, envs[L], LEVELS[L], all_days, stop_early=True)
        if len(ch) == len(all_days) and all(c.ok for c in ch):
            break
        L += 1
    return dict(bus=bus, size_mw=size, stress_level=level, verified_level=L)


def verify(ratings, df, workers):
    """Two-stage map. Stress days give an exact *lower* bound: the level below fails on a real day. A
    full-year check then confirms or raises every flexible entry. "Not hostable" is already exact, since a
    failing day exists at every level."""
    from gridflex.hosting import LEVELS, NOT_HOSTABLE

    tasks = [(ratings, int(r.bus), float(r.size_mw), int(r.level)) for r in df.itertuples()
             if (not r.inferred) and 0 < r.level < NOT_HOSTABLE]
    with ProcessPoolExecutor(workers) as ex:
        res = pd.DataFrame(list(ex.map(_verify_one, tasks)))
    out = df.copy()
    out["stress_level"] = out["level"]
    for r in res.itertuples():
        m = (out.bus == r.bus) & (out.size_mw == r.size_mw)
        out.loc[m, "level"] = r.verified_level
    # keep the map monotone in size (a bigger campus never needs less flexibility)
    out = out.sort_values(["bus", "size_mw"])
    out["level"] = out.groupby("bus")["level"].cummax()
    out["verified_full_year"] = out.stress_level.between(1, NOT_HOSTABLE - 1) & ~out.inferred
    out["level_name"] = [LEVELS[int(v)]["name"] if v < NOT_HOSTABLE else "not hostable" for v in out.level]
    print(res.to_string(index=False))
    return out


def validate(ratings, entries, workers):
    """Full-year (every day, 48 h windows) check of map entries: the map's level should pass on every day,
    and the level below should fail on at least one."""
    from gridflex.hosting import NOT_HOSTABLE

    tasks = [(ratings, b, s, lv, L) for b, s, lv in entries for L in (lv - 1, lv) if 0 <= L < NOT_HOSTABLE]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_validate_one, tasks))
    for r in rows:
        print(r, flush=True)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, ratings: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from gridflex.hosting import LEVELS

    INK, INK2, GRID, SURFACE, RED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#e34948"
    # L0 (no flexibility needed) in neutral gray; L1-L5 on a sequential blue ramp (reference palette steps 250-700)
    RAMP = ["#b9b8b3", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    bus = pd.read_csv(ROOT / "data" / "raw" / "rts_gmlc" / "bus.csv").set_index("Bus ID")
    br = pd.read_csv(ROOT / "data" / "raw" / "rts_gmlc" / "branch.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    for ax, size in zip(axes.ravel(), SIZES):
        d = df[df.size_mw == size].set_index("bus")
        for _, l in br.iterrows():
            a, b = bus.loc[l["From Bus"]], bus.loc[l["To Bus"]]
            ax.plot([a.lng, b.lng], [a.lat, b.lat], color=GRID, lw=0.9, zorder=1)
        lv = d.reindex(bus.index).level
        host = lv < len(LEVELS)
        ax.scatter(bus.lng[host], bus.lat[host], c=[RAMP[int(v)] for v in lv[host]], s=34, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3)
        ax.scatter(bus.lng[~host], bus.lat[~host], marker="x", c=RED, s=26, linewidths=1.4, zorder=3)
        n0 = int((lv == 0).sum())
        nh = int(host.sum())
        ax.set_title(f"{size:,.0f} MW campus", loc="left")
        ax.text(0.02, 0.02, f"{n0} buses as an inflexible load\n+{nh - n0} more with SLO-feasible flexibility\n{len(lv) - nh} not hostable",
                transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=RAMP[i], markeredgecolor=SURFACE, markersize=8, label=LEVELS[i]["name"])
               for i in range(len(LEVELS))]
    handles.append(Line2D([], [], marker="x", ls="", color=RED, markersize=7, label="not hostable"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02),
               title="Minimum flexibility needed to connect without load shedding", title_fontsize=8.5)
    fig.suptitle(f"Where can an LLM-inference campus connect, and how flexible must it be? (RTS-GMLC, lines at {ratings:g}x)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    tag = f"_ratings{ratings:g}" if ratings != 0.6 else ""
    fig.savefig(ROOT / "paper" / "figures" / f"fig1_flex_interconnection_map{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "paper" / "figures" / f"fig1_flex_interconnection_map{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--buses", type=int, nargs="*")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--verify", action="store_true", help="full-year verification of flexible entries (stage 2)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"map_ratings{args.ratings:g}.csv"
    if args.verify:
        df = pd.read_csv(path)
        out = verify(args.ratings, df, args.workers)
        out.to_csv(OUT / f"map_verified_ratings{args.ratings:g}.csv", index=False)
        print(out.pivot_table(index="level_name", columns="size_mw", values="bus", aggfunc="count", fill_value=0))
        figure(out, args.ratings)
        return
    if args.validate:
        df = pd.read_csv(path)
        picks = []
        for b in (101, 122, 207, 303, 313):
            for size in SIZES:
                r = df[(df.bus == b) & (df.size_mw == size) & (~df.inferred)]
                if len(r) and int(r.level.iloc[0]) < 6:
                    picks.append((b, size, int(r.level.iloc[0])))
        validate(args.ratings, picks, args.workers).to_csv(OUT / f"validation_ratings{args.ratings:g}.csv", index=False)
        return
    _, _, net, _, days = _setup(args.ratings)
    buses = args.buses or [int(b) for b in net.bus_ids]
    print(f"{len(buses)} buses x {len(SIZES)} sizes on {len(days)} stress days", flush=True)
    t = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rs in ex.map(map_bus, [(b, args.ratings) for b in buses]):
            rows += rs
            print(f"  bus {rs[0]['bus']}: " + ", ".join(f"{r['size_mw']:.0f}MW->{r['level_name']}" for r in rs), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"done in {time.time() - t:.0f}s")
    print(df.pivot_table(index="level_name", columns="size_mw", values="bus", aggfunc="count", fill_value=0))
    figure(df, args.ratings)


if __name__ == "__main__":
    main()

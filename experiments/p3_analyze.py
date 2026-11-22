"""Phase 3 analysis: turn results/p3 into the X% / Y% numbers, Fig. 3 (value by mode) and Fig. 4 (paradox).

Definitions (every number in the paper uses these):

* hosting cost / emissions: inflexible DC minus no DC, on the same grid.
* delta(mode): mode minus inflexible. A negative value means flexibility helped.
  It is reported as % of hosting cost / emissions and as % of the system total.
* congestion cost: generation cost on the real (0.6x-derated) network minus
  the cost of the *same* loads on a copper plate.
* hostable: the inflexible DC causes < 0.1 GWh/yr of involuntary load shedding. At
  non-hostable sites, cost deltas are dominated by the value of lost load
  ($10k/MWh), so those sites are reported in unserved energy and left out of
  the cost/emission % figures.
* backlog adjustment: deferrable work still unserved at year end, relative to
  the inflexible run, is charged at the DC bus's mean LMP / LME from the
  inflexible run.

    python experiments/p3_analyze.py [--main-site 313]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results" / "p3"
FIG = ROOT / "paper" / "figures"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b3"
MODE_LABEL = {"price": "Price-following\n(day-ahead LMP)", "avg_ci": "Carbon-following\n(average CI)",
              "lme": "Carbon-following\n(marginal, LME)", "coopt": "Co-optimised\nwith the grid"}
MODES = list(MODE_LABEL)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def dc_means(name):
    h = pd.read_csv(P3 / "runs" / f"{name}.csv.gz", usecols=["dc0_lmp", "dc0_lme"])
    return h.dc0_lmp.mean(), h.dc0_lme.mean()


def summarise(tot: pd.DataFrame) -> pd.DataFrame:
    t = tot.set_index("name")
    rows = []
    for name, r in t.iterrows():
        if r.copper or pd.isna(r.site) or name.endswith("_copper"):
            continue
        site = int(r.site)
        cp = r.carbon_price
        suffix = f"_cp{cp:g}" if cp else ""
        ref = f"{site}_inflexible{suffix}"
        none_cost, none_em = t.loc["none", "gen_cost"], t.loc["none", "emissions"]
        if ref not in t.index:
            continue
        lmp_bar, lme_bar = dc_means(ref)
        d_backlog = r.backlog_end_mwh - t.loc[ref, "backlog_end_mwh"]
        cost = r.gen_cost + d_backlog * lmp_bar
        em = r.emissions + d_backlog * lme_bar
        host_cost = t.loc[ref, "gen_cost"] - (none_cost if not cp else np.nan)
        host_em = t.loc[ref, "emissions"] - (none_em if not cp else np.nan)
        row = dict(
            name=name, site=site, mode=r["mode"], levers=r.levers, carbon_price=cp,
            dc_twh=r.dc_energy / 1e6, dc_shed_gwh=r.dc_shed / 1e3, dc_shift_gwh=r.dc_flex / 1e3, late_mwh_h=r.dc_late,
            backlog_adj_mwh=d_backlog,
            gen_cost_musd=cost / 1e6, emissions_kt=em / 1e3, curtailment_gwh=r.curtailment / 1e3,
            congestion_rent_musd=r.congestion_rent / 1e6,
            hosting_cost_musd=host_cost / 1e6, hosting_emissions_kt=host_em / 1e3,
            d_cost_musd=(cost - t.loc[ref, "gen_cost"]) / 1e6, d_emissions_kt=(em - t.loc[ref, "emissions"]) / 1e3,
            d_curtail_gwh=(r.curtailment - t.loc[ref, "curtailment"]) / 1e3,
            grid_shed_gwh=r.grid_shed / 1e3, d_grid_shed_gwh=(r.grid_shed - t.loc[ref, "grid_shed"]) / 1e3,
            hostable=bool(t.loc[ref, "grid_shed"] < 100.0),  # < 0.1 GWh/yr of involuntary shedding
        )
        row["d_cost_pct_hosting"] = 100 * row["d_cost_musd"] / row["hosting_cost_musd"]
        row["d_emis_pct_hosting"] = 100 * row["d_emissions_kt"] / row["hosting_emissions_kt"]
        row["d_cost_pct_system"] = 100 * (cost - t.loc[ref, "gen_cost"]) / t.loc[ref, "gen_cost"]
        row["d_emis_pct_system"] = 100 * (em - t.loc[ref, "emissions"]) / t.loc[ref, "emissions"]
        cu = f"{name}_copper"
        if cu in t.index and f"{ref}_copper" in t.index:
            cc = r.gen_cost - t.loc[cu, "gen_cost"]
            cc_ref = t.loc[ref, "gen_cost"] - t.loc[f"{ref}_copper", "gen_cost"]
            cc_none = t.loc["none", "gen_cost"] - t.loc["none_copper", "gen_cost"]
            row.update(congestion_cost_musd=cc / 1e6, d_congestion_cost_musd=(cc - cc_ref) / 1e6,
                       d_congestion_pct=100 * (cc - cc_ref) / cc_ref,
                       hosting_congestion_musd=(cc_ref - cc_none) / 1e6)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_value(s: pd.DataFrame, site: int):
    d = s[(s.site == site) & (s.levers == "flexible") & (s.carbon_price == 0) & s["mode"].isin(MODES)].set_index("mode").loc[MODES]
    metrics = [("d_cost_pct_hosting", "Cost of hosting the DC"), ("d_emis_pct_hosting", "Emissions from hosting the DC"),
               ("d_congestion_pct", "Grid congestion cost")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    y = np.arange(len(MODES))
    for a, (col, title) in zip(ax, metrics):
        v = d[col].to_numpy()
        a.barh(y, v, color=[BLUE if x < 0 else RED for x in v], height=0.55)
        a.axvline(0, color=INK2, lw=0.8)
        for yi, x in zip(y, v):
            a.annotate(f"{x:+.1f}%", (x, yi), xytext=(4 if x >= 0 else -4, 0), textcoords="offset points",
                       ha="left" if x >= 0 else "right", va="center", fontsize=8, color=INK)
        span = max(abs(v).max(), 0.5)
        a.set_xlim(min(v.min(), 0) - 0.45 * span, max(v.max(), 0) + 0.45 * span)
        a.tick_params(axis="y", length=0)
        a.set_title(title, fontsize=9)
        a.set_xlabel("change vs inflexible DC (%)")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [MODE_LABEL[m] for m in MODES])
    ax[0].invert_yaxis()
    fig.suptitle(f"Value of SLO-grounded flexibility, 300 MW facility at bus {site} (one year)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig3_value_by_mode")


def fig_paradox(s: pd.DataFrame):
    d = s[(s.levers == "flexible") & (s.carbon_price == 0) & s["mode"].isin(MODES) & s.hostable]
    sites = sorted(d.site.unique())
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    y = np.arange(len(sites))
    w = 0.2
    for a, (col, title) in zip(ax, [("d_cost_pct_hosting", "Cost change"), ("d_emis_pct_hosting", "Emissions change")]):
        for j, m in enumerate(MODES):
            v = d[d["mode"] == m].set_index("site").loc[sites, col].to_numpy()
            yy = y + (j - 1.5) * w
            a.barh(yy, v, height=w * 0.85, color=[BLUE if x < 0 else RED for x in v])
            for yi, x in zip(yy, v):
                a.annotate(m.replace("avg_ci", "avg CI").replace("coopt", "co-opt").replace("lme", "LME"), (0, yi),
                           xytext=(-3 if x > 0 else 3, 0), textcoords="offset points", ha="right" if x > 0 else "left",
                           va="center", fontsize=6, color=INK2)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_title(title, fontsize=9)
        a.set_xlabel("% of the DC's hosting impact (vs inflexible)")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [f"bus {s}" for s in sites])
    ax[0].invert_yaxis()
    fig.suptitle("Same flexibility, different signal: blue = better than inflexible, red = worse",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_signal_paradox")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-site", type=int, default=313)
    args = ap.parse_args()
    style()
    tot = pd.read_csv(P3 / "totals.csv")
    s = summarise(tot)
    s.to_csv(P3 / "summary.csv", index=False)
    cols = ["name", "dc_twh", "dc_shed_gwh", "dc_shift_gwh", "late_mwh_h", "backlog_adj_mwh", "hosting_cost_musd", "hosting_emissions_kt",
            "d_cost_pct_hosting", "d_emis_pct_hosting", "d_cost_pct_system", "d_emis_pct_system", "d_curtail_gwh",
            "congestion_cost_musd", "d_congestion_pct", "hosting_congestion_musd", "grid_shed_gwh", "d_grid_shed_gwh", "hostable"]
    pd.set_option("display.width", 250)
    print(s[[c for c in cols if c in s]].round(2).to_string(index=False))
    fig_value(s, args.main_site)
    fig_paradox(s)


if __name__ == "__main__":
    main()

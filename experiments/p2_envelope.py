"""Phase 2: SLO-aware fleet -> flexibility envelope (RQ1).

Outputs (results/p2/*.csv, paper/figures/fig2*.{png,pdf}):
  fig2a  power cap vs SLO capacity and vs server power, per service
  fig2b  one week of the envelope for a 300 MW facility (firm, shed band, deferrable headroom)
  fig2c  flexibility vs SLO tightness (shed) and vs batch share (shift)

    python experiments/p2_envelope.py [--quick]
"""
import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridflex.envelope import build_envelope
from gridflex.fleet import cached_table, load_config
from gridflex.grid import load_rts_gmlc
from gridflex.workload import load_tokens, weekly_profile, year_profile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "p2"
FIG = ROOT / "paper" / "figures"
# Reference categorical palette (light): slot 1 blue, 2 orange, 3 aqua; text stays in ink tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SVC = {"conv": ("Conversation", BLUE), "code": ("Code assistant", ORANGE)}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_power_latency(it: pd.DataFrame, cfg):
    rho = cfg["interactive"]["provisioning_utilisation"]
    rows = []
    for svc, g in it.groupby("service"):
        mu1 = g.mu1.iloc[0]
        for k, gk in g.groupby("kappa"):
            # server power at the nominal operating load (rho * mu1), interpolated
            p = np.interp(rho * mu1, gk.rate, gk.power_w)
            rows.append(dict(service=svc, kappa=k, capacity_frac=gk.mu.iloc[0] / mu1, power_kw=p / 1e3,
                             feasible_at_nominal=gk.mu.iloc[0] >= rho * mu1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "power_cap_tradeoff.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for svc, g in df.groupby("service"):
        lab, col = SVC[svc]
        g = g.sort_values("kappa")
        ax[0].plot(g.kappa * 100, g.capacity_frac * 100, color=col, marker="o", ms=4)
        ax[1].plot(g.kappa * 100, g.power_kw, color=col, marker="o", ms=4)
        ax[1].annotate(lab, (g.kappa.iloc[0] * 100, g.power_kw.iloc[0]), xytext=(4, 0), textcoords="offset points",
                       color=INK2, fontsize=8, ha="left", va="center")
        ax[0].annotate(lab, (g.kappa.iloc[0] * 100, g.capacity_frac.iloc[0] * 100), xytext=(4, 0), textcoords="offset points",
                       color=INK2, fontsize=8, va="center")
    ax[0].axhline(rho * 100, color=INK2, lw=1, ls="--")
    ax[0].annotate(f"nominal load ({rho:.0%})", (54, rho * 100), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=8, color=INK2)
    ax[0].set(title="SLO capacity under a GPU power cap", xlabel="GPU power cap (% of TDP)", ylabel="req/s meeting p99 SLO\n(% of uncapped)")
    ax[1].set(title="Server power at nominal load", xlabel="GPU power cap (% of TDP)", ylabel="kW per 8-GPU server")
    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    fig.tight_layout()
    save(fig, "fig2a_power_cap_tradeoff")
    return df


def fig_week(env, start_day=14):
    h = slice(24 * start_day, 24 * (start_day + 7))
    ts = env.timestamps[h]
    firm, shed, flex, load = env.firm[h], env.shed_max[h], env.flex_max[h], env.inflexible_load[h]
    pd.DataFrame(dict(time=ts, firm=firm, shed_max=shed, flex_max=flex, inflexible=load, batch_arrivals=env.batch_arrivals[h])).to_csv(
        OUT / "envelope_week.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.fill_between(ts, firm + flex, firm, color=AQUA, alpha=0.35, lw=0, label="headroom for deferrable batch work")
    ax.fill_between(ts, firm, firm - shed, color=ORANGE, alpha=0.55, lw=0, label="sheddable within SLO (GPU power cap)")
    ax.plot(ts, firm, color=BLUE, lw=1.5, label="firm: interactive at full power + idle spares")
    ax.plot(ts, load, color=INK, lw=1.2, label="inflexible reference (batch served on arrival)")
    ax.axhline(env.facility_mw, color=INK2, lw=1, ls="--")
    ax.annotate(f"facility rating {env.facility_mw:.0f} MW", (ts[0], env.facility_mw), xytext=(2, -10), textcoords="offset points", fontsize=8, color=INK2)
    ax.set(title=f"Flexibility envelope of a {env.facility_mw:.0f} MW LLM-inference data center (one week)", ylabel="MW at the grid", ylim=(0, env.facility_mw * 1.08))
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_xlim(ts[0], ts[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    save(fig, "fig2b_envelope_week")


def sweeps(cfg, tokens, net, yp, quick):
    slo_scales = [0.75, 1.0, 1.5, 2.0] if quick else [0.6, 0.75, 1.0, 1.25, 1.5, 2.0]
    fills = [0.0, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for s in slo_scales:
        c = copy.deepcopy(cfg)
        for svc in c["slo"]:
            c["slo"][svc] = {k: v * s for k, v in c["slo"][svc].items()}
        it, bt = cached_table(c, tokens)
        env = build_envelope(c, 313, 300.0, net.timestamps, yp, it, bt)
        sm = env.summary()
        rows.append(dict(sweep="slo", value=s, shed_mw=sm["shed_max_mean_mw"], shed_share=sm["shed_share_of_load"],
                         batch_share=sm["batch_share_of_energy"], mean_mw=sm["mean_mw"]))
    it, bt = cached_table(cfg, tokens)
    for f in fills:
        env = build_envelope(cfg, 313, 300.0, net.timestamps, yp, it, bt, batch_fill=f)
        sm = env.summary()
        rows.append(dict(sweep="fill", value=f, shed_mw=sm["shed_max_mean_mw"], shed_share=sm["shed_share_of_load"],
                         batch_share=sm["batch_share_of_energy"], mean_mw=sm["mean_mw"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "flex_sweeps.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a = df[df.sweep == "slo"]
    # tighter SLOs cut per-replica capacity, so the same facility serves less traffic at a lower load
    ax[0].plot(a.value, a.shed_share * 100, color=ORANGE, marker="o", ms=4)
    ax[0].set(title="Sheddable power vs SLO tightness", xlabel="SLO scale (x default: TTFT 1 s, TPOT 100 ms)", ylabel="mean sheddable power\n(% of mean load)")
    b = df[df.sweep == "fill"]
    ax[1].plot(b.value * 100, b.batch_share * 100, color=AQUA, marker="o", ms=4)
    ax[1].plot(b.value * 100, b.shed_share * 100, color=ORANGE, marker="o", ms=4)
    ax[1].annotate("shiftable (batch, 24 h deadline)", (b.value.iloc[-1] * 100, b.batch_share.iloc[-1] * 100), xytext=(-8, -2),
                   textcoords="offset points", ha="right", va="top", fontsize=8, color=INK2)
    ax[1].annotate("sheddable (power cap)", (b.value.iloc[-1] * 100, b.shed_share.iloc[-1] * 100), xytext=(-4, 6),
                   textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax[1].set(title="Flexibility vs batch fill", xlabel="batch work (% of spare server capacity)", ylabel="% of facility energy")
    fig.tight_layout()
    save(fig, "fig2c_flex_sweeps")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    cfg = load_config()
    tokens = load_tokens()
    it, bt = cached_table(cfg, tokens)
    it.to_csv(OUT / "fleet_interactive_table.csv", index=False)
    bt.to_csv(OUT / "fleet_batch_table.csv", index=False)
    net = load_rts_gmlc()
    yp = year_profile(net.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    env = build_envelope(cfg, 313, 300.0, net.timestamps, yp, it, bt)
    pd.Series(env.summary()).to_csv(OUT / "envelope_summary_300mw.csv", header=["value"])
    print(pd.Series(env.summary()).to_string())
    print(fig_power_latency(it, cfg).round(3).to_string())
    fig_week(env)
    print(sweeps(cfg, tokens, net, yp, args.quick).round(3).to_string())


if __name__ == "__main__":
    main()

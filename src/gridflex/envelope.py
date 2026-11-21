"""Fleet simulation -> hourly linear flexibility envelope for one AI data center.

Given a facility size (MW at the grid connection), the replica tables from
:func:`gridflex.fleet.build_table` and the interactive trace profile, this
builds year-long hourly arrays (all facility MW, PUE included):

* ``firm[h]``: nominal power without deferrable work. Interactive replicas run
  at full power, provisioned for the hour's peak minute at
  ``provisioning_utilisation`` of their SLO capacity. Spare servers idle in
  hot standby, or sleep.
* ``shed_max[h]``: power saved by capping the active interactive replicas to
  the lowest cap whose SLO capacity still covers the peak-minute load. Latency
  rises, but p99 TTFT/TPOT stay within the SLO.
* ``flex_max[h]``: extra power if every spare server runs batch work at the
  batch power cap (the most energy-efficient one).
* ``batch_arrivals[h]``: deferrable energy arriving each hour (MWh). Each
  arrival must be served within ``deadline_h``.

These are linear in the dispatch variables, so the SLOs become linear OPF
constraints (see :class:`gridflex.opf.DataCenter`).

The *inflexible* reference serves batch work as soon as capacity allows
(earliest-deadline-first with no grid signal) and never sheds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .opf import DataCenter


@dataclass
class Envelope:
    bus: int
    facility_mw: float
    n_servers: int
    timestamps: pd.DatetimeIndex
    firm: np.ndarray
    shed_max: np.ndarray
    flex_max: np.ndarray
    batch_arrivals: np.ndarray
    deadline_h: int
    shed_hours_per_day: float
    shed_cost: float
    late_cost: float
    inflexible_batch: np.ndarray   # ASAP batch schedule (MWh/h) used by the inflexible reference
    info: dict

    # ---- cumulative deferrable-energy bounds on the global clock
    @property
    def cum_arrived(self) -> np.ndarray:
        return np.cumsum(self.batch_arrivals)

    @property
    def cum_due(self) -> np.ndarray:
        D = self.deadline_h
        ca = self.cum_arrived
        return np.r_[np.zeros(D), ca[:-D]] if D > 0 else ca

    @property
    def inflexible_load(self) -> np.ndarray:
        return self.firm + self.inflexible_batch

    def window(self, t0: int, T: int, mode: str = "flexible", served_before: float | None = None) -> DataCenter:
        """A :class:`DataCenter` for hours ``[t0, t0+T)``.

        ``served_before`` is the cumulative deferrable energy served before
        ``t0``. Rolling-horizon runs pass the real history. Otherwise it is
        assumed to have followed the deadline schedule.

        Modes: ``inflexible`` (fixed load), ``flexible`` (shed + shift),
        ``shift`` (deferral only) and ``shed`` (capping only, batch served ASAP).
        """
        h = slice(t0, t0 + T)
        if mode == "inflexible":
            return DataCenter.inflexible(self.bus, self.inflexible_load[h], name=f"dc@{self.bus}")
        shed = mode in ("flexible", "shed")
        shift = mode in ("flexible", "shift")
        budget = self.shed_hours_per_day * T / 24.0 * float(self.shed_max[h].max()) if shed else 0.0
        if not shift:
            return DataCenter(self.bus, firm=self.inflexible_load[h], shed_max=self.shed_max[h] if shed else None,
                              shed_budget=budget, shed_cost=self.shed_cost, name=f"dc@{self.bus}")
        if served_before is None:
            served_before = float(self.cum_due[t0 - 1]) if t0 > 0 else 0.0
        cmax = self.cum_arrived[h] - served_before
        cmin = np.maximum(self.cum_due[h] - served_before, 0.0)
        return DataCenter(
            self.bus,
            firm=self.firm[h],
            shed_max=self.shed_max[h] if shed else None,
            shed_budget=budget,
            shed_cost=self.shed_cost,
            flex_max=self.flex_max[h],
            cum_min=cmin,
            cum_max=cmax,
            late_cost=self.late_cost,
            name=f"dc@{self.bus}",
        )

    def summary(self) -> dict:
        load = self.inflexible_load
        return dict(
            bus=self.bus,
            facility_mw=self.facility_mw,
            n_servers=self.n_servers,
            mean_mw=float(load.mean()),
            peak_mw=float(load.max()),
            load_factor=float(load.mean() / self.facility_mw),
            firm_mean_mw=float(self.firm.mean()),
            shed_max_mean_mw=float(self.shed_max.mean()),
            shed_share_of_load=float(self.shed_max.mean() / load.mean()),
            flex_max_mean_mw=float(self.flex_max.mean()),
            batch_twh=float(self.batch_arrivals.sum() / 1e6),
            batch_share_of_energy=float(self.batch_arrivals.sum() / load.sum()),
            **self.info,
        )


def _interp_power(table: pd.DataFrame, kappa: float, rate) -> np.ndarray:
    t = table[table.kappa == kappa].sort_values("rate")
    r = np.r_[0.0, t.rate.to_numpy()]
    p = np.r_[t.power_w.iloc[0] - (t.power_w.iloc[1] - t.power_w.iloc[0]) / (t.rate.iloc[1] - t.rate.iloc[0]) * t.rate.iloc[0],
              t.power_w.to_numpy()]
    return np.interp(rate, r, p)


def asap_schedule(arrivals: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Serve deferrable energy FIFO as soon as capacity allows (no grid signal)."""
    out = np.zeros_like(arrivals)
    backlog = 0.0
    for h in range(len(arrivals)):
        backlog += arrivals[h]
        out[h] = min(backlog, cap[h])
        backlog -= out[h]
    return out


def build_envelope(cfg: dict, bus: int, facility_mw: float, timestamps: pd.DatetimeIndex, year_profile: dict,
                   interactive: pd.DataFrame, batch: pd.DataFrame, batch_fill: float | None = None,
                   deadline_h: int | None = None) -> Envelope:
    """Build the hourly envelope for one data center. See the module docstring."""
    pue = cfg["facility"]["pue"]
    rho = cfg["interactive"]["provisioning_utilisation"]
    u_peak = cfg["interactive"]["peak_utilisation"]
    fill = cfg["batch"]["fill_of_spare"] if batch_fill is None else batch_fill
    D = cfg["batch"]["deadline_h"] if deadline_h is None else deadline_h
    max_w = float(batch.max_w.iloc[0])
    idle_w = float(batch.idle_w.iloc[0])
    spare_w = cfg["facility"]["sleep_w"] if cfg["facility"]["idle_servers_sleep"] else idle_w
    n_servers = int(facility_mw * 1e6 / (pue * max_w))

    services = sorted(interactive.service.unique())
    caps = sorted(interactive.kappa.unique())
    mu = {(s, k): float(interactive[(interactive.service == s) & (interactive.kappa == k)].mu.iloc[0]) for s in services for k in caps}

    # scale the trace so that peak-hour interactive replicas = u_peak * n_servers
    need_unit = sum(year_profile[s][0] * year_profile[s][1] / (rho * mu[(s, 1.0)]) for s in services)
    scale = u_peak * n_servers / need_unit.max()

    H = len(timestamps)
    n_active = np.zeros(H)
    p_nom = np.zeros(H)
    p_min = np.zeros(H)
    for s in services:
        rate, pf = year_profile[s]
        lam, lam_pk = scale * rate, scale * rate * pf
        n = np.maximum(np.ceil(lam_pk / (rho * mu[(s, 1.0)])), 1.0)
        per = lam / n
        per_pk = lam_pk / n
        sub = interactive[interactive.service == s]
        pn = n * _interp_power(sub, 1.0, per)
        pm = pn.copy()
        for k in caps:  # lowest-power cap whose SLO capacity covers the peak-minute load
            ok = mu[(s, k)] >= per_pk
            pm = np.where(ok, np.minimum(pm, n * _interp_power(sub, k, per)), pm)
        n_active += n
        p_nom += pn
        p_min += pm
    spare = n_servers - n_active
    if spare.min() < 0:
        raise ValueError("interactive demand exceeds the fleet; lower peak_utilisation")

    # batch: run spare servers at the most energy-efficient cap
    b = batch.loc[batch.joules_per_token.idxmin()]
    to_mw = pue / 1e6
    firm = (p_nom + spare * spare_w) * to_mw
    shed_max = np.maximum(p_nom - p_min, 0.0) * to_mw
    flex_max = spare * (float(b.power_w) - spare_w) * to_mw
    # batch energy = `fill` x the spare capacity, arriving at a flat rate within each day
    days = len(flex_max) // 24
    daily = flex_max[: days * 24].reshape(days, 24).mean(1)
    arrivals = np.repeat(fill * daily, 24)
    if len(arrivals) < H:
        arrivals = np.r_[arrivals, np.full(H - len(arrivals), arrivals[-1])]
    inflex_batch = asap_schedule(arrivals, flex_max)

    return Envelope(
        bus=bus, facility_mw=facility_mw, n_servers=n_servers, timestamps=timestamps,
        firm=firm, shed_max=shed_max, flex_max=flex_max, batch_arrivals=arrivals, deadline_h=D,
        shed_hours_per_day=cfg["flex"]["shed_hours_per_day"], shed_cost=cfg["flex"]["shed_cost_per_mwh"],
        late_cost=cfg["facility"]["late_cost_per_mwh_h"], inflexible_batch=inflex_batch,
        info=dict(trace_scale=float(scale), batch_kappa=float(b.kappa), batch_joules_per_token=float(b.joules_per_token),
                  peak_active_replicas=float(n_active.max()), mean_active_replicas=float(n_active.mean()),
                  batch_fill=fill, deadline_h=D),
    )

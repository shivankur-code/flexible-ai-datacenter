"""Test-grid loaders, linear generator cost / emission models, and PTDF.

The main grid is NREL's RTS-GMLC (73 buses, 3 areas, 2020 hourly data). It is
loaded into a :class:`Network`: plain numpy arrays that the OPF in
:mod:`gridflex.opf` consumes directly.

Modelling choices (all documented so the paper can state them):

* **Linear costs.** A thermal unit's marginal cost is
  ``fuel price x full-load average heat rate + VOM``. The heat rate comes from
  RTS-GMLC's piecewise heat-rate curve. CO2 intensity uses the same heat rate,
  so cost and emissions stay consistent. Renewables, hydro and run-of-river cost 0.
* **No unit commitment.** By default ``Pmin`` is relaxed to 0. This is the
  usual LP relaxation used in DC-OPF studies of load flexibility. Nuclear is
  the exception: it stays at must-run ``Pmin``.
* **Excluded units.** CSP (which needs a thermal-storage model), the single
  storage unit, synchronous condensers, and the HVDC link in ``dc_branch.csv``
  are not modelled. Together they are ~250 MW of a ~14.5 GW fleet.
* **Load.** Regional (area) hourly load is split across buses in proportion to
  each bus's static ``MW Load``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LB_PER_TONNE = 2204.62262
RENEWABLE_TYPES = ("PV", "RTPV", "WIND", "HYDRO", "ROR")
EXCLUDED_TYPES = ("CSP", "STORAGE", "SYNC_COND")


@dataclass
class Network:
    """A DC power-flow network with hourly time series.

    Arrays are indexed by bus ``n`` (N), line ``l`` (L), generator ``g`` (G)
    and hour of year ``h`` (H).
    """

    name: str
    bus_ids: np.ndarray            # (N,) external bus ids
    bus_area: np.ndarray           # (N,)
    line_ids: np.ndarray           # (L,)
    line_from: np.ndarray          # (L,) bus index
    line_to: np.ndarray            # (L,) bus index
    line_x: np.ndarray             # (L,) reactance, p.u.
    line_rating: np.ndarray        # (L,) MW, continuous rating
    gen_ids: np.ndarray            # (G,)
    gen_bus: np.ndarray            # (G,) bus index
    gen_type: np.ndarray           # (G,) e.g. CC, CT, STEAM, WIND, PV
    gen_fuel: np.ndarray           # (G,)
    gen_pmin: np.ndarray           # (G,) MW
    gen_pmax: np.ndarray           # (G,) MW nameplate
    gen_cost: np.ndarray           # (G,) $/MWh
    gen_co2: np.ndarray            # (G,) tCO2/MWh
    gen_ramp: np.ndarray           # (G,) MW/h (inf if unconstrained)
    gen_avail: np.ndarray          # (G, H) available MW per hour
    load: np.ndarray               # (N, H) MW
    timestamps: pd.DatetimeIndex   # (H,)
    slack: int = 0
    _ptdf: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_bus(self) -> int:
        return len(self.bus_ids)

    @property
    def n_line(self) -> int:
        return len(self.line_ids)

    @property
    def n_gen(self) -> int:
        return len(self.gen_ids)

    @property
    def n_hours(self) -> int:
        return self.load.shape[1]

    @property
    def is_renewable(self) -> np.ndarray:
        return np.isin(self.gen_type, RENEWABLE_TYPES)

    def bus_index(self, bus_id) -> int:
        idx = np.flatnonzero(self.bus_ids == bus_id)
        if len(idx) != 1:
            raise KeyError(f"unknown bus {bus_id}")
        return int(idx[0])

    @property
    def ptdf(self) -> np.ndarray:
        """(L, N) power transfer distribution factors w.r.t. :attr:`slack`."""
        if self._ptdf is None:
            self._ptdf = compute_ptdf(self.n_bus, self.line_from, self.line_to, self.line_x, self.slack)
        return self._ptdf

    def with_ratings(self, scale: float) -> "Network":
        """Return a copy with all line ratings multiplied by ``scale``.

        RTS-GMLC is lightly congested at nameplate ratings. Derating lines is
        the standard way to create a congested test case. Every result must
        report the scale it used.
        """
        import copy

        net = copy.copy(self)
        net.line_rating = self.line_rating * scale
        net.name = f"{self.name}[ratings x{scale:g}]"
        return net


def compute_ptdf(n_bus: int, f: np.ndarray, t: np.ndarray, x: np.ndarray, slack: int = 0) -> np.ndarray:
    """DC-power-flow PTDF matrix: ``flow = PTDF @ injection`` (MW)."""
    n_line = len(f)
    A = np.zeros((n_line, n_bus))
    A[np.arange(n_line), f] = 1.0
    A[np.arange(n_line), t] = -1.0
    b = 1.0 / x
    Bbus = A.T @ (b[:, None] * A)
    keep = np.r_[0:slack, slack + 1:n_bus]
    X = np.zeros((n_bus, n_bus))
    X[np.ix_(keep, keep)] = np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return (b[:, None] * A) @ X


def _avg_heat_rate(row: pd.Series) -> float:
    """Full-load average heat rate (BTU/kWh = MMBTU/MWh x 1000)."""
    pcts, hrs = [], []
    for k in range(5):
        p = row.get(f"Output_pct_{k}")
        if pd.isna(p):
            break
        h = row["HR_avg_0"] if k == 0 else row.get(f"HR_incr_{k}")
        if pd.isna(h) or float(p) <= 0 or (pcts and float(p) <= pcts[-1]):
            break
        pcts.append(float(p))
        hrs.append(float(h))
    if not pcts:
        return 0.0
    heat = pcts[0] * hrs[0] + sum((pcts[k] - pcts[k - 1]) * hrs[k] for k in range(1, len(pcts)))
    return heat / pcts[-1]


def load_rts_gmlc(path: Path | str | None = None, relax_pmin: bool = True) -> Network:
    """Load RTS-GMLC (downloading it into ``data/raw/rts_gmlc`` if needed)."""
    from .datasets import DEFAULT_RTS_DIR, download_rts_gmlc

    path = Path(path) if path is not None else DEFAULT_RTS_DIR
    if not (path / "gen.csv").exists():
        download_rts_gmlc(path)

    bus = pd.read_csv(path / "bus.csv")
    br = pd.read_csv(path / "branch.csv")
    gen = pd.read_csv(path / "gen.csv")

    bus_ids = bus["Bus ID"].to_numpy()
    pos = {b: i for i, b in enumerate(bus_ids)}

    # --- time index (2020, hourly, leap year) ---
    reg = pd.read_csv(path / "DAY_AHEAD_regional_Load.csv")
    ts = pd.to_datetime(dict(year=reg.Year, month=reg.Month, day=reg.Day)) + pd.to_timedelta(reg.Period - 1, unit="h")
    H = len(reg)

    # --- loads: regional load split by static bus participation ---
    area = bus["Area"].to_numpy()
    static = bus["MW Load"].to_numpy(dtype=float)
    load = np.zeros((len(bus_ids), H))
    for a in np.unique(area):
        m = area == a
        share = static[m] / static[m].sum()
        load[m] = share[:, None] * reg[str(a)].to_numpy()[None, :]

    # --- generators ---
    gen = gen[~gen["Unit Type"].isin(EXCLUDED_TYPES)].reset_index(drop=True)
    gtype = gen["Unit Type"].to_numpy()
    pmax = gen["PMax MW"].to_numpy(dtype=float)
    pmin = gen["PMin MW"].to_numpy(dtype=float)
    hr = gen.apply(_avg_heat_rate, axis=1).to_numpy() / 1000.0  # MMBTU/MWh
    fuel_price = gen["Fuel Price $/MMBTU"].fillna(0).to_numpy(dtype=float)
    vom = gen["VOM"].fillna(0).to_numpy(dtype=float)
    co2_lb = gen["Emissions CO2 Lbs/MMBTU"].fillna(0).to_numpy(dtype=float)
    renewable = np.isin(gtype, RENEWABLE_TYPES)
    cost = np.where(renewable, 0.0, fuel_price * hr + vom)
    co2 = np.where(renewable, 0.0, co2_lb * hr / LB_PER_TONNE)
    ramp = gen["Ramp Rate MW/Min"].fillna(0).to_numpy(dtype=float) * 60.0
    ramp = np.where(renewable | (ramp <= 0) | (ramp >= pmax), np.inf, ramp)
    if relax_pmin:
        pmin = np.where(gtype == "NUCLEAR", pmin, 0.0)
    pmin = np.where(renewable, 0.0, pmin)

    avail = np.repeat(pmax[:, None], H, axis=1)
    for fname in ("DAY_AHEAD_wind.csv", "DAY_AHEAD_pv.csv", "DAY_AHEAD_rtpv.csv", "DAY_AHEAD_hydro.csv"):
        df = pd.read_csv(path / fname)
        for j, uid in enumerate(gen["GEN UID"]):
            if uid in df.columns:
                avail[j] = np.minimum(df[uid].to_numpy(dtype=float), pmax[j])
    avail = np.maximum(avail, 0.0)

    return Network(
        name="RTS-GMLC",
        bus_ids=bus_ids,
        bus_area=area,
        line_ids=br["UID"].to_numpy(),
        line_from=br["From Bus"].map(pos).to_numpy(),
        line_to=br["To Bus"].map(pos).to_numpy(),
        line_x=br["X"].to_numpy(dtype=float),
        line_rating=br["Cont Rating"].to_numpy(dtype=float),
        gen_ids=gen["GEN UID"].to_numpy(),
        gen_bus=gen["Bus ID"].map(pos).to_numpy(),
        gen_type=gtype,
        gen_fuel=gen["Fuel"].to_numpy(),
        gen_pmin=pmin,
        gen_pmax=pmax,
        gen_cost=cost,
        gen_co2=co2,
        gen_ramp=ramp,
        gen_avail=avail,
        load=load,
        timestamps=pd.DatetimeIndex(ts),
    )

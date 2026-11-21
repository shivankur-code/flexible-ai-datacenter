import copy

import numpy as np
import pandas as pd
import pytest

from gridflex.grid import Network, compute_ptdf, load_rts_gmlc
from gridflex.opf import DataCenter, solve_opf


def toy_network(rating=60.0, T=3, load_bus2=100.0) -> Network:
    """3 buses in a triangle with equal reactances. Bus 0 has a cheap coal
    unit, bus 1 a gas unit, and bus 2 the load. With equal x, a transfer from
    bus 0 to bus 2 splits 2/3 on line 0-2 and 1/3 on 0-1-2."""
    load = np.zeros((3, T))
    load[2] = load_bus2
    return Network(
        name="toy3",
        bus_ids=np.array([1, 2, 3]),
        bus_area=np.ones(3, int),
        line_ids=np.array(["01", "12", "02"]),
        line_from=np.array([0, 1, 0]),
        line_to=np.array([1, 2, 2]),
        line_x=np.ones(3),
        line_rating=np.array([1e4, 1e4, rating]),
        gen_ids=np.array(["coal", "gas"]),
        gen_bus=np.array([0, 1]),
        gen_type=np.array(["STEAM", "CC"]),
        gen_fuel=np.array(["Coal", "NG"]),
        gen_pmin=np.zeros(2),
        gen_pmax=np.array([500.0, 500.0]),
        gen_cost=np.array([20.0, 30.0]),
        gen_co2=np.array([1.0, 0.4]),
        gen_ramp=np.full(2, np.inf),
        gen_avail=np.full((2, T), 500.0),
        load=load,
        timestamps=pd.date_range("2020-01-01", periods=T, freq="h"),
    )


# ---------------------------------------------------------------- PTDF
def test_ptdf_triangle():
    P = compute_ptdf(3, np.array([0, 1, 0]), np.array([1, 2, 2]), np.ones(3), slack=0)
    # 1 MW injected at bus 2 and withdrawn at the slack (bus 0): -1/3 on 0-1 and 1-2, -2/3 on 0-2
    np.testing.assert_allclose(P[:, 2], [-1 / 3, -1 / 3, -2 / 3])
    np.testing.assert_allclose(P[:, 0], 0.0)


# ---------------------------------------------------------------- toy OPF, hand-checkable
def test_toy_uncongested_dispatches_coal():
    r = solve_opf(toy_network(rating=1e4), 0, 3)
    np.testing.assert_allclose(r.p[:, 0], [100, 0], atol=1e-6)
    np.testing.assert_allclose(r.lmp, 20.0, atol=1e-6)
    np.testing.assert_allclose(r.lme, 1.0, atol=1e-6)  # coal on the margin everywhere
    assert abs(r.congestion_rent) < 1e-6


def test_toy_congested_lmp_lme_and_rent():
    # Line 0-2 (60 MW) binds. With coal at 0 and gas at 1, flow on 0-2 is
    # (2/3)pc + (1/3)pg = 60 and pc + pg = 100, so pc = 80, pg = 20.
    r = solve_opf(toy_network(rating=60.0), 0, 3)
    np.testing.assert_allclose(r.p[:, 0], [80, 20], atol=1e-6)
    # Serving 1 more MW at bus 2 within the limit takes dpc = -1, dpg = +2:
    # LMP = -20 + 60 = 40 and LME = -1 + 0.8 = -0.2 (extra load there LOWERS emissions).
    np.testing.assert_allclose(r.lmp[:, 0], [20, 30, 40], atol=1e-6)
    np.testing.assert_allclose(r.lme[:, 0], [1.0, 0.4, -0.2], atol=1e-6)
    # rent = 100*40 - 80*20 - 20*30 = 1800 per hour
    assert r.congestion_rent == pytest.approx(3 * 1800.0)


def test_toy_flexible_dc_moves_to_cheap_hours():
    net = toy_network(rating=60.0, T=2, load_bus2=0.0)
    net.gen_avail[0, 1] = 0.0  # coal is unavailable in hour 1
    base = np.array([50.0, 50.0])
    inflex = solve_opf(net, 0, 2, [DataCenter.inflexible(3, base)])
    flex = solve_opf(net, 0, 2, [DataCenter(bus=3, p_min=np.zeros(2), p_max=np.full(2, 100.0),
                                            cum_min=np.array([0.0, 100.0]), cum_max=np.array([100.0, 100.0]))])
    assert flex.d.sum() == pytest.approx(100.0)
    assert flex.gen_cost < inflex.gen_cost
    # Hour 0 is capped by the line: 60 / (2/3) = 90 MW of coal reaches bus 2, plus gas for the rest.
    assert flex.d[0, 0] > 50.0


# ---------------------------------------------------------------- RTS-GMLC
@pytest.fixture(scope="module")
def rts():
    return load_rts_gmlc()


def test_rts_loader(rts):
    assert (rts.n_bus, rts.n_line) == (73, 120)
    assert rts.n_gen == 153 and rts.n_hours == 8784
    assert np.all(np.isfinite(rts.load)) and rts.load.min() >= 0
    assert np.all(rts.gen_cost[rts.is_renewable] == 0)
    assert np.all(rts.gen_avail <= rts.gen_pmax[:, None] + 1e-9)
    # coal is cheaper but dirtier than CC gas: the merit order behind the Knittel effect
    coal = rts.gen_fuel == "Coal"
    cc = rts.gen_type == "CC"
    assert rts.gen_cost[coal].mean() < rts.gen_cost[cc].mean()
    assert rts.gen_co2[coal].mean() > rts.gen_co2[cc].mean()


@pytest.fixture(scope="module")
def rts_congested(rts):
    return rts.with_ratings(0.6)


def test_rts_feasibility(rts_congested):
    net = rts_congested
    r = solve_opf(net, 24 * 15, 24)
    gen_at_bus = np.zeros((net.n_bus, 24))
    np.add.at(gen_at_bus, net.gen_bus, r.p)
    np.testing.assert_allclose(gen_at_bus + r.shed - r.inj, net.load[:, 360:384], atol=1e-5)
    assert np.abs(r.inj.sum(0)).max() < 1e-5
    assert np.all(np.abs(r.flow) <= net.line_rating[:, None] + 1e-5)
    rg = np.isfinite(net.gen_ramp)
    assert np.all(np.abs(np.diff(r.p[rg], axis=1)) <= net.gen_ramp[rg, None] + 1e-5)
    assert r.binding_lines.any(), "derated RTS should be congested"
    assert r.shed.sum() < 1e-6


def test_rts_lmp_matches_highs_duals(rts_congested):
    r = solve_opf(rts_congested, 24 * 15, 24)
    hd = r.info["highs_row_dual"][: rts_congested.n_bus * 24].reshape(24, -1).T
    np.testing.assert_allclose(np.abs(r.lmp), np.abs(hd), atol=1e-6)


@pytest.mark.parametrize("bus", [101, 207, 313])
def test_rts_marginals_match_finite_difference(rts_congested, bus):
    """Adding eps MW at one bus and hour changes cost by about LMP*eps and emissions by about LME*eps."""
    t0, h, eps = 24 * 15, 18, 0.01
    net = rts_congested
    r0 = solve_opf(net, t0, 24)
    b = net.bus_index(bus)
    net2 = copy.copy(net)
    net2.load = net.load.copy()
    net2.load[b, t0 + h] += eps
    r1 = solve_opf(net2, t0, 24, compute_lme=False)
    assert (r1.objective - r0.objective) / eps == pytest.approx(r0.lmp[b, h], abs=1e-3)
    assert (r1.emissions - r0.emissions) / eps == pytest.approx(r0.lme[b, h], abs=1e-3)


def test_rts_shiftable_dc_never_costs_more(rts_congested):
    t0 = 24 * 15
    load = np.full(24, 300.0)
    inflex = solve_opf(rts_congested, t0, 24, [DataCenter.inflexible(313, load)])
    flex = solve_opf(rts_congested, t0, 24, [DataCenter.shiftable(313, load, flex=0.3, deadline_h=6)])
    assert flex.d.sum() == pytest.approx(load.sum(), rel=1e-6)
    assert flex.objective <= inflex.objective + 1e-6

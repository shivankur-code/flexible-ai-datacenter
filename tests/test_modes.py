import numpy as np
import pandas as pd
import pytest

from gridflex.envelope import Envelope, asap_schedule
from gridflex.grid import Network
from gridflex.modes import best_response, run_year, signals_from
from gridflex.opf import DataCenter

H = 24 * 4


def toy(rating=60.0) -> Network:
    """3-bus triangle (as in test_opf). Coal at bus 1 is cheap and dirty; gas at bus 2 is
    dearer and cleaner. Bus 3 load follows a daily cycle, and coal is unavailable 12:00-18:00."""
    t = np.arange(H)
    load = np.zeros((3, H))
    load[2] = 60 + 30 * np.sin(2 * np.pi * (t % 24) / 24)
    avail = np.full((2, H), 500.0)
    avail[0, (t % 24 >= 12) & (t % 24 < 18)] = 0.0
    return Network(
        name="toy3", bus_ids=np.array([1, 2, 3]), bus_area=np.ones(3, int), line_ids=np.array(["01", "12", "02"]),
        line_from=np.array([0, 1, 0]), line_to=np.array([1, 2, 2]), line_x=np.ones(3),
        line_rating=np.array([1e4, 1e4, rating]), gen_ids=np.array(["coal", "gas"]), gen_bus=np.array([0, 1]),
        gen_type=np.array(["STEAM", "CC"]), gen_fuel=np.array(["Coal", "NG"]), gen_pmin=np.zeros(2),
        gen_pmax=np.array([500.0, 500.0]), gen_cost=np.array([20.0, 30.0]), gen_co2=np.array([1.0, 0.4]),
        gen_ramp=np.full(2, np.inf), gen_avail=avail, load=load,
        timestamps=pd.date_range("2020-01-01", periods=H, freq="h"),
    )


def toy_env(bus=3) -> Envelope:
    firm = np.full(H, 20.0)
    flex_max = np.full(H, 30.0)
    arrivals = np.full(H, 10.0)
    return Envelope(bus=bus, facility_mw=60.0, n_servers=0, timestamps=pd.date_range("2020-01-01", periods=H, freq="h"),
                    firm=firm, shed_max=np.full(H, 2.0), flex_max=flex_max, batch_arrivals=arrivals, deadline_h=12,
                    shed_hours_per_day=4, shed_cost=0.0, late_cost=500.0,
                    inflexible_batch=asap_schedule(arrivals, flex_max), info={})


def test_best_response_follows_signal_within_envelope():
    T = 6
    dc = DataCenter(3, firm=np.full(T, 10.0), shed_max=np.full(T, 2.0), shed_budget=4.0, flex_max=np.full(T, 10.0),
                    cum_min=np.array([0, 0, 0, 5, 10, 15.0]), cum_max=np.full(T, 15.0))
    sig = np.array([5, 5, 1, 1, 9, 9.0])
    d, sh, fx, late = best_response(dc, sig)
    np.testing.assert_allclose(fx, [0, 0, 10, 5, 0, 0], atol=1e-6)  # cheapest hours, within the deadlines
    np.testing.assert_allclose(sh, [0, 0, 0, 0, 2, 2], atol=1e-6)   # shed in the dearest hours, up to the budget
    np.testing.assert_allclose(d, 10 - sh + fx)
    assert late.sum() == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["inflexible", "price", "avg_ci", "lme", "coopt"])
def test_rolling_horizon_serves_all_work(mode):
    net, env = toy(), toy_env()
    inflex = run_year(net, [env], "inflexible", days=3)
    sig = None if mode in ("inflexible", "coopt") else signals_from(inflex, net, [env], mode)
    r = run_year(net, [env], mode, signals=sig, days=3)
    h = r.hourly
    assert len(h) == 72
    # served deferrable energy plus the end backlog equals arrivals
    assert h.dc_flex.sum() + r.info["backlog_end_mwh"] == pytest.approx(env.batch_arrivals[:72].sum(), rel=1e-6)
    assert h.dc_late.sum() == pytest.approx(0.0, abs=1e-6)
    assert (h.dc_shed <= 2.0 + 1e-9).all()
    assert h.grid_shed.sum() == pytest.approx(0.0, abs=1e-6)


def test_coopt_beats_inflexible_and_moves_work_out_of_gas_hours():
    net, env = toy(rating=1e4), toy_env()  # uncongested: coal hours are the cheap hours
    inflex = run_year(net, [env], "inflexible", days=3)
    co = run_year(net, [env], "coopt", days=3)
    assert co.hourly.gen_cost.sum() < inflex.hourly.gen_cost.sum()
    gas_hours = (co.hourly.hour % 24 >= 12) & (co.hourly.hour % 24 < 18)
    assert co.hourly.dc_flex[gas_hours].sum() < inflex.hourly.dc_flex[gas_hours].sum()


def test_congestion_can_make_gas_hours_cheaper_at_the_dc_bus():
    """With line 0-2 binding in the high-load coal hours, the DC bus prices at $40
    (redispatch coal -> gas) against $30 in the gas-only hours. Co-optimisation
    then moves work *into* the gas hours: the network overturns the merit order."""
    net, env = toy(rating=60.0), toy_env()
    inflex = run_year(net, [env], "inflexible", days=3)
    co = run_year(net, [env], "coopt", days=3)
    assert co.hourly.gen_cost.sum() < inflex.hourly.gen_cost.sum()
    gas_hours = (co.hourly.hour % 24 >= 12) & (co.hourly.hour % 24 < 18)
    assert inflex.hourly.dc0_lmp[gas_hours].max() < inflex.hourly.dc0_lmp.max()
    assert co.hourly.dc_flex[gas_hours].sum() > inflex.hourly.dc_flex[gas_hours].sum()


def test_price_following_can_raise_emissions():
    """The Knittel mechanism in miniature. Cheap hours are coal hours, so moving
    work into them lowers cost but raises CO2, while LME-following does the opposite."""
    net, env = toy(rating=1e4), toy_env()
    inflex = run_year(net, [env], "inflexible", days=3)
    price = run_year(net, [env], "price", signals=signals_from(inflex, net, [env], "price"), days=3)
    lme = run_year(net, [env], "lme", signals=signals_from(inflex, net, [env], "lme"), days=3)
    adj = lambda r: r.hourly.emissions.sum()
    assert price.hourly.gen_cost.sum() < inflex.hourly.gen_cost.sum()
    assert adj(price) > adj(inflex)
    assert adj(lme) < adj(price)

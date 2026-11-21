import numpy as np
import pandas as pd
import pytest

from gridflex.envelope import asap_schedule, build_envelope
from gridflex.fleet import Replica, build_table, capacity, load_config, saturation, simulate
from gridflex.workload import load_tokens, weekly_profile, year_profile


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def conv():
    t = load_tokens()
    d = t[t.service == "conv"]
    return d.ctx_tokens.to_numpy(), d.gen_tokens.to_numpy()


# ---------------------------------------------------------------- replica model
def test_replica_physics(cfg):
    full, capped = Replica(cfg, 1.0), Replica(cfg, 0.6)
    assert full.kv_capacity_tokens > 1e6  # 8x80 GB minus 140 GB weights at 320 KB/token
    t1, w1 = full.prefill(2048)
    t2, w2 = capped.prefill(2048)
    assert t2 > t1 and w2 < w1                       # a cap slows prefill and lowers power
    assert w1 <= full.max_w + 1e-9 and w2 >= capped.idle_w
    assert full.decode(128, 200_000)[0] > full.decode(8, 20_000)[0]  # bigger batch / KV is slower
    # a cap above the phase's demand does nothing
    assert Replica(cfg, 0.99).decode(1, 1000) == full.decode(1, 1000)


def test_simulator_conserves_requests_and_power_bounds(cfg, conv):
    rep = Replica(cfg)
    r = simulate(rep, 3.0, *conv, duration_s=200, warmup_s=20)
    assert r.n_done > 400
    assert rep.idle_w <= r.avg_power_w <= rep.max_w
    assert r.ttft_p50 <= r.ttft_p99 and r.tpot_p50 <= r.tpot_p99


def test_latency_rises_with_load_and_cap(cfg, conv):
    rep = Replica(cfg)
    lo, hi = simulate(rep, 1.0, *conv, duration_s=300), simulate(rep, 9.0, *conv, duration_s=300)
    assert hi.ttft_p99 > lo.ttft_p99 and hi.avg_power_w > lo.avg_power_w
    capped = simulate(Replica(cfg, 0.6), 9.0, *conv, duration_s=300)
    assert capped.ttft_p99 > hi.ttft_p99 and capped.avg_power_w < hi.avg_power_w


def test_capacity_shrinks_under_cap_and_batch_is_more_efficient(cfg, conv):
    slo = cfg["slo"]["conv"]
    mu1, r1 = capacity(Replica(cfg, 1.0), *conv, slo["ttft_s"], slo["tpot_s"], duration_s=200, iters=8)
    mu6, _ = capacity(Replica(cfg, 0.6), *conv, slo["ttft_s"], slo["tpot_s"], duration_s=200, iters=8)
    assert 0 < mu6 < mu1
    assert r1.ttft_p99 <= slo["ttft_s"] and r1.tpot_p99 <= slo["tpot_s"]
    sat = saturation(Replica(cfg, 1.0), *conv, n_requests=800)
    assert sat.joules_per_token < r1.joules_per_token  # batching amortises weights and idle power


# ---------------------------------------------------------------- envelope
@pytest.fixture(scope="module")
def env(cfg):
    it, bt = build_table(cfg, load_tokens(), duration_s=150)
    ts = pd.date_range("2020-01-01", periods=24 * 14, freq="h")
    yp = year_profile(ts, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    return build_envelope(cfg, 313, 300.0, ts, yp, it, bt)


def test_envelope_respects_facility_and_physics(env):
    assert np.all(env.firm + env.flex_max <= env.facility_mw + 1e-6)
    assert np.all(env.shed_max >= 0) and np.all(env.shed_max <= env.firm)
    assert np.all(env.flex_max >= 0)
    np.testing.assert_allclose(env.inflexible_batch.sum(), env.batch_arrivals.sum(), rtol=1e-6)  # ASAP keeps up
    assert np.all(env.inflexible_batch <= env.flex_max + 1e-9)


def test_envelope_windows(env):
    t0 = 48
    w = env.window(t0, 24, "flexible")
    assert np.all(w.cum_min <= w.cum_max + 1e-9)
    assert w.cum_max[-1] - w.cum_min[-1] <= env.batch_arrivals[t0 - env.deadline_h + 24: t0 + 24].sum() + 1e-6
    inf = env.window(t0, 24, "inflexible")
    np.testing.assert_allclose(inf.firm, env.inflexible_load[t0:t0 + 24])
    assert not inf.shed_max.any() and not inf.flex_max.any()
    sh = env.window(t0, 24, "shed")
    assert sh.shed_budget == pytest.approx(env.shed_hours_per_day * env.shed_max[t0:t0 + 24].max())


def test_asap_schedule():
    out = asap_schedule(np.array([5.0, 0, 0, 5]), np.array([2.0, 2, 2, 2]))
    np.testing.assert_allclose(out, [2, 2, 1, 2])

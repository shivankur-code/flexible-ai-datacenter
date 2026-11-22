import numpy as np
import pytest

from gridflex.fleet import build_table, load_config
from gridflex.grid import load_rts_gmlc
from gridflex.hosting import LEVELS, NOT_HOSTABLE, envelopes_for, min_level, scaled_slo, stress_days
from gridflex.workload import load_tokens, weekly_profile, year_profile


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    tok = load_tokens()
    tables = {1.0: build_table(cfg, tok, duration_s=120), 2.0: build_table(scaled_slo(cfg, 2.0), tok, duration_s=120)}
    net = load_rts_gmlc().with_ratings(0.6)
    yp = year_profile(net.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    return cfg, tables, net, yp


def test_stress_days(setup):
    cfg, tables, net, yp = setup
    days = stress_days(net)
    assert len(days) == len(set(days)) == 48          # 4 per month (default)
    assert len(stress_days(net, per_month=2)) <= 24
    assert max(days) < net.n_hours // 24 - 1  # every stress day has a lookahead day


def test_ladder_keeps_energy_and_adds_headroom(setup):
    cfg, tables, net, yp = setup
    envs = envelopes_for(cfg, tables, 313, 250.0, net, yp)
    e = [env.inflexible_load.sum() for env in envs]
    np.testing.assert_allclose(e, e[0], rtol=1e-9)                       # same work and energy on every rung
    assert envs[5].shed_max.sum() > envs[4].shed_max.sum()                # relaxed SLO -> more capping headroom
    assert [env.deadline_h for env in envs[2:5]] == [4, 12, 24]


def test_easy_site_is_hostable_inflexibly_and_radial_pocket_is_not(setup):
    cfg, tables, net, yp = setup
    days = stress_days(net)[:3]
    L, _ = min_level(net, envelopes_for(cfg, tables, 313, 100.0, net, yp), days)
    assert L == 0
    L, log = min_level(net, envelopes_for(cfg, tables, 207, 250.0, net, yp), days)
    assert L == NOT_HOSTABLE and len(log) == len(LEVELS)

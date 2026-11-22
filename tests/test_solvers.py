import numpy as np
import pytest
import torch

from gridflex.envelope import build_envelope
from gridflex.fleet import build_table, load_config
from gridflex.grid import load_rts_gmlc
from gridflex.opf import solve_opf, solve_opf_screened
from gridflex.solvers.data import make_dataset
from gridflex.solvers.problem import DispatchFamily, gauge_map, inverse_gauge, max_violation, ray_project
from gridflex.workload import load_tokens, weekly_profile, year_profile


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    it, bt = build_table(cfg, load_tokens(), duration_s=120)
    net = load_rts_gmlc().with_ratings(0.6)
    yp = year_profile(net.timestamps, weekly_profile(cfg["interactive"]["utc_offset_h"]))
    env = build_envelope(cfg, 313, 300.0, net.timestamps, yp, it, bt)
    fam = DispatchFamily(net, 313)
    D = make_dataset(net, fam, [env], [15, 200])
    return net, env, fam, D


def test_encoding_reproduces_highs(setup):
    net, env, fam, D = setup
    b, s = D["batch"], D["s_star"]
    np.testing.assert_allclose(fam.objective(s, b).numpy(), D["obj_star"].numpy(), rtol=1e-9)
    assert max_violation(fam, s, b).max() < 1e-6
    # the distributed slack is zero at a feasible optimum
    assert fam.primal(s, b)["delta"].abs().max() < 1e-9


def test_anchor_strictly_interior(setup):
    net, env, fam, D = setup
    b = D["batch"]
    s0 = fam.anchor(b)
    assert not torch.isnan(fam.last_anchor_shed).any()
    assert fam.constraints(s0, b).amax() < 0
    assert torch.minimum(s0, 1 - s0).amin() > 0


def test_gauge_and_ray_are_exactly_feasible_for_any_input(setup):
    net, env, fam, D = setup
    b = D["batch"]
    s0 = fam.anchor(b)
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        z = torch.rand(s0.shape, generator=g, dtype=s0.dtype) * 2 - 1
        assert max_violation(fam, gauge_map(fam, z, s0, b), b).max() <= 1e-7
        junk = torch.rand(s0.shape, generator=g, dtype=s0.dtype) * 1.4 - 0.2      # wildly infeasible, even off-box
        assert max_violation(fam, ray_project(fam, junk, s0, b), b).max() <= 1e-7


def test_ray_keeps_feasible_points_and_gauge_round_trips(setup):
    net, env, fam, D = setup
    b, s = D["batch"], D["s_star"]
    s0 = fam.anchor(b)
    assert torch.allclose(ray_project(fam, s, s0, b), s, atol=1e-10)
    z = inverse_gauge(fam, s, s0, b)
    assert torch.allclose(gauge_map(fam, z, s0, b), s, atol=1e-9)


@pytest.mark.parametrize("init", ["empty", "full", "random"])
def test_screened_solve_is_exact_from_any_start(setup, init):
    net, env, fam, D = setup
    t0 = 24 * 200
    dc = [env.window(t0, 24, "flexible")]
    full = solve_opf(net, t0, 24, dc, compute_lme=False)
    T, L = 24, net.n_line
    mask = {"empty": np.zeros((T, L), bool), "full": np.ones((T, L), bool),
            "random": np.random.default_rng(0).random((T, L)) < 0.05}[init]
    r = solve_opf_screened(net, t0, 24, dc, initial=mask, compute_lme=False)
    assert r.objective == pytest.approx(full.objective, rel=1e-9)
    assert np.all(np.abs(r.flow) <= net.line_rating[:, None] + 1e-6)

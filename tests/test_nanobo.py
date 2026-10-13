import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nanobo as nb  # noqa: E402


def test_unit_roundtrip():
    X = nb.latin_hypercube(10, seed=1)
    assert np.allclose(nb.from_unit(nb.to_unit(X)), X)
    assert (X >= nb.BOUNDS[:, 0]).all() and (X <= nb.BOUNDS[:, 1]).all()


def test_latin_hypercube_uses_every_slice_once():
    U = nb.latin_hypercube(8, seed=3, unit=True)
    for row in nb.slices_used(U):
        assert sorted(row) == list(range(8))


def test_scores():
    assert nb.desirability_score(10.0, 3.0) == 1.0
    assert nb.desirability_score(10.0, 20.0) == 0.0          # one bad property -> zero
    assert nb.penalty_score(10.0, 4.0) == 100.0
    assert nb.penalty_score(14.0, 6.0) == 100 - 5 * 16 - 15 * 4


def test_expected_improvement_basics():
    assert nb.expected_improvement(0.5, 1e-9, best=0.7) < 1e-6           # surely worse
    assert abs(nb.expected_improvement(0.9, 1e-9, best=0.7, xi=0) - 0.2) < 1e-6  # surely better
    assert nb.expected_improvement(0.65, 0.12, 0.7, 0) > nb.expected_improvement(0.71, 0.01, 0.7, 0)


def test_virtual_lab_is_reproducible():
    X = nb.latin_hypercube(5, seed=0)
    a = nb.VirtualLab(seed=4).run(X)
    b = nb.VirtualLab(seed=4).run(X)
    assert np.allclose(a, b)


def test_kriging_believer_returns_distinct_points():
    lab = nb.VirtualLab(seed=1)
    X = nb.latin_hypercube(10, seed=2)
    y = nb.desirability_score(*lab.run(X))
    U = nb.to_unit(X)
    gp = nb.make_gp(n_restarts=1).fit(U, y)
    picks = nb.kriging_believer(gp, U, y, 4, n_candidates=500, seed=0)
    assert picks.shape == (4, 3)
    d = np.linalg.norm(picks[:, None] - picks[None], axis=2) + np.eye(4)
    assert d.min() >= 0.05


def test_campaign_shapes():
    df = nb.run_bo(n_init=4, batch=3, n_rounds=1, seed=0, n_restarts=0, n_candidates=300)
    assert len(df) == 7 and set(df["reason"]) == {"LHS", "model"}
    assert len(nb.best_so_far(df)) == 7

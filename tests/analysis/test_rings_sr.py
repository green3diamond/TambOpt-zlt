"""CPU self-test of the ring-share formula search's scoring (symbolic/rings_sr.py) on toy profiles with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import particles, rings_sr


def profile(a=1.2, r0=30.0, n=3):
    return np.tile(a * (np.log(rings_sr.EDGES) - np.log(r0)), (n, 1))


def test_edges_and_clipping_match_the_tables():
    assert np.array_equal(rings_sr.EDGES, particles.EDGES[1:-1])
    y = particles.cum_logodds(np.r_[np.ones(1), np.zeros(particles.N_RING - 1)][None, :])
    assert (np.abs(y) >= rings_sr.LIMIT).all()                               # all energy in the first ring: nothing to measure


def test_shares_sum_to_one_and_a_dip_is_removed():
    y = profile()
    y[0, 5] = y[0, 3] - 1.0                                                   # logit F falls: an impossible profile
    s = rings_sr.to_shares(y)
    assert np.allclose(s.sum(1), 1.0) and (s >= 0).all() and s.shape == (3, particles.N_RING)


def test_measures_read_a_perfect_and_a_shifted_profile():
    y = profile()
    shares, ok = rings_sr.to_shares(y), np.ones(y.shape, bool)
    good = rings_sr.measures(y, y, ok, shares)
    assert abs(good["kl"]) < 1e-9 and good["rmse"] == 0 and good["not monotonic"] == 0
    bad = rings_sr.measures(y + 0.5, y, ok, shares)
    assert bad["kl"] > 0.01 and np.isclose(bad["rmse"], 0.5) and np.isclose(bad["rmse band"], 0.5)
    only = ok.copy()
    only[:, 0] = False                                                       # an unmeasurable edge does not count
    z = y.copy()
    z[:, 0] -= 9.0
    assert rings_sr.measures(z, y, only, shares)["rmse"] == 0


def test_expand_matches_edges_and_inputs():
    v = np.arange(12.0).reshape(2, 6)
    X, y = rings_sr.expand(v, np.arange(44.0).reshape(2, 22))
    assert X.shape == (44, 7) and np.isclose(np.exp(X[1, 0]), rings_sr.EDGES[1])
    assert np.array_equal(X[22, 1:], v[1]) and y[22] == 22.0


def test_feature_order_matches_the_variables():
    x = np.arange(8.0)[None, :]
    assert x[0, list(rings_sr.FEATURE_COLUMNS)].tolist() == [1.0, 3.0, 4.0, 6.0, 7.0]
    assert rings_sr.VARIABLES[1:] == ["age", "ln_k", "l", "c", "ln_Ek", "ln_rm"]


def test_merge_folds_scores_only_judged_planes():
    y = np.vstack([profile(n=2), profile(n=2), profile(n=2)])
    fold = np.array([0, 0, 1, 1, -1, -1])
    shares, ok = rings_sr.to_shares(y), np.ones(y.shape, bool)
    found = {0: {3: (y[:2] + 1.0, "a"), 9: (y[:2], "b")}, 1: {5: (y[2:4], "c")}}
    rows = rings_sr.merge_folds(found, fold, y, ok, shares)
    assert sorted(rows) == [5, 9] and rows[5]["formulas"] == ["a", "c"] and rows[9]["formulas"] == ["b", "c"]
    assert rows[9]["rmse"] == 0 and np.isclose(rows[5]["rmse"], np.sqrt(0.5))
    assert rings_sr.choose({5: {"rmse": 0.104}, 9: {"rmse": 0.100}, 20: {"rmse": 0.2}}) == 5


def test_fractional_logit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    v = np.column_stack([rng.uniform(0.3, 2, 300), rng.uniform(0, 3.2, 300), rng.normal(0, 0.3, 300), rng.uniform(0, 1, 300),
                         rng.normal(0, 1, 300), rng.normal(0, 1, 300)])
    X, _ = rings_sr.expand(v, np.zeros((300, 22)))
    D = rings_sr.design("b", X)
    beta = np.array([-3.0, 1.2, 0.05, -0.8, 0.4, 0.2, 0.1])
    f = 1 / (1 + np.exp(-D @ beta))
    assert np.allclose(rings_sr.fit_fractional(D, f), beta, atol=1e-3)
    assert rings_sr.form_profile("b", beta, v).shape == (300, 22)
    for form, n in rings_sr.FORM_NUMBERS.items():
        assert rings_sr.design(form, X).shape[1] == n


def test_plane_inputs_order():
    x = np.tile(np.arange(8.0), (2, 1))
    v = rings_sr.plane_inputs(x, np.array([0, 3]))
    assert np.allclose(v[1], [1.0, np.log(4.0), 3.0, 4.0, 6.0, 7.0])


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)

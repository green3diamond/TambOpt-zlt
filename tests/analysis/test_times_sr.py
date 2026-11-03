"""CPU self-test of the arrival-time formula search's scoring (symbolic/times_sr.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import particles, rings_sr, times_sr


def test_rings_match_the_tables():
    assert times_sr.N_RING == particles.N_RING
    assert (np.diff(times_sr.LN_R) > 0).all()
    assert np.isclose(np.exp(times_sr.LN_R[4]), np.sqrt(10.0 * 20.0))         # the 10-20 m ring


def test_expand_matches_rings_and_inputs():
    v = np.arange(12.0).reshape(2, 6)
    X = times_sr.expand(v)
    assert X.shape == (46, 7) and X[24, 0] == times_sr.LN_R[1] and np.array_equal(X[24, 1:], v[1])


def test_weighted_rmse_uses_counts_and_skips_empty_rings():
    y, n = np.array([[1.0, 5.0, np.nan]]), np.array([[3.0, 1.0, 9.0]])
    pred = np.array([[1.0, 7.0, 0.0]])
    assert np.isclose(times_sr.weighted_rmse(pred, y, n), np.sqrt(4.0 * 1.0 / 4.0))
    assert times_sr.weighted_rmse(pred, y, np.array([[3.0, 0.0, 9.0]])) == 0.0


def test_merge_folds_scores_only_judged_planes():
    y, n = np.zeros((6, 2)), np.ones((6, 2))
    fold = np.array([0, 0, 1, 1, -1, -1])
    found = {0: {3: (np.ones((2, 2)), "a"), 9: (np.zeros((2, 2)), "b")}, 1: {5: (np.zeros((2, 2)), "c")}}
    rows = times_sr.merge_folds(found, fold, y, n)
    assert sorted(rows) == [5, 9] and rows[5]["formulas"] == ["a", "c"] and rows[9]["rmse"] == 0
    assert np.isclose(rows[5]["rmse"], np.sqrt(0.5))
    assert times_sr.choose({5: {"rmse": 0.104}, 9: {"rmse": 0.100}, 20: {"rmse": 0.2}}) == 5


def toy_rows(n=3000):
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(0, 8, n), rng.uniform(0.3, 2.5, n), rng.uniform(0, 3.2, n), rng.uniform(-0.5, 0.6, n),
                            rng.uniform(0, 1, n), rng.uniform(-5, 5, n), rng.uniform(3, 6, n)])


def test_forms_recover_known_constants():
    X = toy_rows()
    for form, theta in (("a", (-5.7,)), ("b", (-4.0, 1.6, 0.7)), ("c", (0.6, 0.35, -0.02, 0.8)), ("d", (0.6, 0.35, -0.02, 0.8, 0.3, -0.2))):
        y = times_sr.FORMS[form][0](np.array(theta), X)
        got = times_sr.fit_form(form, X, y)
        assert np.allclose(np.abs(got), np.abs(theta), atol=1e-3), (form, got)


def test_geometric_delay_at_a_known_point():
    # r = 100 m, plane 12: extra path 100^2 / (2 x 6000 m) = 0.833 m = 2.78 ns
    X = np.array([[np.log(100.0), 1.0, np.log(12.0), 0.0, 0.5, 0.0, 4.0]])
    t = 10 ** times_sr.form_a((times_sr.GEOMETRIC,), X)[0] - 1
    assert abs(t - 2.78) < 0.01


def test_choose_common_takes_the_fewest_within_tolerance():
    assert times_sr.choose_common({"a": 0.2, "c": 0.1015, "d": 0.100}, {"a": 1, "c": 4, "d": 6}) == "c"
    assert times_sr.choose_common({"c": 0.103, "d": 0.100}, {"c": 4, "d": 6}) == "d"


def test_stands_needs_the_mean_and_every_group():
    assert times_sr.stands({"lead": [0.5, 1.0, 1.2, 1.2], "delay": [1.0, 1.0, 1.0, 1.0]})
    assert not times_sr.stands({"lead": [0.5, 0.5, 0.5, 1.6], "delay": [1.0, 1.0, 1.0, 1.0]})
    assert not times_sr.stands({"lead": [1.0, 1.0, 1.0, 1.0], "delay": [1.3, 1.3, 1.3, 1.3]})


def test_score_rows_uses_judged_planes_only():
    y, n = np.zeros((4, times_sr.N_RING)), np.ones((4, times_sr.N_RING))
    pred = np.zeros_like(y)
    pred[2:] = 9.0
    d = {"y": y, "n": n, "fold": np.array([0, 1, -1, -1])}
    assert times_sr.score_rows(pred, d)["rmse"] == 0.0


def test_shared_constants_skip_inputs_and_group_lists():
    text = "f = (#7 / -0.12180786) * ((#6 * #7) * 0.016059645); h = square(#1 * 1e-3); A = [-1.7, 0.2]; B = [2.1, 1.8]; C = [0.1, -0.4]"
    assert times_sr.shared_constants(text) == 3
    assert times_sr.shared_constants("f = #1; h = #2; A = [1.0]") == 0


def test_ring_times_match_the_form_on_expanded_rows():
    x, k = np.arange(1.0, 9.0), 11
    theta = np.array([0.6, 0.35, -0.02, 0.8])
    got = times_sr.ring_times(theta, "c", x, k)
    v = rings_sr.plane_inputs(x[None], np.array([k]))
    assert got.shape == (times_sr.N_RING,) and np.array_equal(got, times_sr.form_c(theta, times_sr.expand(v)))
    assert np.isclose(times_sr.expand(v)[0, 2], np.log(12.0))


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

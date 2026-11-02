"""CPU self-test of the arrival-time formula search's scoring (symbolic/times_sr.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import particles, times_sr


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

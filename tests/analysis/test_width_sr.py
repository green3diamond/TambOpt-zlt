"""CPU self-test of the width formula search's scoring (symbolic/width_sr.py) on toy data with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import width_sr


def test_zone_gaps_read_the_offset_of_each_zone():
    k = np.tile(np.arange(24), 5)
    y = np.zeros(len(k))
    pred = np.where(k < 8, 0.3, np.where((k >= 10) & (k < 18), -0.2, np.where(k >= 20, 0.1, 9.0)))
    assert np.allclose(width_sr.zone_gaps(pred, y, k), [0.3, -0.2, 0.1])
    y[k == 3] = [0.0, 0.0, 0.0, 5.0, 5.0]                                    # the target's median, not its mean
    assert np.allclose(width_sr.zone_gaps(pred, y, k), [0.3, -0.2, 0.1])


def test_out_of_fold_never_sees_the_scored_fold():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(400), rng.normal(size=400)])
    fold = np.tile([0, 1], 200)
    y = x @ np.array([1.0, 2.0])
    y[fold == 1] += 5.0                                                      # fold 1 is shifted: a leak would hide it
    pred = width_sr.out_of_fold(x, y, fold)
    assert np.allclose((y - pred)[fold == 1], 5.0, atol=0.3) and np.allclose((y - pred)[fold == 0], -5.0, atol=0.3)


def test_choose_takes_the_smallest_size_near_the_best():
    rows = {3: {"rmse": 0.200}, 5: {"rmse": 0.1009}, 9: {"rmse": 0.100}, 15: {"rmse": 0.102}}
    assert width_sr.choose(rows) == 5


def test_merge_folds_uses_each_folds_largest_formula_not_above_the_size():
    fold = np.array([0, 0, 1, 1])
    y, k, steep = np.zeros(4), np.array([0, 1, 0, 1]), np.ones(4, bool)
    found = {0: {1: (np.array([1.0, 1.0]), "a"), 5: (np.array([3.0, 3.0]), "b")}, 1: {3: (np.array([2.0, 2.0]), "c")}}
    rows = width_sr.merge_folds(found, fold, y, k, steep)
    assert sorted(rows) == [3, 5]                                            # size 1: fold 1 has no formula yet
    assert rows[3]["formulas"] == ["a", "c"] and rows[5]["formulas"] == ["b", "c"]
    assert np.isclose(rows[3]["rmse"], np.sqrt((1 + 4) / 2))                 # fold 0's formula scored on fold 1's rows


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

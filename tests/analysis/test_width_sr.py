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


def test_fit_tanh_recovers_known_constants():
    rng = np.random.default_rng(2)
    p, L = rng.uniform(0, 3.2, 800), rng.normal(size=(800, 2))
    y = 1.3 * np.tanh(1.5 * (p - 1.8)) - 0.6 + L @ np.array([0.2, -0.1]) + rng.normal(0, 0.01, 800)
    theta = width_sr.fit_tanh(p, L, y)
    if theta[2] < 0:                                                         # (-A, -B) is the same curve
        theta[[0, 2]] *= -1
    assert np.allclose(theta, [1.5, 1.8, 1.3, -0.6, 0.2, -0.1], atol=0.03)
    assert np.sqrt(np.mean((width_sr.tanh_predict(theta, p, L) - y) ** 2)) < 0.012
    empty = width_sr.fit_tanh(p, L[:, :0], 1.3 * np.tanh(1.5 * (p - 1.8)) - 0.6)
    assert len(empty) == 4


def test_fit_tanh_keeps_c_inside_its_range():
    p = np.linspace(0, 3.2, 500)
    theta = width_sr.fit_tanh(p, p[:, None][:, :0], 0.1 * np.exp(0.8 * p), (0.0, 3.2))
    assert 0.0 <= theta[1] <= 3.2 and abs(theta[2]) < 1e4


def test_spread_and_memory_are_recovered():
    rng = np.random.default_rng(3)
    k = np.tile(np.arange(24), 300)
    unit = np.repeat(np.arange(300), 24)
    p = np.log(k + 1.0)
    z = np.concatenate([np.array(ar1_toy(rng, 0.8)) for _ in range(300)])
    e = np.exp(-1.0 - 0.5 * p) * z
    a, b = width_sr.fit_spread(p, e)
    assert abs(a + 1.0) < 0.05 and abs(b + 0.5) < 0.03
    assert abs(width_sr.lag_one(z, unit, k) - 0.8) < 0.02


def ar1_toy(rng, rho, n=24):
    a = [rng.normal()]
    for _ in range(n - 1):
        a.append(rho * a[-1] + np.sqrt(1 - rho ** 2) * rng.normal())
    return a


def test_law_mean_and_sd_read_the_saved_order():
    w = np.array([1.5, 1.8, 1.3, -0.6, 0.2, -0.1, -1.0, -0.5, 0.8])
    k, l, age = np.array([0, 5, 23]), np.array([0.1, 0.0, -0.2]), np.array([0.3, 1.0, 1.5])
    want = 1.3 * np.tanh(1.5 * (np.log(k + 1.0) - 1.8)) - 0.6 + 0.2 * l - 0.1 * age
    assert np.allclose(width_sr.law_mean(w, k, l, age), want)
    assert np.allclose(width_sr.law_sd(w, k), np.exp(-1.0 - 0.5 * np.log(k + 1.0)))


def test_choose_form_takes_the_fewest_numbers_near_the_best():
    rows, sizes = {"a": 0.200, "b": 0.1015, "c": 0.100}, {"a": 4, "b": 6, "c": 9}
    assert width_sr.choose_form(rows, sizes) == "b"


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

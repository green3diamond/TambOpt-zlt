"""CPU self-test of the interval helpers and the evaluation scripts' pure functions."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))

import numpy as np

import detection_stats as ds
import recon_resolution as rr
from analysis_core import stats


def test_bootstrap_median_interval_width():
    x = np.random.default_rng(1).normal(size=4000)
    lo, hi = stats.bootstrap_ci(x, n_boot=400)
    expected_half = 1.2533 / np.sqrt(len(x))
    assert 0.75 < (hi - lo) / 2 / expected_half < 1.25, (lo, hi)


def test_paired_interval_of_identical_arrays_is_zero():
    x = np.random.default_rng(2).normal(size=100)
    assert stats.paired_bootstrap_ci(x, x, n_boot=50) == (0.0, 0.0)


def test_wilson_known_values():
    lo, hi = stats.wilson(np.array([5, 0, 3]), np.array([10, 10, 0]))
    assert np.isclose(lo[0], 0.5 - np.sqrt(0.025 + 0.0025) / 1.1) and np.isclose(hi[0], 1 - lo[0])
    assert np.isclose(lo[1], 0.0) and hi[1] > 0 and np.isnan(lo[2])


def test_bin_index_closes_outer_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    assert stats.bin_index(np.array([0.0, 0.5, 1.0, 2.9, 3.0]), edges).tolist() == [0, 0, 1, 2, 2]


def test_angular_error():
    x, y = np.array([[1.0, 0, 0]]), np.array([[0, 2.0, 0]])
    assert np.isclose(rr.angular_deg(x, y)[0], 90.0)
    assert np.isclose(rr.angular_deg(3 * x, x)[0], 0.0, atol=1e-5)
    assert np.isclose(rr.angular_deg(-x, x)[0], 180.0)


def test_energy_error_in_dex():
    pred = np.array([[0, 0, 1.0, 0.75]])
    prim = np.array([[0, 0, 1.0, 0.25, 0, 0, 0, 0]])
    assert np.isclose(rr.errors(pred, prim)[1][0], 0.5 * rr.SPAN)


def test_confusion_counts():
    k = np.array([[True, True, False, False]])
    s = np.array([[True, False, True, False]])
    c = ds.confusion(k, s)
    assert (c["both"], c["kernel_only"], c["surrogate_only"], c["neither"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5


def test_efficiency_fractions():
    n_lit = np.array([0, 5, 10, 20, 1, 30])
    x = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
    eff = ds.efficiency(n_lit, x, np.array([0.0, 1.0, 2.0]))
    assert eff["counts"] == [3, 3]
    assert np.allclose(eff["10"]["frac"], [1 / 3, 2 / 3]) and np.allclose(eff["1"]["frac"], [2 / 3, 1.0])


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

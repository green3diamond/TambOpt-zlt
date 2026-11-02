"""CPU self-test of the fair judge (symbolic/judge.py): the fold split, and on toy clouds that a source drawn from the
target's law scores at its floor and clearly beats a corrupted one, whose cells are shifted across track."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from symbolic import judge
from test_suite import toy_clouds, toy_dirs

N = 12
M = 4


def test_fold_split_balances_class_and_band():
    rng = np.random.default_rng(0)
    cls = np.array(["EM", "hadronic"] * 40)
    u, l = np.r_[rng.uniform(-0.09, 0.09, 24), rng.uniform(0.11, 0.5, 56)], rng.uniform(5, 6, 80)
    fold = judge.fold_of(cls, l, u)
    for c in ("EM", "hadronic"):
        for m in (np.abs(u) < 0.1, np.abs(u) >= 0.1):
            assert abs(fold[m & (cls == c)].sum() * 2 - (m & (cls == c)).sum()) <= 1
    m = (cls == "EM") & (np.abs(u) >= 0.1)
    assert (np.abs(np.diff(fold[m][np.argsort(l[m])])) == 1).all()              # alternating in energy order


def test_good_source_at_floor_corrupted_one_loses():
    rng = np.random.default_rng(1)
    dirs = toy_dirs(rng, N)
    rows = np.repeat(np.arange(N), M)
    target = judge.observables(toy_clouds(rng, dirs), dirs, np.arange(N))
    good = judge.observables(toy_clouds(rng, dirs[rows]), dirs, rows)
    bad = judge.observables(toy_clouds(rng, dirs[rows], shift=150.0), dirs, rows)
    assert np.array_equal(np.unique(good["abs_t1"][3]), np.arange(M)) and np.unique(target["abs_t1"][3]).tolist() == [0]
    r = judge.score(target["abs_t1"], good["abs_t1"], bad["abs_t1"], np.arange(N), rng, n_boot=50)
    assert r["symbolic"] <= 3 * r["floor symbolic"], r
    assert r["neural chain"] >= 5 * r["floor neural chain"], r
    assert judge.verdict(r, "abs_t1") == "PASS clear", r
    swapped = judge.score(target["abs_t1"], bad["abs_t1"], good["abs_t1"], np.arange(N), rng, n_boot=50)
    assert judge.verdict(swapped, "abs_t1") == "FAIL clear", swapped
    assert judge.verdict(r, "lead_time") == "reported"


def test_resampling_keeps_primaries_paired():
    o = (np.array([1.0, 2.0, 5.0]), np.ones(3), np.array([0, 1, 2]), np.zeros(3, int))
    p = (np.array([1.0, 2.0, 9.0]), np.ones(3), np.array([0, 1, 2]), np.zeros(3, int))
    assert judge.dist(o, p, np.array([1.0, 1.0, 0.0])) == 0.0                    # primary 2 out of both sides
    assert judge.dist(o, p, np.array([0.0, 0.0, 0.0])) != judge.dist(o, p, np.array([0.0, 0.0, 0.0]))     # empty: NaN


def test_per_plane_sees_plane_order_the_pooled_distance_misses():
    v = np.arange(24.0)
    one, zero = np.ones(24), np.zeros(24, int)
    a = (v, one, zero, zero, np.arange(24))
    b = (v, one, zero, zero, np.arange(24)[::-1])
    assert judge.dist(a, b) == 0.0
    assert judge.per_plane(a, b) == np.mean(np.abs(v - v[::-1]))
    assert judge.per_plane(a, a) == 0.0


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

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


def test_without_a_chain_the_symbolic_side_is_scored_and_only_reported():
    rng = np.random.default_rng(1)
    dirs = toy_dirs(rng, N)
    rows = np.repeat(np.arange(N), M)
    target = judge.observables(toy_clouds(rng, dirs), dirs, np.arange(N))
    good = judge.observables(toy_clouds(rng, dirs[rows]), dirs, rows)
    r = judge.score(target["abs_t1"], good["abs_t1"], None, np.arange(N), rng, n_boot=50)
    assert "neural chain" not in r and r["symbolic"] <= 3 * r["floor symbolic"], r
    assert judge.verdict(r, "abs_t1") == "reported"


def test_a_comparison_draw_is_judged_like_the_chain_under_its_own_name():
    rng = np.random.default_rng(1)
    dirs = toy_dirs(rng, N)
    rows = np.repeat(np.arange(N), M)
    target = judge.observables(toy_clouds(rng, dirs), dirs, np.arange(N))
    good = judge.observables(toy_clouds(rng, dirs[rows]), dirs, rows)
    bad = judge.observables(toy_clouds(rng, dirs[rows], shift=150.0), dirs, rows)
    r = judge.score(target["abs_t1"], good["abs_t1"], bad["abs_t1"], np.arange(N), rng, n_boot=50, other="comparison")
    assert "neural chain" not in r and {"comparison", "floor comparison", "median comparison"} <= set(r), r
    assert judge.verdict(r, "abs_t1", "comparison") == "PASS clear", r
    assert judge.verdict(r, "abs_t1") == "reported"                              # no chain in r
    swapped = judge.score(target["abs_t1"], bad["abs_t1"], good["abs_t1"], np.arange(N), rng, n_boot=50, other="comparison")
    assert judge.verdict(swapped, "abs_t1", "comparison") == "FAIL clear", swapped


def test_one_piece_scores_one_observable_of_its_group_only():
    rng = np.random.default_rng(2)
    dirs = toy_dirs(rng, N)
    rows = np.repeat(np.arange(N), M)
    obs = {"target": judge.observables(toy_clouds(rng, dirs), dirs, np.arange(N)),
           "symbolic": judge.observables(toy_clouds(rng, dirs[rows]), dirs, rows),
           "comparison": judge.observables(toy_clouds(rng, dirs[rows], shift=150.0), dirs, rows)}
    cls = np.array(["EM", "hadronic"] * (N // 2))
    prim = {"cls": cls, "u": dirs[:, 2]}
    judge._DATA.clear()
    judge._DATA.update(prim=prim, judged=np.ones(N, bool), other="comparison",
                       **{src: {"electron": o} for src, o in obs.items()})
    band = "steep" if judge.suite.BANDS["steep"](dirs[cls == "EM", 2]).sum() >= 2 else "grazing"
    s, cn, b, name, size, r = judge.one_piece(("electron", "EM", band, "abs_t1", 7, 50))
    group = np.flatnonzero((cls == "EM") & judge.suite.BANDS[band](dirs[:, 2]))
    keep = lambda o: judge.take(o, np.isin(o[2], group))
    direct = judge.score(*(keep(obs[k]["abs_t1"]) for k in ("target", "symbolic", "comparison")), group,
                         np.random.default_rng(7), 50, "comparison")
    assert (s, cn, b, name, size) == ("electron", "EM", band, "abs_t1", len(group)) and len(group) > 0
    assert r.pop("verdict") == judge.verdict(direct, "abs_t1", "comparison")
    assert r == direct, (r, direct)
    judge._DATA.clear()


def test_compare_needs_its_fit_lists_and_no_chain():
    import argparse
    for bad in ({"chain": "given", "compare_fit": ["x"]}, {"chain": "none", "compare_fit": None}):
        args = argparse.Namespace(out=".", folder=".", truth="h5", compare="d{}", fit=["x"], folds=[0], **bad)
        try:
            judge.stage_run(None, args, None)
        except ValueError as exc:
            assert "--compare needs" in str(exc), exc
        else:
            raise AssertionError(f"accepted {bad}")


def toy_h5(folder, name, cells, total, d, cap):
    import h5py
    from symbolic import sources
    os.makedirs(folder, exist_ok=True)
    for s in sources.CAP_NEW:
        with h5py.File(os.path.join(folder, f"{name}_{sources.CORSIKA_NAME[s]}.h5"), "w") as f:
            f.attrs["nmax"] = cap if cap else sources.CAP_NEW[s]
            v = f.create_dataset("showers", (1,), dtype=h5py.vlen_dtype(np.float32))
            v[0] = cells.astype(np.float32).ravel()
            f["num_points"] = np.array([len(cells)])
            f["directions"] = np.array([d], np.float32)
            f.create_group("plane_stats")["energy"] = total.astype(np.float32)


def test_h5_truth_reads_stored_cells_and_refuses_inconsistent_files():
    import tempfile
    d = np.array([0.0, 0.6, 0.8])
    cells = np.array([[10.0, 20.0, 3, 5.0, 1e-6], [0.0, 0.0, 3, 2.0, 1e-6], [5.0, 5.0, 7, 1.0, 2e-6]])
    total = np.zeros(24)
    total[3], total[7] = 8.0, 1.0
    with tempfile.TemporaryDirectory() as tmp:
        prim = {"sims": np.array([os.path.join(tmp, "a")]), "names": np.array(["a"]), "dirs": np.array([d])}
        toy_h5(prim["sims"][0], "a", cells, total, d, None)
        out = judge.read_truth_h5(prim, [0])
        assert all(np.allclose(out[s][0], cells, atol=1e-6) for s in out)
        for bad in (dict(total=total * 0.5), dict(cap=4096), dict(d=np.array([0.6, 0.0, 0.8]))):
            sim = os.path.join(tmp, "b" + str(len(os.listdir(tmp))))
            toy_h5(sim, "b", **{**dict(cells=cells, total=total, d=d, cap=None), **bad})
            try:
                judge.read_truth_h5({**prim, "sims": np.array([sim])}, [0])
            except ValueError:
                continue
            raise AssertionError(f"accepted an inconsistent file: {list(bad)}")


def test_histogram_resampling_matches_the_exact_one():
    rng = np.random.default_rng(7)
    dirs = toy_dirs(rng, N)
    rows = np.repeat(np.arange(N), M)
    target = judge.observables(toy_clouds(rng, dirs), dirs, np.arange(N))
    good = judge.observables(toy_clouds(rng, dirs[rows]), dirs, rows)
    near = judge.observables(toy_clouds(rng, dirs[rows], shift=20.0), dirs, rows)
    for name in ("abs_t1", "cell_energy", "centre"):
        exact = judge.score(target[name], good[name], near[name], np.arange(N), np.random.default_rng(3), n_boot=60)
        fast = judge.score(target[name], good[name], near[name], np.arange(N), np.random.default_rng(3), n_boot=60,
                           boot_bins=2 ** 16)
        for k, v in exact.items():
            if k != "difference 95%":
                assert v == fast[k], (name, k)                                  # every other number exact
        width = exact["difference 95%"][1] - exact["difference 95%"][0]
        gap = np.abs(np.subtract(exact["difference 95%"], fast["difference 95%"])).max()
        assert gap <= 0.02 * width + 1e-9, (name, exact["difference 95%"], fast["difference 95%"])
        assert judge.verdict(exact, name) == judge.verdict(fast, name), name


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

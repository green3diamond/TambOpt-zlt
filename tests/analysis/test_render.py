"""CPU self-test of the preprocessing forward model (symbolic/render.py) against the preprocessing script itself."""
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from symbolic import lateral, render

PREPROCESS = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/preprocess_showers.py"
D = np.array([0.55, -0.75, 0.2]) / np.linalg.norm([0.55, -0.75, 0.2])
D_X = np.array([0.1, -0.95, 0.2]) / np.linalg.norm([0.1, -0.95, 0.2])   # across track ~ ground x: the script's key fits


def preprocess_module():
    spec = importlib.util.spec_from_file_location("preprocess_showers", PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy_particles(rng, n=6000, planes=(3, 7, 12), spread=40.0):
    k = rng.choice(planes, n)
    a, b = rng.normal(0, spread, n), rng.normal(0, spread, n)
    return np.column_stack([k, a, b, rng.gamma(1.5, 2.0, n), rng.uniform(0, 50e-9, n) + k * 1.7e-6])


def script_cells(mod, k, xy, e, t):
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "plane_index": k, "kinetic_energy": e, "time": t,
                       "p_energy": np.full(len(k), 1e6)})
    out = mod._cluster_hits(df, cell_size=render.CELL, np=np, random_shift=False, include_time=True)
    return out[["x", "y", "plane_idx", "energy", "time"]].to_numpy(np.float64)


def by_cell(cells):
    return cells[np.lexsort((cells[:, 1], cells[:, 0], cells[:, 2]))]


def brute_force(k, xy, e, cell=render.CELL):
    acc = {}
    for p, (x, y), w in zip(k, xy.astype(np.float32), e):
        key = (int(p), int(np.floor(x / np.float32(cell))), int(np.floor(y / np.float32(cell))))
        acc[key] = acc.get(key, 0.0) + w
    return acc


def test_ground_frame_round_trips_through_lateral_to_plane():
    rng = np.random.default_rng(1)
    p = toy_particles(rng, 500)
    k = p[:, 0].astype(int)
    xy = render.to_ground(k, p[:, 1], p[:, 2], D)
    h, perp, c = lateral.frame(D)
    axis = (k + 1)[:, None] * render.atmosphere.PLANE_SPACING * D[None, :2]
    assert np.allclose(lateral.to_plane(xy - axis, h, perp, c), p[:, 1:3], atol=1e-6)


def test_cluster_matches_the_preprocessing_script_without_overflow():
    rng = np.random.default_rng(2)
    p = toy_particles(rng)
    k = p[:, 0].astype(int)
    xy = render.to_ground(k, p[:, 1], p[:, 2], D_X)
    ours = by_cell(render.cluster(k, xy, p[:, 3], p[:, 4]))
    theirs = by_cell(script_cells(preprocess_module(), k, xy, p[:, 3], p[:, 4]))
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    assert np.allclose(ours[:, :3], theirs[:, :3], atol=1e-3) and np.allclose(ours[:, 3], theirs[:, 3], rtol=1e-5)
    assert np.allclose(ours[:, 4], theirs[:, 4], rtol=1e-6)


def test_cluster_is_exact_where_the_script_key_overflows():
    rng = np.random.default_rng(3)
    n = 3000
    k = np.full(n, 4)
    xy = np.column_stack([rng.uniform(0, 30, n), rng.uniform(0, 600, n)])     # narrow in x, long in y: stride too small
    e, t = rng.gamma(1.5, 2.0, n), rng.uniform(0, 1e-7, n)
    ours = render.cluster(k, xy, e, t)
    truth = brute_force(k, xy, e)
    got = {(int(p), int(np.floor(x / render.CELL)), int(np.floor(y / render.CELL))): w for x, y, p, w, _ in ours}
    assert set(got) == set(truth) and all(np.isclose(got[q], truth[q]) for q in truth)
    theirs = script_cells(preprocess_module(), k, xy, e, t)
    assert len(theirs) != len(ours) or not np.allclose(by_cell(theirs)[:, :2], by_cell(ours)[:, :2])   # the script errs


def test_radius_cut_keeps_ninety_percent_per_plane_and_cap_keeps_the_brightest():
    rng = np.random.default_rng(4)
    p = toy_particles(rng)
    k = p[:, 0].astype(int)
    xy = render.to_ground(k, p[:, 1], p[:, 2], D)
    keep = render.radius_cut(k, xy)
    for q in np.unique(k):
        m = k == q
        r = np.hypot(*xy[m].T)
        assert np.array_equal(keep[m], r <= np.percentile(r, 90)) and abs(keep[m].mean() - 0.9) < 0.01
    cells = render.cluster(k[keep], xy[keep], p[keep, 3], p[keep, 4])
    capped = render.cap(cells, 200)
    assert len(capped) == 200 and np.all(np.diff(capped[:, 2]) >= 0)
    assert np.isclose(capped[:, 3].sum(), np.sort(cells[:, 3])[-200:].sum())
    full = render.render(p, D, 10 ** 9)
    assert np.isclose(full[:, 3].sum(), p[keep, 3].sum())


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

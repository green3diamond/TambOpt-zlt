"""CPU self-test of the CORSIKA-level suite: across-track geometry, and that a corrupted source (cells shifted across
track) stands far above the split-half scale while an independent draw of the same law does not."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import annuli, atmosphere, lateral, suite

N_SHOWERS = 60
N_CELLS = 150


def toy_dirs(rng, n):
    zen = np.radians(rng.uniform(60, 89, n))
    az = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)])


def toy_clouds(rng, dirs, shift=0.0):
    """Cells about the axis: across-track Gaussian (40 m), energy peaking mid-shower, time growing with offset."""
    out = []
    for d in dirs:
        h, perp, c = lateral.frame(d)
        cells = []
        for k in range(24):
            a = rng.normal(shift, 40.0, N_CELLS)
            b = rng.normal(0, 40.0, N_CELLS)
            axis = (k + 1) * atmosphere.PLANE_SPACING * d[:2] / np.linalg.norm(d)
            xy = axis + a[:, None] * perp + (b * c)[:, None] * h
            e = rng.gamma(2.0, 1.0 + 5 * np.exp(-0.5 * ((k - 10) / 4) ** 2), N_CELLS)
            t = (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT + np.abs(a) * 1e-10 + rng.exponential(2e-9, N_CELLS)
            cells.append(np.column_stack([xy, np.full(N_CELLS, k), e, t]))
        out.append(np.concatenate(cells))
    return out


def test_across_track_offset_is_exact():
    d = np.array([0.6, 0.3, 0.2])
    d = d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    axis = 4 * atmosphere.PLANE_SPACING * d[:2]
    cell = np.r_[axis + 50.0 * perp + 300.0 * c * h, 3, 1.0, 4 * atmosphere.PLANE_SPACING / annuli.C_LIGHT + 5e-9]
    cells, pl = suite.tables([cell[None]], d[None])
    assert np.isclose(cells["abs_t1"][0], 50.0) and np.isclose(pl["lead"][0], 5.0) and np.isclose(pl["centre"][0], 50.0)


def test_shifted_source_stands_out_and_a_fresh_draw_does_not():
    rng = np.random.default_rng(3)
    dirs = toy_dirs(rng, N_SHOWERS)
    l, u = rng.uniform(5, 6, N_SHOWERS), dirs[:, 2]
    srcs = {"ref": toy_clouds(rng, dirs), "same": toy_clouds(rng, dirs), "shifted": toy_clouds(rng, dirs, 150.0)}
    sub = {}
    for k, c in srcs.items():
        t = suite.tables(c, dirs)
        sub[k] = {"tables": t, "features": suite.plane_features(t[1], l, u)}
    scale = suite.split_half(sub["ref"], N_SHOWERS, np.arange(N_SHOWERS), rng)
    same, bad = suite.compare(sub["ref"], sub["same"]), suite.compare(sub["ref"], sub["shifted"])
    s, g, b = (scale["abs_t1"]["pooled"], same["abs_t1"]["vs CORSIKA"]["pooled"], bad["abs_t1"]["vs CORSIKA"]["pooled"])
    print(f"      |t1| W1: split-half scale {s:.4f}, fresh draw {g:.4f}, shifted {b:.4f}")
    assert g < 3 * s and b > 10 * s, (s, g, b)
    assert bad["centre"]["vs CORSIKA"]["peak"] > 10 * scale["centre"]["peak"]
    a_same, a_bad = same["classifier AUC"]["vs CORSIKA"], bad["classifier AUC"]["vs CORSIKA"]
    print(f"      AUC fresh draw {a_same:.3f}, shifted {a_bad:.3f}, halves {scale['classifier AUC']['pooled']:.3f}")
    assert a_same < 0.62 and a_bad > 0.9, (a_same, a_bad)
    assert len(same["cells"]["vs CORSIKA"]["per_plane"]) == 24


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

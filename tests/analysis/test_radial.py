"""CPU self-test of the radial law fit (symbolic/radial.py): planes drawn from known NKG shapes, binned into ground
cells and across-track tables the way the data are, must give back their scale and age."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import annuli, lateral, radial, render

TRUTHS = ((20.0, 1.2), (60.0, 0.8), (150.0, 1.6))


def toy_shares(rng, r0, s, n_showers=40, n=4000, cos_z=0.3):
    """Pooled windowed across-track shares of toy planes with an NKG radial profile."""
    w = radial.nkg_weights(np.array([np.log(r0)]), np.array([s]))[0, 0]
    cdf = np.cumsum(w) / w.sum()
    rows = []
    for _ in range(n_showers):
        phi_az = rng.uniform(0, 2 * np.pi)
        d = np.array([np.sqrt(1 - cos_z ** 2) * np.cos(phi_az), np.sqrt(1 - cos_z ** 2) * np.sin(phi_az), cos_z])
        r = np.interp(rng.uniform(0, 1, n), cdf, radial.R_GRID)
        phi = rng.uniform(0, 2 * np.pi, n)
        k = np.full(n, 6)
        xy = render.to_ground(k, r * np.cos(phi), r * np.sin(phi), d)
        cells = render.cluster(k, xy, rng.gamma(1.5, 1.0, n), np.zeros(n))
        h, perp, c = lateral.frame(d)
        e_ann = annuli.plane_profiles(cells, 6, d, h, perp, c, "across")[0][: radial.N_WIN]
        rows.append(e_ann / e_ann.sum())
    return np.array(rows)


def test_fit_recovers_scale_and_age_of_nkg_planes():
    rng = np.random.default_rng(6)
    G = radial.share_table()
    logG = np.log(G + radial.EPS)
    for r0, s in TRUTHS:
        obs = toy_shares(rng, r0, s)
        ir, js = np.unravel_index(radial.fit_planes(obs, logG), G.shape[:2])
        got_r0, got_s = np.exp(np.median(radial.LN_R0[ir])), np.median(radial.AGE[js])
        print(f"      truth r0 {r0:.0f} s {s:.2f} -> fitted r0 {got_r0:.1f} s {got_s:.2f}")
        assert abs(np.log(got_r0 / r0)) < 0.2 and abs(got_s - s) < 0.15, (r0, s, got_r0, got_s)
        assert radial.w1(radial.pooled(G[ir, js]), radial.pooled(obs)) < 0.01


def test_scale_fit_at_a_fixed_age_recovers_the_scale():
    rng = np.random.default_rng(7)
    G = radial.share_table()
    logG = np.log(G + radial.EPS)
    for r0, s in TRUTHS:
        obs = toy_shares(rng, r0, s, n_showers=20)
        js = np.full(len(obs), int(np.rint((s - radial.AGE[0]) / (radial.AGE[1] - radial.AGE[0]))))
        got = np.exp(np.median(radial.LN_R0[radial.fit_scale(obs, logG, js)]))
        print(f"      age fixed at {s:.2f}: truth r0 {r0:.0f} -> fitted {got:.1f}")
        assert abs(np.log(got / r0)) < 0.1, (r0, s, got)


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

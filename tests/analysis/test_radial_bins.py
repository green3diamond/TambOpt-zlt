"""CPU self-test of the ring-share inversion (symbolic/radial_bins.py): toy planes with known shares of energy in
rings of in-plane radius, binned into ground cells and across-track tables the way the data are, must give the
shares back."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import annuli, lateral, radial_bins as rb, render


def toy_obs(rng, q, n_showers=60, n=6000, cos_z=0.3):
    """Across-track shares (inside OUTER) of toy planes whose energy sits in the rings with shares q."""
    rows = []
    for _ in range(n_showers):
        az = rng.uniform(0, 2 * np.pi)
        d = np.array([np.sqrt(1 - cos_z ** 2) * np.cos(az), np.sqrt(1 - cos_z ** 2) * np.sin(az), cos_z])
        ring = rng.choice(len(q), n, p=q)
        lo, hi = rb.RINGS[ring], rb.RINGS[ring + 1]
        r = np.sqrt(rng.uniform(lo ** 2, hi ** 2))
        phi = rng.uniform(0, 2 * np.pi, n)
        k = np.full(n, 6)
        cells = render.cluster(k, render.to_ground(k, r * np.cos(phi), r * np.sin(phi), d), np.ones(n), np.zeros(n))
        h, perp, c = lateral.frame(d)
        e = annuli.plane_profiles(cells, 6, d, h, perp, c, "across")[0][: rb.N_BIN]
        rows.append(e / e.sum())
    return np.array(rows)


def test_inversion_recovers_ring_shares():
    rng = np.random.default_rng(8)
    M = rb.ring_matrix()
    q_true = np.r_[0.05, 0.05, 0.15, 0.20, 0.15, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.0]
    assert len(q_true) == rb.N_RING, (len(q_true), rb.N_RING)
    q_true = q_true / q_true.sum()
    obs = toy_obs(rng, q_true)
    q_planes = rb.ring_shares(obs, M)
    q = q_planes.mean(0)
    pred = rb.across(q_planes, M).mean(0)
    inner = rb.RINGS[1:] <= 20                       # rings within two cells: weakly separable, only their sum counts
    print(f"      inside 20 m: true {q_true[inner].sum():.3f} got {q[inner].sum():.3f}; outer max error "
          f"{np.abs(q - q_true)[~inner].max():.4f}; projected max error {np.abs(pred - obs.mean(0)).max():.4f}")
    assert np.abs(pred - obs.mean(0)).max() < 0.005
    assert abs(q[inner].sum() - q_true[inner].sum()) < 0.02 and np.abs(q - q_true)[~inner].max() < 0.03


def test_a_needle_core_is_reproduced_in_the_first_bins():
    rng = np.random.default_rng(9)
    M = rb.ring_matrix()
    q_true = np.r_[0.70, 0.05, 0.03, 0.02, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.0, 0.0, 0.0]
    q_true = q_true / q_true.sum()
    obs = toy_obs(rng, q_true)
    pred = rb.across(rb.ring_shares(obs, M), M).mean(0)
    print(f"      first two bins observed {obs.mean(0)[:2].round(4)} predicted {pred[:2].round(4)}")
    assert np.max(np.abs(pred[:2] - obs.mean(0)[:2])) < 0.005


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

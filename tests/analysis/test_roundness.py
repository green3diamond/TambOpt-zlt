"""CPU self-test of the roundness measure (symbolic/roundness.py): a round shower in the plane, binned into ground
cells at a steep-ish zenith, must look round; the same shower stretched along track must not."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import render, roundness


def toy_direction(rng, cos_z=0.32):
    """A steep-ish shower at a random azimuth, so the axis falls at a random place within its 10 m cell."""
    phi = rng.uniform(0, 2 * np.pi)
    return np.array([np.sqrt(1 - cos_z ** 2) * np.cos(phi), np.sqrt(1 - cos_z ** 2) * np.sin(phi), cos_z])


def offsets_of(rng, stretch, n_showers=150, n=1500):
    """Pooled symmetrized offsets of toy showers whose in-plane density is round (stretch 1) or stretched along."""
    A, B, w = [], [], []
    for _ in range(n_showers):
        d = toy_direction(rng)
        k = rng.choice([4, 8, 12], n)
        r = 30 * rng.pareto(2.5, n)                    # heavy-tailed radial profile
        phi = rng.uniform(0, 2 * np.pi, n)
        xy = render.to_ground(k, r * np.cos(phi), stretch * r * np.sin(phi), d)
        a, b, ww = roundness.offsets(render.cluster(k, xy, rng.gamma(1.5, 1.0, n), np.zeros(n)), d, rng)
        A.append(a), B.append(b), w.append(ww)
    return np.concatenate(A), np.concatenate(B), np.concatenate(w)


def test_round_showers_look_round_and_stretched_ones_do_not():
    rng = np.random.default_rng(5)
    res = {}
    for stretch in (1.0, 1.5):
        A, B, w = offsets_of(rng, stretch)
        res[stretch] = roundness.w1(A, B, w, w, roundness.RULE_WINDOW)
    print(f"      W1 round {res[1.0]:.4f}, stretched 1.5x {res[1.5]:.4f}")
    assert res[1.0] < 0.02 and res[1.5] > 3 * res[1.0], res


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

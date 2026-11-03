"""CPU self-test of the particle draw (symbolic/draw.py) on a toy law with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import annuli, atmosphere, draw, particles

N_F = 8


def toy_law():
    """Stored share 0.9, rings 0-15 equal shares and the rest empty, mean energy 0.05 GeV, lead 5 ns, mean 20 ns."""
    stored = np.zeros((N_F, 2))
    stored[0, 0] = np.log(9.0)
    rings = np.zeros((N_F, particles.N_RING))
    rings[0, particles.OUTER:] = -50.0
    ring_law = lambda v: np.column_stack([np.full(particles.N_RING, v), np.zeros((particles.N_RING, N_F - 1))])
    return {"stored": stored, "rings": rings, "mean_energy": ring_law(np.log(0.05)), "lead": ring_law(np.log10(6.0)),
            "delay": ring_law(np.log10(21.0)), "spread": np.tile([3.0, 0.8], (particles.N_RING, 1))}


def test_plane_particles_carry_the_law():
    rng = np.random.default_rng(1)
    x = np.r_[1.0, np.zeros(N_F - 1)]
    k, e_k = 6, 900.0
    p = draw.plane_particles(rng, k, x, e_k, toy_law(), "electron", np.zeros(particles.OUTER))
    assert np.isclose(p[:, 3].sum(), e_k / 0.9, rtol=1e-6)                       # energy before the cut
    assert p[:, 3].min() >= particles.E_MIN["electron"]
    assert abs(len(p) / (e_k / 0.9 / 0.05) - 1) < 0.03                           # count = energy / mean energy
    r = np.hypot(p[:, 1], p[:, 2])
    assert r.max() <= particles.EDGES[particles.OUTER]                           # empty rings stay empty
    ring = np.searchsorted(particles.EDGES, r, side="right") - 1
    share = np.bincount(ring, p[:, 3], particles.N_RING)[: particles.OUTER] / p[:, 3].sum()
    assert np.allclose(share, 1 / particles.OUTER, atol=1e-9)                   # ring energies exact
    delay = (p[:, 4] - (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT) * 1e9
    assert delay.min() >= 5.0 - 1e-3 and abs(delay.mean() - 20.0) < 0.5


def test_noise_offsets_and_ring_fill():
    rng = np.random.default_rng(2)
    net = [(rng.normal(size=(15, 4)), np.zeros(4)), (rng.normal(size=(4, particles.OUTER)), np.zeros(particles.OUTER))]
    off = draw.noise(net, rng.normal(size=(7, N_F)), "muon", rng)
    assert off.shape == (7, particles.OUTER) and np.isfinite(off).all()
    B = np.arange(12.0).reshape(6, 2)
    B[[0, 3, 5]] = np.nan
    F = draw.fill_rings(B)
    assert np.array_equal(F[0], B[1]) and np.array_equal(F[3], B[2]) and np.array_equal(F[5], B[4])


def test_time_law_replaces_the_ring_laws_only_when_given():
    law, x = toy_law(), np.eye(N_F)[0]
    lead, delay = draw.ring_times(law, x, 5)
    assert np.allclose(lead, np.log10(6.0)) and np.allclose(delay, np.log10(21.0))
    theta = np.array([0.6, 0.35, -0.02, 0.8])
    law["time"] = {"form": "c", "lead": theta, "delay": 2 * theta}
    lead, delay = draw.ring_times(law, x, 5)
    assert lead.shape == (particles.N_RING,) and np.array_equal(lead, times_sr_c(theta, x, 5))


def times_sr_c(theta, x, k):
    from symbolic import times_sr
    return times_sr.ring_times(theta, "c", x, k)


def test_time_law_must_come_with_non_default_laws():
    assert draw.time_law_path(draw.LAWS, None) == draw.TIME_LAW
    assert draw.time_law_path("other", "mine.npz") == "mine.npz"
    assert draw.time_law_path("other", "none") is None
    try:
        draw.time_law_path("other", None)
    except ValueError:
        return
    raise AssertionError("a fold's laws drew with the time law of all showers")


def test_width_law_must_come_with_non_default_laws():
    assert draw.width_law_path(draw.LAWS, None) == draw.WIDTH_LAW
    assert draw.width_law_path("fold0/particle_laws.npz", "fold0/width_law.npz") == "fold0/width_law.npz"
    try:
        draw.width_law_path("fold0/particle_laws.npz", None)
    except ValueError:
        return
    raise AssertionError("a fold's laws must not fall back to the default width law")


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

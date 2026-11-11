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
    p = draw.plane_particles(rng, k, x, e_k, toy_law(), "electron")
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


def test_ring_fill():
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


def test_energy_law_must_come_with_non_default_laws():
    assert draw.energy_law_path(draw.LAWS, None) == draw.ENERGY_LAW
    assert draw.energy_law_path("other", "mine.npz") == "mine.npz"
    assert draw.energy_law_path("other", "none") is None
    try:
        draw.energy_law_path("other", None)
    except ValueError:
        return
    raise AssertionError("a fold's laws drew with the mean energy law of all showers")


def test_width_law_must_come_with_non_default_laws():
    assert draw.width_law_path(draw.LAWS, None) == draw.WIDTH_LAW
    assert draw.width_law_path("fold0/particle_laws.npz", "fold0/width_law.npz") == "fold0/width_law.npz"
    try:
        draw.width_law_path("fold0/particle_laws.npz", None)
    except ValueError:
        return
    raise AssertionError("a fold's laws must not fall back to the default width law")


def test_total_energy_inputs_divide_by_the_stored_share():
    law = toy_law()
    X, ks, e_k = np.linspace(300, 900, 3), np.array([4, 8, 12]), np.array([10.0, 100.0, 1000.0])
    x_s, x_t = draw.plane_inputs(X, 600.0, 5.8, 0.8, e_k, ks, {**law, "plane_energy": "total"})
    assert np.allclose(x_t[:, 6] - x_s[:, 6], -np.log(0.9))                       # total = stored / 0.9
    assert np.allclose(np.delete(x_t, 6, 1), np.delete(x_s, 6, 1))
    x_s2, x_2 = draw.plane_inputs(X, 600.0, 5.8, 0.8, e_k, ks, {**law, "plane_energy": "stored"})
    assert np.array_equal(x_2, x_s2) and np.array_equal(x_s2, x_s)
    p = draw.plane_particles(np.random.default_rng(1), 6, x_t[0], 900.0, law, "electron", x_stored=x_s[0])
    assert np.isclose(p[:, 3].sum(), 900.0 / 0.9, rtol=1e-6)                       # stored law reads x_stored


def test_total_plane_energies_skip_the_stored_law():
    law = {**toy_law(), "plane_energy": "total"}
    X, ks, e_k = np.linspace(300, 900, 3), np.array([4, 8, 12]), np.array([10.0, 100.0, 1000.0])
    x_s, x = draw.plane_inputs(X, 600.0, 5.8, 0.8, e_k, ks, law, "total")
    x_ref, _ = draw.plane_inputs(X, 600.0, 5.8, 0.8, e_k, ks, law)
    assert np.array_equal(x, x_s) and np.array_equal(x, x_ref)                     # inputs from e_k itself
    p = draw.plane_particles(np.random.default_rng(1), 6, x[0], 900.0, law, "electron", total=True)
    assert np.isclose(p[:, 3].sum(), 900.0, rtol=1e-9)                            # no division by the stored share
    try:
        draw.plane_inputs(X, 600.0, 5.8, 0.8, e_k, ks, {**law, "plane_energy": "stored"}, "total")
    except ValueError as exc:
        assert "fitted on total" in str(exc)
    else:
        raise AssertionError("total energies accepted laws fitted on stored energy")


def test_total_plane_energies_refuse_old_caps_and_store_what_the_render_keeps():
    from modules.constants import SPECIES_NAMES
    law = {**toy_law(), "plane_energy": "total", "along": np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0]}
    laws = {(s, cn): law for s in SPECIES_NAMES for cn in ("EM", "hadronic")}
    E = {s: np.r_[np.zeros(5), np.full(3, 50.0), np.zeros(16)] for s in SPECIES_NAMES}
    job = (0, 1, 5.5, 0.8, "EM", np.linspace(600, 1600, 24), np.array([0.0, 0.6, 0.8]), E,
           {s: 900.0 for s in SPECIES_NAMES}, laws)
    try:
        draw.shower((*job, "old", "total"))
    except ValueError as exc:
        assert "--caps new" in str(exc)
    else:
        raise AssertionError("total energies accepted the old caps")
    _, new, _, _ = draw.shower((*job, "new", "total"))
    for s in SPECIES_NAMES:
        e = np.bincount(new[s][:, 2].astype(int), new[s][:, 3], 24)[5:8]
        assert (e <= 50.0 + 1e-6).all() and (e > 0.8 * 50.0).all()                # at most the total, as CORSIKA


def test_energy_kind_defaults_to_stored_for_old_files():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "a.npz"), rows=np.arange(2))
        np.savez(os.path.join(d, "b.npz"), rows=np.arange(2), plane_energy=np.array("total"))
        assert draw.energy_kind(np.load(os.path.join(d, "a.npz"))) == "stored"
        assert draw.energy_kind(np.load(os.path.join(d, "b.npz"))) == "total"


def test_new_caps_store_what_the_render_keeps_without_rescaling():
    from modules.constants import SPECIES_NAMES
    from symbolic import width_sr
    law = {**toy_law(), "plane_energy": "stored", "along": np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0]}
    laws = {(s, cn): law for s in SPECIES_NAMES for cn in ("EM", "hadronic")}
    E = {s: np.r_[np.zeros(5), np.full(3, 50.0), np.zeros(16)] for s in SPECIES_NAMES}
    X = np.linspace(600, 1600, 24)
    d = np.array([0.0, 0.6, 0.8])
    job = (0, 1, 5.5, 0.8, "EM", X, d, E, {s: 900.0 for s in SPECIES_NAMES}, laws)
    _, old, _, _ = draw.shower(job)
    _, new, _, _ = draw.shower((*job, "new"))
    for s in SPECIES_NAMES:
        k = old[s][:, 2].astype(int)
        assert np.allclose(np.bincount(k, old[s][:, 3], 24)[5:8], 50.0)            # old: rescaled to E_k
        e_new = np.bincount(new[s][:, 2].astype(int), new[s][:, 3], 24)[5:8]
        assert (e_new < 50.0 / 0.9 + 1e-6).all() and (e_new > 0.8 * 50.0 / 0.9).all()   # new: what the render keeps


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

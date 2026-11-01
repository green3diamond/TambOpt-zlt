"""CPU self-test of the particle tables (symbolic/particles.py stage A) on toy particles with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import particles, render

D = np.array([0.55, -0.75, 0.2]) / np.linalg.norm([0.55, -0.75, 0.2])


def toy(rng, n=8000, planes=(0, 5, 11, 23)):
    k = rng.choice(planes, n)
    a, b = rng.normal(0, 60, n) * rng.choice([0.05, 1, 20], n), rng.normal(0, 60, n)
    p = np.column_stack([k, a, b, rng.gamma(1.2, 2.0, n), rng.uniform(0, 50e-9, n) + k * 1.7e-6])
    return p, np.hypot(a, b)


def test_tables_match_brute_force_and_render():
    rng = np.random.default_rng(3)
    p, r = toy(rng)
    n_max = 900
    tab = particles.tables(p, r, D, n_max)
    k, e = p[:, 0].astype(int), p[:, 3]
    keep = render.radius_cut(k, render.to_ground(k, p[:, 1], p[:, 2], D))
    ring = np.array([np.flatnonzero(particles.EDGES <= x)[-1] for x in r])
    for plane in range(particles.N_PLANE):
        for i in range(particles.N_RING):
            m = (k == plane) & (ring == i)
            assert tab["ring_count"][plane, i] == m.sum()
            assert tab["ring_count_cut"][plane, i] == (m & keep).sum()
            assert np.isclose(tab["ring_energy"][plane, i], e[m].sum())
            assert np.isclose(tab["ring_energy_cut"][plane, i], e[m & keep].sum())
            assert tab["hist"][plane, i].sum() == m.sum()
    assert np.allclose(tab["E_total"], tab["ring_energy"].sum(1))
    assert np.allclose(tab["E_cut"], tab["ring_energy_cut"].sum(1))
    cells = render.render(p, D, n_max)
    assert np.allclose(tab["E_stored"], np.bincount(cells[:, 2].astype(int), cells[:, 3], particles.N_PLANE))
    assert tab["n_cells"] > n_max                                   # the cap acts, so E_stored < E_cut is tested
    assert (tab["E_stored"] <= tab["E_cut"] + 1e-9).all() and tab["E_stored"].sum() < tab["E_cut"].sum()


def test_histogram_bins_log_energy_over_ring_mean():
    p = np.array([[3, 0.5, 0.0, 1.0, 0.0], [3, 0.0, 0.4, 3.0, 0.0],
                  [3, 300.0, 0.0, 1e-9, 0.0], [3, 0.0, 310.0, 1.0, 0.0]])
    r = np.hypot(p[:, 1], p[:, 2])
    h = particles.tables(p, r, D, 10)["hist"][3]
    assert h.sum() == 4 and h[0].sum() == 2 and h[12].sum() == 2    # rings [0, 1) m and [280, 400) m
    assert h[0, 56] == 1 and h[0, 61] == 1                          # log10(1 / 2) = -0.30, log10(3 / 2) = 0.18
    assert h[12, 0] == 1 and h[12, 63] == 1                         # log10(2e-9) clipped into the first bin; log10(2)


def test_check_flags_a_count_mismatch():
    rng = np.random.default_rng(4)
    p, r = toy(rng, 2000)
    tab = particles.tables(p, r, D, 500)
    stats = {"ring_count": tab["ring_count"].copy(), "ring_energy": tab["ring_energy"] * (1 + 1e-6),
             "energy": tab["E_total"], "count": tab["ring_count"].sum(1)}
    assert particles.check(tab, stats)["pass"]
    stats["ring_count"][5, 4] += 1
    assert not particles.check(tab, stats)["pass"]


def toy_rows(rng, n_showers=40, offset=0.0):
    """Planes of toy showers with a known linear law plus a per-shower and a per-plane noise; `offset` shifts the
    held-out half (a wrong law there)."""
    unit = np.repeat(np.arange(n_showers), 24)
    x = np.column_stack([np.ones(len(unit)), rng.normal(size=(len(unit), 3))])
    b = np.array([0.5, 1.0, -0.7, 0.3])
    test = particles.split_showers(unit, np.full(len(unit), "EM"), rng.uniform(5, 6, n_showers))
    y = x @ b + rng.normal(0, 0.3, n_showers)[unit] + rng.normal(0, 0.2, len(unit)) + offset * test
    return x, y, test, unit, b


def test_ridge_law_passes_when_right_and_fails_when_shifted():
    x, y, test, unit, b = toy_rows(np.random.default_rng(5))
    assert abs(test.mean() - 0.5) < 0.01 and len(np.unique(unit[test])) == 20
    res, B = particles.run_law("ring", x, y[:, None], np.ones((len(y), 1), bool), test, unit)
    assert res["ring 0"]["pass"] and np.allclose(B[0], b, atol=0.15)
    x, y, test, unit, _ = toy_rows(np.random.default_rng(5), offset=0.6)
    assert particles.run_law("ring", x, y[:, None], np.ones((len(y), 1), bool), test, unit)[0]["ring 0"]["pass"] is False


def test_share_law_recovers_a_fractional_logit():
    rng = np.random.default_rng(6)
    x, _, test, unit, b = toy_rows(rng)
    y = 1 / (1 + np.exp(-(x @ b)))
    res, B = particles.run_law("share", x, y, np.ones(len(y), bool), test, unit)
    assert np.allclose(B[:, 0] - B[:, 1], b, atol=2e-2) and res["R2"] > 0.99


def test_burr_draws_have_mean_one_and_the_fit_recovers_them():
    rng = np.random.default_rng(8)
    c, k = 2.5, 0.9
    v = particles.burr_sample(rng, c, k, 400000)
    assert abs(np.mean(particles.burr_cdf(v, c, k)) - 0.5) < 0.005           # draws follow the CDF (PIT uniform)
    big = particles.burr_sample(rng, 4.0, 1.0, 2000000)
    assert abs(big.mean() - 1) < 0.01                                        # mean 1 (finite variance case)
    e_min, means = 0.01, np.array([0.03, 0.2, 1.5])                         # rings of different mean energy
    h = np.zeros((len(means), len(particles.HIST) - 1))
    for i, m in enumerate(means):
        e = particles.energies(rng, c, k, m, e_min, 200000)
        assert e.min() >= e_min
        u = np.log10(e / m)                                                  # tables bin by the ring's actual mean
        h[i] = np.bincount(np.clip(np.searchsorted(particles.HIST, u, side="right") - 1, 0, h.shape[1] - 1),
                           minlength=h.shape[1])
    (cf, kf), w1, _ = particles.fit_burr(h, means, e_min)
    assert abs(cf - c) < 0.1 and abs(kf - k) < 0.1 and w1 < 0.02


def test_multinomial_stays_finite_with_a_nearly_empty_ring():
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(500), rng.normal(size=(500, 2))])
    w = rng.dirichlet([5, 3, 2], 500)
    w = np.column_stack([w, np.zeros(500)])
    w[x[:, 1] > 2, 3] = 0.2                                                  # separable: only far-out planes use it
    w /= w.sum(1, keepdims=True)
    B = particles.fit_multinomial(x, w)
    assert np.isfinite(B).all() and np.abs(B).max() < 50


def test_stretch_moves_energy_outward_and_widths_are_recovered():
    q = np.diff(np.r_[0.0, 1 - np.exp(-particles.EDGES[1:-1] / 40.0), 1.0])         # an exponential-ish profile
    assert np.allclose(particles.stretch(q, 0.0), q, atol=1e-5)
    wide = particles.stretch(q, 0.5)
    assert np.isclose(wide.sum(), 1.0) and (np.cumsum(wide)[:-1] <= np.cumsum(q)[:-1] + 1e-9).all()
    truth = np.array([-0.8, -0.2, 0.0, 0.3, 1.1])
    got = particles.plane_widths(np.array([particles.stretch(q, v) for v in truth]), np.tile(q, (len(truth), 1)))
    assert np.allclose(got, truth, atol=0.011)


def test_width_law_recovers_slope_and_memory():
    rng = np.random.default_rng(10)
    unit, k = np.repeat(np.arange(60), 24), np.tile(np.arange(24), 60)
    x = np.column_stack([np.ones(len(unit)), rng.normal(size=(len(unit), 2))])
    e = np.concatenate([0.15 * particles_ar1(rng, 0.8) for _ in range(60)])
    lam = x @ np.array([-0.3, 0.4, 0.0]) + e
    test = particles.split_showers(unit, np.full(len(unit), "EM"), rng.uniform(5, 6, 60))
    w, res = particles.width_law(x, lam, unit, k, test)
    assert np.allclose(w[:3], [-0.3, 0.4, 0.0], atol=0.03) and abs(w[3] - 0.15) < 0.02 and abs(w[4] - 0.8) < 0.05
    assert res["held-out R2"] > 0.8


def particles_ar1(rng, rho, n=24):
    a = np.empty(n)
    a[0] = rng.normal()
    for j in range(1, n):
        a[j] = rho * a[j - 1] + np.sqrt(1 - rho ** 2) * rng.normal()
    return a


def test_too_few_held_out_showers_are_not_judged():
    x, y, test, unit, _ = toy_rows(np.random.default_rng(7), n_showers=6)
    assert particles.run_law("ring", x, y[:, None], np.ones((len(y), 1), bool), test, unit)[0]["ring 0"]["pass"] is None


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

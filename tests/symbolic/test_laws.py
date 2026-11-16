"""CPU self-test of the in-plane laws: each fit recovers known constants from a toy made with that law."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np
from scipy.special import softmax

from symbolic import data, energy, rings, rows, spread, times, width

EDGES = data.EDGES[1:-1]


def ar1_toy(rng, rho, n=24):
    a = [rng.normal()]
    for _ in range(n - 1):
        a.append(rho * a[-1] + np.sqrt(1 - rho ** 2) * rng.normal())
    return a


def test_rings_recover_a_multinomial_and_stay_finite_with_an_empty_ring():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(4000), rng.normal(size=(4000, 2))])
    B = np.array([[0.0, 0.5, -0.3], [0.0, 0.8, 0.1], [0.0, -0.4, 0.6]])
    got = rings.fit(x, softmax(x @ B, axis=1), alpha=0.0)
    assert np.allclose(got, B, atol=1e-3)
    w = np.column_stack([rng.dirichlet([5, 3, 2], 500), np.zeros(500)])
    w[x[:500, 1] > 2, 3] = 0.2                                               # separable: only far-out planes use it
    assert np.isfinite(rings.fit(x[:500], w / w.sum(1, keepdims=True))).all()


def test_stretch_moves_energy_outward_and_widths_are_recovered_on_the_band():
    q = np.diff(np.r_[0.0, 1 - (1 + EDGES / 100.0) ** -1.5, 1.0])
    assert np.allclose(rings.stretch(q, 0.0), q, atol=1e-5)
    wide = rings.stretch(q, 0.5)
    assert np.isclose(wide.sum(), 1.0) and (np.cumsum(wide)[:-1] <= np.cumsum(q)[:-1] + 1e-9).all()
    truth = np.array([-0.8, -0.2, 0.0, 0.3, 1.1])
    got = rings.plane_widths(np.array([rings.stretch(q, v) for v in truth]), np.tile(q, (len(truth), 1)))
    assert np.allclose(got, truth, atol=0.011)
    c = np.cumsum(rings.stretch(q, 0.3))[:-1]
    c[EDGES < 10.0] = c[EDGES < 10.0] ** 3                                   # the core changes, the band does not
    assert rings.BAND.sum() == 10 and abs(rings.plane_widths(np.diff(np.r_[0.0, c, 1.0])[None], q[None])[0] - 0.3) < 0.011


def test_width_recovers_tanh_spread_and_memory():
    rng = np.random.default_rng(2)
    p, L = rng.uniform(0, 3.2, 800), rng.normal(size=(800, 2))
    y = 1.3 * np.tanh(1.5 * (p - 1.8)) - 0.6 + L @ np.array([0.2, -0.1]) + rng.normal(0, 0.01, 800)
    theta = width.fit_tanh(p, L, y)
    if theta[2] < 0:                                                         # (-A, -B) is the same curve
        theta[[0, 2]] *= -1
    assert np.allclose(theta, [1.5, 1.8, 1.3, -0.6, 0.2, -0.1], atol=0.03)
    rng = np.random.default_rng(3)
    k, unit = np.tile(np.arange(24), 300), np.repeat(np.arange(300), 24)
    z = np.concatenate([ar1_toy(rng, 0.8) for _ in range(300)])
    a, b = width.fit_spread(np.log(k + 1.0), np.exp(-1.0 - 0.5 * np.log(k + 1.0)) * z)
    assert abs(a + 1.0) < 0.05 and abs(b + 0.5) < 0.03 and abs(width.lag_one(z, unit, k) - 0.8) < 0.02
    w = np.array([1.5, 1.8, 1.3, -0.6, 0.2, -0.1, -1.0, -0.5, 0.8])
    k, l, age = np.array([0, 5, 23]), np.array([0.1, 0.0, -0.2]), np.array([0.3, 1.0, 1.5])
    assert np.allclose(width.mean(w, k, l, age), 1.3 * np.tanh(1.5 * (np.log(k + 1.0) - 1.8)) - 0.6 + 0.2 * l - 0.1 * age)
    assert np.allclose(width.sd(w, k), np.exp(-1.0 - 0.5 * np.log(k + 1.0)))


def toy_rows(rng, n=300):
    """Plane rows with every field a time or energy law reads."""
    x = np.column_stack([np.ones(n), rng.uniform(0.3, 1.8, n), np.zeros(n), rng.uniform(-0.5, 1.0, n),
                         rng.uniform(0.05, 0.9, n), np.zeros(n), rng.normal(0, 1, n), rng.normal(0.5, 0.3, n)])
    return {"x": x, "k": rng.integers(0, 24, n), "i": np.arange(n) // 6, "ring_count": rng.integers(0, 50, (n, data.N_RING))}


def test_times_recover_a_power_law_and_skip_empty_rings():
    rng = np.random.default_rng(3)
    r = toy_rows(rng)
    X = times.expand(rows.plane_inputs(r["x"], r["k"]))
    theta = np.array([-4.0, 1.6, 0.7])
    r["lead"] = times.law(theta, X).reshape(-1, data.N_RING)
    r["lead"][r["ring_count"] == 0] = np.nan
    assert np.allclose(times.fit(r, "lead"), theta, atol=1e-4)
    assert np.allclose(times.ring_times(theta, r["x"][7], r["k"][7]), times.law(theta, X[7 * data.N_RING:8 * data.N_RING]))
    # r = 100 m, plane 12: extra path 100^2 / (2 x 6000 m) = 0.833 m = 2.78 ns
    t = 10 ** times.law((times.GEOMETRIC, 2.0, 1.0), np.array([[np.log(100.0), 1.0, np.log(12.0)]]))[0] - 1
    assert abs(t - 2.78) < 0.01


def test_energy_constants_on_raw_inputs_equal_the_trained_sum():
    rng = np.random.default_rng(4)
    r = toy_rows(rng, 600)
    X = times.expand(rows.plane_inputs(r["x"], r["k"]))
    y = (np.tanh(X[:, 0] - 3.0) + 0.5 * np.tanh(X[:, 6])).reshape(-1, data.N_RING)
    r["ring_energy"] = np.exp(y) * r["ring_count"]
    err, theta = energy.fit(r, seeds=1, epochs=300)
    assert len(theta) == 9 * energy.TERMS + 1 and np.isfinite(err)
    m = r["ring_count"].ravel() > 0
    assert np.sqrt(np.mean((energy.tanh_sum(theta, X[m]) - y.ravel()[m]) ** 2)) < 0.5 * y.ravel()[m].std()
    assert np.allclose(energy.ring_means(theta, r["x"][5], r["k"][5]), energy.tanh_sum(theta, X[5 * data.N_RING:6 * data.N_RING]))


def test_burr_draws_have_mean_one_and_the_fit_recovers_them():
    rng = np.random.default_rng(5)
    c, k = 2.5, 0.9
    assert abs(np.mean(spread.cdf(spread.sample(rng, c, k, 400000), c, k)) - 0.5) < 0.005    # draws follow the CDF
    assert abs(spread.sample(rng, 4.0, 1.0, 2000000).mean() - 1) < 0.01
    e_min, means = 0.01, np.array([0.03, 0.2, 1.5])
    h = np.zeros((len(means), len(data.HIST) - 1))
    for i, m in enumerate(means):
        e = e_min + (m - e_min) * spread.sample(rng, c, k, 200000)
        u = np.log10(e / m)
        h[i] = np.bincount(np.clip(np.searchsorted(data.HIST, u, side="right") - 1, 0, h.shape[1] - 1), minlength=h.shape[1])
    cf, kf = spread.fit_ring(h, means, e_min)
    assert abs(cf - c) < 0.1 and abs(kf - k) < 0.1


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

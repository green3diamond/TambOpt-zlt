"""CPU self-test of the tanh-sum form (symbolic/tiny_net.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import energy_sr, tiny_net


def test_numbers_count_every_constant():
    assert tiny_net.numbers(1) == 10 and tiny_net.numbers(4) == 37


def test_two_terms_recover_a_two_term_toy_and_one_cannot():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 7))
    y = np.tanh(2 * X[:, 0] - 1) - 0.7 * np.tanh(1.5 * X[:, 1] + 1)
    val = rng.random(3000) < 0.2
    err = {u: tiny_net.train(X, y, val, u, 0, epochs=1500)[0] for u in (1, 2)}
    assert err[2] < 0.05 and err[1] > 2 * err[2], err


def test_folded_constants_equal_the_standardised_sum():
    rng = np.random.default_rng(1)
    X = rng.normal(3.0, 2.0, size=(500, 7))
    w, b, a, c = rng.normal(size=(3, 7)), rng.normal(size=3), rng.normal(size=(1, 3)), rng.normal(size=1)
    mu, sd, y_mu, y_sd = X.mean(0), X.std(0), 1.7, 0.4
    want = (np.tanh((X - mu) / sd @ w.T + b) @ a.T + c).ravel() * y_sd + y_mu
    theta = tiny_net.constants(w, b, a, c, mu, sd, y_mu, y_sd)
    assert len(theta) == tiny_net.numbers(3)
    assert np.allclose(energy_sr.tanh_sum(theta, X), want)


def test_saved_constants_reproduce_the_network_error():
    rng = np.random.default_rng(2)
    X = rng.normal(5.0, 3.0, size=(3000, 7))
    y = 2.0 + 0.5 * np.tanh(X[:, 0] - 5) - 0.3 * np.tanh(X[:, 2] - 4)
    val = rng.random(3000) < 0.2
    best, predict, theta = tiny_net.train(X, y, val, 2, 0, epochs=500)
    mse = np.mean((predict(X[val]) - y[val]) ** 2) / y[~val].var()
    assert abs(mse - best) < 1e-3 * max(best, 1e-3) + 1e-6, (mse, best)
    v = rng.normal(size=(4, 6))
    from symbolic import times_sr
    assert np.allclose(energy_sr.predict(theta, "tanh", v), energy_sr.tanh_sum(theta, times_sr.expand(v)).reshape(4, -1))


def test_smallest_passing_uses_both_bars():
    r = {1: [1.6, 1.0, 1.0, 1.0], 2: [1.2, 1.3, 1.3, 1.3], 3: [1.0, 1.0, 1.0, 1.0], 4: [0.5, 0.5, 0.5, 0.5]}
    assert tiny_net.smallest_passing(r) == 3
    assert tiny_net.smallest_passing({1: [2.0, 2.0, 2.0, 2.0]}) is None


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

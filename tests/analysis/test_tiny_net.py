"""CPU self-test of the tanh-sum form (symbolic/tiny_net.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import tiny_net


def test_numbers_count_every_constant():
    assert tiny_net.numbers(1) == 10 and tiny_net.numbers(4) == 37


def test_two_terms_recover_a_two_term_toy_and_one_cannot():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 7))
    y = np.tanh(2 * X[:, 0] - 1) - 0.7 * np.tanh(1.5 * X[:, 1] + 1)
    val = rng.random(3000) < 0.2
    err = {u: tiny_net.train(X, y, val, u, 0, epochs=1500)[0] for u in (1, 2)}
    assert err[2] < 0.05 and err[1] > 2 * err[2], err


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

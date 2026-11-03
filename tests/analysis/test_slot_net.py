"""CPU self-test of the slot network (symbolic/slot_net.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import slot_net


def test_split_keeps_each_shower_on_one_side():
    showers = np.repeat(np.arange(20), 5)
    val = slot_net.split_showers(showers, np.random.default_rng(0))
    assert val.sum() == 20
    for s in range(20):
        assert val[showers == s].all() or not val[showers == s].any()


def test_passes_needs_the_mean_and_every_group():
    assert slot_net.passes([0.8, 0.9, 1.0, 1.2])
    assert not slot_net.passes([0.5, 0.5, 0.5, 1.3])
    assert not slot_net.passes([1.1, 1.1, 1.1, 1.1])


def test_best_by_size_takes_the_lowest_search_loss():
    a, b = {3: ("p", "x", 0.2), 5: ("q", "y", 0.1)}, {3: ("r", "z", 0.1)}
    got = slot_net.best_by_size([a, b])
    assert got[3][1] == "z" and got[5][1] == "y"


def test_network_learns_a_group_dependent_toy_and_beats_the_mean():
    rng = np.random.default_rng(0)
    n = 6000
    X = np.column_stack([rng.uniform(0, 8, n), rng.uniform(0, 3, n)])
    g = rng.integers(0, 2, n)
    y = np.tanh(X[:, 0] - 4) * (1 + g) + 0.3 * X[:, 1]
    Q = np.column_stack([X, g == 0, g == 1]).astype(float)
    val = rng.random(n) < 0.2
    predict = slot_net.train(Q, y, val, 0, max_epochs=150)
    assert np.mean((predict(Q) - y) ** 2) < 0.03 * y.var()


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

"""CPU self-test of the structured network (symbolic/struct_net.py) on toy rows with known pieces."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import struct_net


def toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(2.0, 1.5, size=(n, 7))
    f = np.tanh(X[:, 0] - X[:, 2])
    y = 0.5 + f + 0.3 * X[:, 3] - 0.2 * X[:, 5]
    return X, y, f, rng.random(n) < 0.2


def test_additive_toy_is_recovered_and_f_holds_its_own_piece():
    X, y, f, val = toy()
    best, predict, saved = struct_net.train(X, y, val, 0, epochs=1500)
    assert np.sqrt(np.mean((predict(X[val]) - y[val]) ** 2)) < 0.05
    got = struct_net.predict(saved, X, "f")
    assert np.corrcoef(got, f)[0, 1] > 0.99
    assert np.std((got - got.mean()) - (f - f.mean())) < 0.05


def test_pieces_do_not_see_each_others_inputs():
    X, y, f, val = toy()
    saved = struct_net.train(X, y, val, 0, epochs=50)[2]
    Q = X[:5].copy()
    moved = Q.copy()
    moved[:, list(struct_net.PIECES["h"])] += 1.0
    assert np.allclose(struct_net.predict(saved, Q, "f"), struct_net.predict(saved, moved, "f"))


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

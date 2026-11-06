"""CPU self-test of the piece-by-piece composition in symbolic/struct_sr.py."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import struct_sr, times_sr


def test_judged_inputs_follow_the_expanded_rows():
    v = np.random.default_rng(0).normal(size=(5, 6))
    jf, jh = struct_sr.judged_inputs(v)
    X = times_sr.expand(v)
    assert np.allclose(jf[:, 0], X[:, 0]) and np.allclose(jf[:, 1], X[:, 2]) and np.allclose(jf[:, 2], X[:, 0] - X[:, 2])
    assert np.allclose(jh, X[:: times_sr.N_RING][:, [1, 3, 4, 5, 6]])


def test_exact_pieces_compose_to_the_network():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(4, 6))
    jf, jh = struct_sr.judged_inputs(v)
    f, h = np.tanh(jf[:, 2]), 0.3 * jh[:, 1] - 0.1 * jh[:, 3]
    net = 1.5 + (f - 0.2).reshape(-1, times_sr.N_RING) + (h + 0.4)[:, None]
    q = {"k.base": 1.5 - 0.2 + 0.4}
    got = struct_sr.compose(q, "k", {7: (f, "f", 0.0)}, {9: (h, "h", 0.0)}, 7, 9)
    assert np.allclose(got, net)


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

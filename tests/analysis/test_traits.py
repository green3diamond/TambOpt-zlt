"""CPU self-test of the noise-input trait model (symbolic/traits.py) on toy showers through the real projection: one
true trait per shower, shared by the species with different strengths, tilts every plane; planes add Dirichlet
scatter. K = 1 must pass rules 1-3 and K = 0 must fail, and the fitted traits of the training showers must follow the
true ones (|corr| >= 0.9); with planes shuffled across showers (no trait left) K = 0 must pass."""
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import radial_bins, traits, variety

N_K = 24
COEF = np.array([1.0, 0.5, 1.0])               # trait strength per species
EPOCHS = 1000


def toy(rng, n_sh=1200):
    M = radial_bins.ring_matrix()
    S, n_r, n_f = traits.S, radial_bins.N_RING, 3
    X = np.concatenate([np.ones((n_sh, S, N_K, 1)), rng.standard_normal((n_sh, S, N_K, n_f - 1))], -1)
    base = np.zeros((n_sh, S, N_K, n_r))
    for j in range(S):
        B = np.column_stack([np.zeros(n_f), rng.normal(0, 0.2, (n_f, n_r - 1))])
        B[0, 1:] -= 0.4 * np.arange(1, n_r)
        base[:, j] = X[:, j] @ B
    z = rng.standard_normal(n_sh)
    eta = base.copy()
    eta[..., variety.INNER] += 0.6 * (z[:, None, None, None] * COEF[None, :, None, None]) * variety.tilt_direction()
    qbar = np.exp(eta - eta.max(-1, keepdims=True))
    qbar /= qbar.sum(-1, keepdims=True)
    mask = rng.random((n_sh, S, N_K)) < 0.8
    q = np.maximum(rng.gamma(np.maximum(300.0 * qbar[mask], 1e-6)), 1e-300)
    O = np.zeros((n_sh, S, N_K, radial_bins.N_BIN))
    O[mask] = radial_bins.across(q / q.sum(1, keepdims=True), M)
    return {"i": np.arange(n_sh), "O": O, "X": X, "base": base, "mask": mask, "z": z}, M


def fit_judge(K, t, M, rng):
    test = t["i"] % 5 == 0
    model, info = traits.fit(K, traits.take(t, ~test), M, seed=0, epochs=EPOCHS)
    r = traits.judge(model, traits.take(t, test), M, rng)
    if K:
        r["trait corr"] = abs(np.corrcoef(model.z_fit[:, 0].numpy(), t["z"][~test])[0, 1])
        print(f"      K={K} traits: {info['trait steps']} steps, |corr| with the true trait {r['trait corr']:.3f}, "
              f"sd {info['trait z sd'][0]:.2f}, skew {info['trait z skew'][0]:.2f}")
    print(f"      K={K} ({info['epochs']} epochs): {traits.summary_line(r)} -> {r['verdict']}")
    return r


def test_one_trait_needed():
    rng = np.random.default_rng(1)
    t, M = toy(rng)
    assert fit_judge(0, t, M, rng)["verdict"] == "FAIL"
    r = fit_judge(1, t, M, rng)
    assert r["trait corr"] >= 0.9
    assert r["verdict"] == "PASS"


def test_planes_shuffled_across_showers_need_no_trait():
    rng = np.random.default_rng(2)
    t, M = toy(rng, n_sh=800)
    for j in range(traits.S):                  # every (species, plane) slot to a random shower, independently
        for k in range(N_K):
            perm = rng.permutation(len(t["i"]))
            for key in ("O", "X", "base", "mask"):
                t[key][:, j, k] = t[key][perm, j, k]
    assert fit_judge(0, t, M, rng)["verdict"] == "PASS"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc(file=sys.stdout)
    sys.exit(1 if failed else 0)

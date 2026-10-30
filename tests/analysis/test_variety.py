"""CPU self-test of the variety candidates (symbolic/variety.py), on toy planes drawn through the real projection:
the true toy model passes both rules and the law alone fails the spread rule (the harness); each candidate fitted to
toys of its own form gives back its parameters and passes; a candidate without memory fails the memory rule on toys
that have it; planes shuffled across showers (no memory) show none."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import radial_bins, variety

RHO = 0.8
N_K = 24


def planes_and_law(rng, n_showers):
    M = radial_bins.ring_matrix()
    n_r = radial_bins.N_RING
    x = np.column_stack([np.ones(n_showers * N_K), rng.standard_normal((n_showers * N_K, 2))])
    B = np.column_stack([np.zeros(3), rng.normal(0, 0.2, (3, n_r - 1))])
    B[0, 1:] -= 0.4 * np.arange(1, n_r)                     # energy falls off outward, as in a real plane
    d = {"x": x, "i": np.repeat(np.arange(n_showers), N_K), "k": np.tile(np.arange(N_K), n_showers)}
    return d, B, M


def lowrank_truth(rng):
    return {"mu": np.zeros(variety.INNER.sum()), "L": rng.normal(0, 0.25, (variety.INNER.sum(), 2)),
            "rho": np.full(2, RHO)}


def toy(rng, v, draw, n_showers=500):
    d, B, M = planes_and_law(rng, n_showers)
    d["obs"] = draw(v, d, B, M, rng)
    return d, B, M


def split(d):
    return ({key: val[m] for key, val in d.items()} for m in (d["i"] % 5 != 0, d["i"] % 5 == 0))


def fit_and_judge(name, d, B, M, rng):
    fit, draw = variety.MODELS[name]
    train, test = split(d)
    eta, _ = variety.plane_offsets(train["x"], train["obs"], B, M)
    v = fit(train, B, M, eta, rng)
    return v, variety.judge(test, B, M, v, draw, rng)[0]


def test_harness_true_model_passes_and_law_alone_fails():
    rng = np.random.default_rng(2)
    v = lowrank_truth(rng)
    d, B, M = toy(rng, v, variety.draw_logit)
    _, test = split(d)
    r = variety.judge(test, B, M, v, variety.draw_logit, rng)[0]
    print(f"      true model: rule 1 {r['rule 1 spread']}, rule 2 {r['rule 2 memory']}, "
          f"law alone fails {r['sanity: law alone fails rule 1']}, pooled W1 {r['2.3 pooled W1 with variety / law alone / noise']}")
    assert r["rule 1 spread"] and r["rule 2 memory"] and r["sanity: law alone fails rule 1"], r


def test_lowrank_recovers_memory_and_passes():
    rng = np.random.default_rng(3)
    d, B, M = toy(rng, lowrank_truth(rng), variety.draw_logit)
    v, r = fit_and_judge("lowrank", d, B, M, rng)
    print(f"      K {len(v['rho'])}, rho {v['rho'].round(3).tolist()}, rule 1 {r['rule 1 spread']}, "
          f"rule 2 {r['rule 2 memory']}, KL CORSIKA/model {np.round(r['median per-plane KL to the law, CORSIKA / model'], 4)}")
    assert np.all(np.abs(v["rho"][:2] - RHO) < 0.05), v["rho"]
    assert r["verdict"] == "PASS", r


def test_tilt_recovers_scale_and_memory_and_passes():
    rng = np.random.default_rng(6)
    g = variety.tilt_direction()
    v0 = {"mu": np.zeros(len(g)), "L": (0.6 * g)[:, None], "rho": np.array([RHO])}
    d, B, M = toy(rng, v0, variety.draw_logit)
    v, r = fit_and_judge("tilt", d, B, M, rng)
    sd = float(np.linalg.norm(v["L"]))
    print(f"      tilt sd {sd:.3f} (true 0.6), rho {v['rho'].round(3).tolist()}, {r['verdict']}")
    assert abs(sd - 0.6) < 0.09 and abs(v["rho"][0] - RHO) < 0.05, (sd, v["rho"])
    assert r["verdict"] == "PASS", r


def test_dirichlet_recovers_concentration_and_passes():
    rng = np.random.default_rng(7)
    d, B, M = toy(rng, {"alpha": np.array(300.0)}, variety.draw_dirichlet)
    v, r = fit_and_judge("dirichlet", d, B, M, rng)
    print(f"      alpha {float(v['alpha']):.0f} (true 300), {r['verdict']}")
    assert abs(np.log(v["alpha"] / 300)) < 0.25, v["alpha"]
    assert r["verdict"] == "PASS", r


def test_candidate_without_memory_fails_memory_rule():
    rng = np.random.default_rng(8)
    d, B, M = toy(rng, lowrank_truth(rng), variety.draw_logit)
    _, r = fit_and_judge("dirichlet", d, B, M, rng)
    print(f"      dirichlet on a toy with memory: rule 2 {r['rule 2 memory']}")
    assert not r["rule 2 memory"], r


def test_planes_shuffled_across_showers_show_no_memory():
    rng = np.random.default_rng(4)
    d, B, M = toy(rng, lowrank_truth(rng), variety.draw_logit, n_showers=300)
    perm = rng.permutation(len(d["i"]))
    d = {key: (val[perm] if key in ("x", "obs") else val) for key, val in d.items()}
    eta, _ = variety.plane_offsets(d["x"], d["obs"], B, M)
    v = variety.fit_variety(eta, d["i"], d["k"])
    print(f"      shuffled planes: rho {v['rho'].round(3).tolist()}")
    assert np.all(np.abs(v["rho"][:2]) < 0.1), v["rho"]


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

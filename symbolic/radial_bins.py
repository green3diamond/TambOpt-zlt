"""Radial law of the energy in each plane without a named shape (study_plan.md 2.3, option B).

The density in a plane is round (2.2). Each plane is described by the shares of its energy in rings of in-plane
radius (flat density within a ring), inferred from the exact across-track profile: a ring's energy lands in the
across-track bins through a fixed matrix (round density, 10 m cell rounding at a random grid angle), and each plane's
ring shares are the non-negative mixture weights that best explain its across-track shares (EM updates). Laws: a
multinomial logit of the ring shares in annuli.features (with the longitudinal age), per species and class, one form
for every species.

Rule (fixed before the run): on held-out planes the law's pooled across-track profile inside radial.WINDOW is within
2 x split-half noise (energy-weighted Wasserstein-1 on log10(1 + |t1| / m), every plane weighted equally).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import annuli, fit_profiles, generator, planes, radial, render

FOLDER = "radial_bins"
OUTER = 1100.0                                       # m: rings and across-track bins used in the inversion
N_BIN = int(np.searchsorted(annuli.EDGES, OUTER))
BINS = annuli.EDGES[: N_BIN + 1]                     # across-track bins of the data (|t1|)
RINGS = np.r_[0.0, 1.0, 2.0, 5.0, BINS[1:]]          # rings of in-plane radius; finer inside the first cell, where
N_RING = len(RINGS) - 1                              # the e/gamma core sits (unresolved there, harmless for the fit)
EM_ITERS = 300


def ring_matrix(n_r=60, n_phi=180, n_cell=48, seed=0):
    """M[ring, bin]: share of a ring's energy (flat density in the ring) landing in each across-track bin."""
    rng = np.random.default_rng(seed)
    phi = (np.arange(n_phi) + 0.5) * np.pi / n_phi
    theta = rng.uniform(0, np.pi, n_cell)
    u = rng.uniform(-render.CELL / 2, render.CELL / 2, (n_cell, 2))
    eps = u[:, 0] * np.cos(theta) + u[:, 1] * np.sin(theta)
    M = np.zeros((N_RING, N_BIN))
    for i in range(N_RING):
        lo, hi = RINGS[i], RINGS[i + 1]
        r = np.sqrt(lo ** 2 + (hi ** 2 - lo ** 2) * (np.arange(n_r) + 0.5) / n_r)     # equal area steps
        a = np.abs(r[:, None, None] * np.cos(phi)[None, :, None] + eps[None, None, :])
        j = np.digitize(a, BINS) - 1
        M[i] = [(j == b).mean() for b in range(N_BIN)]
    return M


def ring_shares(obs, M, iters=EM_ITERS):
    """Per plane, ring shares q (simplex) maximising sum_b obs_b log (q M)_b, by EM updates."""
    keep = M.sum(1, keepdims=True)                   # a ring's energy can land beyond OUTER: renormalise
    Mn = M / keep
    q = np.full((len(obs), N_RING), 1.0 / N_RING)
    for _ in range(iters):
        pred = q @ Mn
        q = q * ((obs / np.maximum(pred, radial.EPS)) @ Mn.T)
        q /= q.sum(1, keepdims=True)
    return q


def across(q, M):
    """Across-track shares of ring shares q, normalised within OUTER."""
    p = q @ (M / M.sum(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def window(p):
    w = p[:, : radial.N_WIN]
    return w / np.maximum(w.sum(1, keepdims=True), radial.EPS)


def plane_table(root):
    """Per species: across-track shares inside OUTER, features (with the longitudinal age), class and shower."""
    tab = annuli.table(root, "across")
    fs = np.load(os.path.join(root, fit_profiles.SLICE))
    out = {}
    for s, t in tab.items():
        e = t["e_ann"][:, :N_BIN]
        ok = e[:, : radial.N_WIN].sum(1) > 0
        age_long = generator.age(fs["X"][t["i"], t["k"]], fs[f"{s}.xmax"][t["i"]])
        x = annuli.features(age_long, fs["meta.log10_E"][t["i"]], fs["meta.cos_zenith"][t["i"]],
                            np.log(t["e_ann"].sum(1)), t["r_ref"])
        out[s] = {"obs": (e / e.sum(1, keepdims=True))[ok], "x": x[ok], "label": t["label"][ok], "i": t["i"][ok]}
    return out


def fig_profiles(curves):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    lo = annuli.EDGES[: radial.N_WIN]
    for (s, cn), (obs, fit, law) in curves.items():
        ax = axes[list(planes.CLASS.values()).index(cn), SPECIES_NAMES.index(s)]
        for v, col, ls, name in ((obs, "black", "-", "CORSIKA"), (fit, "#2a78d6", "--", "rings, per plane"),
                                 (law, "#eb6834", "-", "rings, from the laws")):
            ax.step(lo + 1, v, where="post", color=col, ls=ls, lw=1.4, label=name)
        ax.set_xscale("log")
        ax.set_title(f"{s}s, {cn}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("1 + |across-track distance| [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of energy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def stage_run(out, args, log):
    M = ring_matrix()
    data = plane_table(args.out)
    rng = np.random.default_rng(args.seed)
    res = {"rings_m": RINGS.tolist(), "rule": __doc__.split("Rule")[1].strip(), "results": {}}
    curves, laws = {}, {}
    kl = lambda o, p: float(np.median(np.sum(o * (np.log(o + radial.EPS) - np.log(p + radial.EPS)), 1)))
    for s, d in data.items():
        q = ring_shares(d["obs"], M)
        for c, cn in planes.CLASS.items():
            m = d["label"] == c
            test = m & (d["i"] % 5 == 0)
            train = m & ~test
            B, opt = generator.fit_shares(d["x"][train], q[train])
            law = window(across(softmax(d["x"][test] @ B, axis=1), M))
            fit = window(across(q[test], M))
            obs = window(d["obs"][test])
            scale = radial.split_half(obs, d["i"][test], rng)
            w_law, w_fit = radial.w1(radial.pooled(law), radial.pooled(obs)), radial.w1(radial.pooled(fit),
                                                                                       radial.pooled(obs))
            r = {"planes": int(m.sum()), "held-out planes": int(test.sum()), "law converged": bool(opt.success),
                 "W1 law / per-plane rings / split-half noise": [w_law, w_fit, scale],
                 "median per-plane KL law / rings": [kl(obs, law), kl(obs, fit)],
                 "mean ring shares (held out) data / law": [q[test].mean(0).round(4).tolist(),
                                                            softmax(d["x"][test] @ B, axis=1).mean(0).round(4).tolist()],
                 "verdict": "PASS" if w_law <= 2 * scale else "FAIL"}
            res["results"][f"{s} {cn}"] = r
            curves[(s, cn)] = (radial.pooled(obs), radial.pooled(fit), radial.pooled(law))
            laws[f"{s}.{cn}.rings"] = B
            log.info("%-8s %-8s planes %6d | W1 law %.4f rings %.4f noise %.4f | KL law %.4f rings %.4f | %s", s, cn,
                     m.sum(), w_law, w_fit, scale, *r["median per-plane KL law / rings"], r["verdict"])
    np.savez(os.path.join(out, "ring_laws.npz"), rings=RINGS, **laws)
    logs.save_fig(fig_profiles(curves), out, "profiles", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()

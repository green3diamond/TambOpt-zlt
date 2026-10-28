"""Radial law of the energy in each plane, from the exact across-track profile (study_plan.md 2.3).

The density in a plane is round (2.2), so its across-track profile fixes it. Model: NKG f(r) = (r/r0)^(s-2)
(1 + r/r0)^(s-4.5) with scale r0 and age s, one form for every species. For each (r0, s) on a grid, the expected
share of the plane's energy in each |t1| bin inside WINDOW is computed from the round density, including the 10 m
cell rounding at a random grid angle (annuli `across` tables bin cell centres). Each training plane gets the grid
point of largest likelihood (shares as weights); laws for ln r0 and s are linear in annuli.features (with the
longitudinal age), per species and class; held-out planes are predicted from the laws.

Rule (fixed before the run): on held-out planes the law's pooled across-track profile is within 2 x split-half noise
(energy-weighted Wasserstein-1 on log10(1 + |t1| / m), every plane weighted equally); fitted (r0, s) off the grid
edges in >= 90% of planes; the lateral age rises with the longitudinal age (reported either way).

--tied (cascade theory): the lateral age is fixed to the plane's longitudinal age and only r0 is fitted. Rule (fixed
before that run): median per-plane KL at most 2 x the free fit's; r0 off its grid edges in >= 90% of planes; the r0
law's pooled held-out profile within 2 x split-half noise.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import annuli, fit_profiles, generator, planes, render

FOLDER = "radial"
WINDOW = 280.0                                  # m, inside the round region and away from the cut (2.2)
N_WIN = int(np.searchsorted(annuli.EDGES, WINDOW))
LN_R0 = np.linspace(np.log(0.5), np.log(2000.0), 120)
AGE = np.arange(0.1, 2.401, 0.025)
R_GRID = np.logspace(-3, 5, 480)                # m, radial integration
SPLITS = 5
CHUNK = 5000
EPS = 1e-12


def bin_probability(n_phi=180, n_cell=48, seed=0):
    """T[r, j]: chance that energy at radius r lands in across-track bin j, cell rounding included."""
    rng = np.random.default_rng(seed)
    phi = (np.arange(n_phi) + 0.5) * np.pi / n_phi
    theta = rng.uniform(0, np.pi, n_cell)
    u = rng.uniform(-render.CELL / 2, render.CELL / 2, (n_cell, 2))
    eps = u[:, 0] * np.cos(theta) + u[:, 1] * np.sin(theta)        # cell offset projected on the across axis
    a = np.abs(R_GRID[:, None, None] * np.cos(phi)[None, :, None] + eps[None, None, :])
    j = np.digitize(a, annuli.EDGES[: N_WIN + 1]) - 1
    return np.stack([(j == b).mean((1, 2)) for b in range(N_WIN)], 1)


def nkg_weights(ln_r0, s):
    """Energy per radial grid step (r^2 f(r) d ln r), shapes (len(ln_r0), len(s), len(R_GRID))."""
    x = R_GRID[None, None, :] / np.exp(ln_r0)[:, None, None]
    s = s[None, :, None]
    with np.errstate(over="ignore", under="ignore"):
        lw = (s - 2) * np.log(x) + (s - 4.5) * np.log1p(x) + 2 * np.log(R_GRID)[None, None, :]
    return np.exp(lw - lw.max(2, keepdims=True))


def share_table():
    """G[i_r0, i_s, j]: expected share of the windowed energy in across-track bin j."""
    G = nkg_weights(LN_R0, AGE) @ bin_probability()
    return G / G.sum(2, keepdims=True)


def fit_planes(obs, logG):
    """Grid index of the largest likelihood for each plane's observed shares (n, N_WIN)."""
    flat = logG.reshape(-1, N_WIN).T
    return np.concatenate([np.argmax(obs[i:i + CHUNK] @ flat, 1) for i in range(0, len(obs), CHUNK)])


def fit_scale(obs, logG, age_idx):
    """Grid index of the best r0 for each plane at its own (fixed) age index."""
    out = []
    for i in range(0, len(obs), CHUNK):
        L = logG[:, age_idx[i:i + CHUNK], :]                       # (n_r0, n, bins)
        out.append(np.argmax(np.einsum("rnb,nb->nr", L, obs[i:i + CHUNK]), 1))
    return np.concatenate(out)


def shares_at(G, ln_r0, s):
    """Shares at arbitrary (ln r0, s), nearest grid point."""
    i = np.clip(np.rint((ln_r0 - LN_R0[0]) / (LN_R0[1] - LN_R0[0])).astype(int), 0, len(LN_R0) - 1)
    j = np.clip(np.rint((s - AGE[0]) / (AGE[1] - AGE[0])).astype(int), 0, len(AGE) - 1)
    return G[i, j]


def mid_log():
    lo, hi = annuli.EDGES[:N_WIN], annuli.EDGES[1:N_WIN + 1]
    return np.log10(1 + 0.5 * (lo + hi))


def w1(p, q):
    """W1 between two binned across-track profiles on log10(1 + |t1|)."""
    return float(stats.wasserstein_distance(mid_log(), mid_log(), p + EPS, q + EPS))


def pooled(sh):
    return sh.mean(0)


def split_half(obs, unit, rng):
    vals = []
    for _ in range(SPLITS):
        half = rng.permutation(np.unique(unit))[: len(np.unique(unit)) // 2]
        m = np.isin(unit, half)
        vals.append(w1(pooled(obs[m]), pooled(obs[~m])))
    return float(np.mean(vals) / np.sqrt(2))


def plane_table(root):
    """Per species: observed windowed shares, features (with the longitudinal age), class, shower and plane."""
    tab = annuli.table(root, "across")
    fs = np.load(os.path.join(root, fit_profiles.SLICE))
    out = {}
    for s, t in tab.items():
        e = t["e_ann"][:, :N_WIN]
        ok = e.sum(1) > 0
        age_long = generator.age(fs["X"][t["i"], t["k"]], fs[f"{s}.xmax"][t["i"]])
        x = annuli.features(age_long, fs["meta.log10_E"][t["i"]], fs["meta.cos_zenith"][t["i"]],
                            np.log(t["e_ann"].sum(1)), t["r_ref"])
        out[s] = {"obs": (e / np.maximum(e.sum(1, keepdims=True), EPS))[ok], "x": x[ok], "age": age_long[ok],
                  "label": t["label"][ok], "i": t["i"][ok], "k": t["k"][ok], "r_ref": t["r_ref"][ok]}
    return out


def fig_profiles(curves):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    lo = annuli.EDGES[:N_WIN]
    for (s, cn), (obs, fit, law) in curves.items():
        ax = axes[list(planes.CLASS.values()).index(cn), SPECIES_NAMES.index(s)]
        for v, col, ls, name in ((obs, "black", "-", "CORSIKA"), (fit, "#2a78d6", "--", "NKG fit per plane"),
                                 (law, "#eb6834", "-", "NKG from the laws")):
            ax.step(lo + 1, v, where="post", color=col, ls=ls, lw=1.4, label=name)
        ax.set_xscale("log")
        ax.set_title(f"{s}s, {cn}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("1 + |across-track distance| [m]")
    axes[0, 0].set_ylabel("share of energy")
    axes[1, 0].set_ylabel("share of energy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_age(ages):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    bins = np.linspace(0.3, 2.0, 18)
    for ax, cn in zip(axes, planes.CLASS.values()):
        for s, col in zip(SPECIES_NAMES, ("#2a78d6", "#8e6fd1", "#eb6834")):
            a_long, a_lat = ages[(s, cn)]
            j = np.digitize(a_long, bins) - 1
            ok = [b for b in range(len(bins) - 1) if (j == b).sum() >= 30]
            ax.plot([(bins[b] + bins[b + 1]) / 2 for b in ok], [np.median(a_lat[j == b]) for b in ok], "o-",
                    color=col, ms=3, label=f"{s}s")
        ax.set_title(f"{cn} showers")
        ax.set_xlabel("longitudinal age")
    axes[0].set_ylabel("lateral age s (median)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def stage_run(out, args, log):
    G = share_table()
    logG = np.log(G + EPS)
    data = plane_table(args.out)
    rng = np.random.default_rng(args.seed)
    res = {"window_m": WINDOW, "grid": {"r0_m": [float(np.exp(LN_R0[0])), float(np.exp(LN_R0[-1]))],
                                        "age": [float(AGE[0]), float(AGE[-1])]},
           "rule": __doc__.split("Rule")[1].strip(), "results": {}}
    curves, ages, laws = {}, {}, {}
    for s, d in data.items():
        ir_free, js_free = np.unravel_index(fit_planes(d["obs"], logG), G.shape[:2])
        if args.tied:
            js = np.clip(np.rint((d["age"] - AGE[0]) / (AGE[1] - AGE[0])).astype(int), 0, len(AGE) - 1)
            ir = fit_scale(d["obs"], logG, js)
            edge = (ir == 0) | (ir == len(LN_R0) - 1)
        else:
            ir, js = ir_free, js_free
            edge = (ir == 0) | (ir == len(LN_R0) - 1) | (js == 0) | (js == len(AGE) - 1)
        ln_r0, age = LN_R0[ir], AGE[js]
        for c, cn in planes.CLASS.items():
            m = d["label"] == c
            test = m & (d["i"] % 5 == 0)
            train = m & ~test & ~edge
            A = np.linalg.lstsq(d["x"][train], ln_r0[train], rcond=None)[0]
            B = np.linalg.lstsq(d["x"][train], age[train], rcond=None)[0]
            law = shares_at(G, d["x"][test] @ A, age[test] if args.tied else d["x"][test] @ B)
            fit = G[ir[test], js[test]]
            obs = d["obs"][test]
            scale = split_half(obs, d["i"][test], rng)
            w_law, w_fit = w1(pooled(law), pooled(obs)), w1(pooled(fit), pooled(obs))
            kl = lambda p: float(np.median(np.sum(obs * (np.log(obs + EPS) - np.log(p + EPS)), 1)))
            rho = float(stats.spearmanr(d["age"][m & ~edge], age[m & ~edge])[0])
            kl_free = kl(G[ir_free[test], js_free[test]])
            r = {"planes": int(m.sum()), "held-out planes": int(test.sum()), "median per-plane KL free fit": kl_free,
                 "off grid edges": float(1 - edge[m].mean()),
                 "W1 law / per-plane fit / split-half noise": [w_law, w_fit, scale],
                 "median per-plane KL law / fit": [kl(law), kl(fit)],
                 "lateral vs longitudinal age, Spearman": rho,
                 "median r0 / r_ref": float(np.median(np.exp(ln_r0[m]) / d["r_ref"][m])),
                 "median s": float(np.median(age[m]))}
            r["verdict"] = "PASS" if (w_law <= 2 * scale and r["off grid edges"] >= 0.9
                                      and (not args.tied or kl(fit) <= 2 * kl_free)) else "FAIL"
            res["results"][f"{s} {cn}"] = r
            curves[(s, cn)] = (pooled(obs), pooled(fit), pooled(law))
            ages[(s, cn)] = (d["age"][m & ~edge], age[m & ~edge])
            laws[f"{s}.{cn}.ln_r0"], laws[f"{s}.{cn}.age"] = A, B
            log.info("%-8s %-8s planes %6d | W1 law %.4f fit %.4f noise %.4f | off edges %.3f | KL law %.4f fit "
                     "%.4f (free %.4f) | age rho %.2f | r0/r_ref %.2f s %.2f | %s", s, cn, m.sum(), w_law, w_fit,
                     scale, r["off grid edges"], *r["median per-plane KL law / fit"], kl_free, rho,
                     r["median r0 / r_ref"], r["median s"], r["verdict"])
    np.savez(os.path.join(out, "radial_laws.npz"), ln_r0_grid=LN_R0, age_grid=AGE, **laws)
    logs.save_fig(fig_profiles(curves), out, "profiles", log)
    logs.save_fig(fig_age(ages), out, "lateral_vs_longitudinal_age", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tied", action="store_true", help="lateral age fixed to the longitudinal age; fit r0 only")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "tied" if args.tied else "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()

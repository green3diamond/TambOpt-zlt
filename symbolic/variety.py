"""Shower-to-shower variety of the in-plane shape (study_plan.md Part 2, between 2.3 and 2.4).

The ring law (radial_bins) gives a plane's mean ring shares. Three candidate forms of how a plane departs from it,
judged by the same rules, one form for every species and class:
- lowrank: a random logit offset on the rings inside radial.WINDOW, shares = softmax(x B + eta), eta = mu + L z, L of
  rank K (principal components keeping VAR_KEPT of the variance), z standard normal per plane and a stationary AR(1)
  along the planes of a shower (one coefficient rho per component);
- tilt: the same with K = 1 and a fixed direction, the logit tilt in log ring radius (a plane more or less
  concentrated than the law), its scale and AR(1) coefficient fitted;
- dirichlet: ring shares Dirichlet about the law's shares with one concentration, independent planes, fitted so that
  the median per-plane KL of the projected shares to the law matches the training planes'.
lowrank and tilt start from each training plane's offset eta* that best explains its across-track shares through the
projection (a tiny ridge leaves no offset where the data cannot see one, e.g. inside the first 10 m cell); mu = mean
eta*; rho = lag-1 correlation of the component scores.
Only planes with at least MIN_CELLS cells inside radial_bins.OUTER enter, fit and judgement, every species: sparse
planes are dominated by particle-count noise, which 2.4 produces by drawing particles.

Rule (fixed before the run), on held-out showers per species and class, model draws through the same projection:
1. spread: W1 between the model and CORSIKA distributions of each plane's across-track share inside 10, 30, 100 and
   280 m within 2 x split-half noise, all four;
2. memory: the correlation of the 30 m share's offset from the law between planes k and k+1, and k and k+3, within
   2 se of CORSIKA's (bootstrap by shower, both sides).
PASS if both hold. Sanity: the law without variety must fail rule 1. Reported only: cross-species correlation of the
30 m offsets, median per-plane KL to the law, the 2.3 pooled-profile rule with the variety added.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import softmax

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import annuli, planes, radial, radial_bins

FOLDER = "variety"
MIN_CELLS = 50
MAX_SHOWERS = 5000                                   # per class, training and held out
VAR_KEPT = 0.95
RIDGE = 1e-5
DRAWS = 4
BOOT = 200
CHUNK = 20000
CUTS_M = (10.0, 30.0, 100.0, 280.0)
CUTS = np.searchsorted(annuli.EDGES, CUTS_M)         # across-track bins inside each cut
INNER = radial_bins.RINGS[1:] <= radial.WINDOW       # rings that carry the random offset
LAGS = (1, 3)


def plane_offsets(x, obs, B, M, ridge=RIDGE):
    """Per plane, the logit offset on the inner rings that best explains its across-track shares."""
    Mn = M / M.sum(1, keepdims=True)
    base = x @ B
    out = np.zeros((len(x), INNER.sum()))
    ok = True
    for a in range(0, len(x), CHUNK):
        b0, o = base[a:a + CHUNK], obs[a:a + CHUNK]
        n = len(o)

        def f(v):
            e = v.reshape(n, -1)
            eta = b0.copy()
            eta[:, INNER] += e
            q = softmax(eta, axis=1)
            p = np.maximum(q @ Mn, radial.EPS)
            g = -(o / p) @ Mn.T
            geta = q * (g - (q * g).sum(1, keepdims=True))
            return -np.sum(o * np.log(p)) + ridge / 2 * np.sum(e ** 2), (geta[:, INNER] + ridge * e).ravel()

        res = minimize(f, np.zeros(n * INNER.sum()), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        out[a:a + n] = res.x.reshape(n, -1)
        ok &= bool(res.success)
    return out, ok


def lag_pairs(i, k, lag):
    """Indices (a, b) of plane pairs of the same shower with k_b = k_a + lag."""
    key = i.astype(np.int64) * 64 + k
    order = np.argsort(key)
    pos = np.minimum(np.searchsorted(key[order], key + lag), len(key) - 1)
    hit = key[order][pos] == key + lag
    return np.nonzero(hit)[0], order[pos[hit]]


def fit_variety(eta, i, k):
    """mu, loadings L (principal components keeping VAR_KEPT) and the lag-1 AR coefficient of each component."""
    mu = eta.mean(0)
    _, s, vt = np.linalg.svd(eta - mu, full_matrices=False)
    share = np.cumsum(s ** 2) / np.sum(s ** 2)
    K = int(np.searchsorted(share, VAR_KEPT) + 1)
    sd = s[:K] / np.sqrt(len(eta))
    z = (eta - mu) @ vt[:K].T / sd
    a, b = lag_pairs(i, k, 1)
    rho = np.array([np.corrcoef(z[a, j], z[b, j])[0, 1] for j in range(K)])
    return {"mu": mu, "L": vt[:K].T * sd, "rho": np.clip(rho, -0.99, 0.99), "var_share": share[:K]}


def draw_offsets(v, i, k, rng):
    """Offsets for every plane (i, k): standard normal z, stationary AR(1) along the planes of each shower."""
    sh, inv = np.unique(i, return_inverse=True)
    K = len(v["rho"])
    z = np.empty((len(sh), int(k.max()) + 1, K))
    z[:, 0] = rng.standard_normal((len(sh), K))
    for t in range(1, z.shape[1]):
        z[:, t] = v["rho"] * z[:, t - 1] + np.sqrt(1 - v["rho"] ** 2) * rng.standard_normal((len(sh), K))
    return v["mu"] + z[inv.ravel(), k] @ v["L"].T


def shares(x, B, M, eta=None):
    """Across-track shares inside OUTER of the law, plus the inner-ring offsets eta if given."""
    logit = x @ B
    if eta is not None:
        logit = logit.copy()
        logit[:, INNER] += eta
    return radial_bins.across(softmax(logit, axis=1), M)


def cut_shares(p):
    return np.cumsum(p, 1)[:, CUTS - 1]


def spread(s_obs, s_mod, i, rng):
    """Per cut: W1 of the model against CORSIKA, and CORSIKA split-half noise (halves by shower, / sqrt 2)."""
    sh = np.unique(i)
    out = []
    for c in range(s_obs.shape[1]):
        noise = []
        for _ in range(radial.SPLITS):
            m = np.isin(i, rng.permutation(sh)[: len(sh) // 2])
            noise.append(stats.wasserstein_distance(s_obs[m, c], s_obs[~m, c]))
        out.append((float(stats.wasserstein_distance(s_mod[:, c], s_obs[:, c])), float(np.mean(noise) / np.sqrt(2))))
    return out


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w)
                                                                 * np.average((y - my) ** 2, weights=w))


def memory(u, i, k, lag, rng):
    """Correlation of u between planes k and k + lag of one shower, and its bootstrap se (resampling showers)."""
    a, b = lag_pairs(i, k, lag)
    sh, inv = np.unique(i[a], return_inverse=True)
    boots = [wcorr(u[a], u[b], rng.multinomial(len(sh), np.full(len(sh), 1 / len(sh)))[inv]) for _ in range(BOOT)]
    return float(np.corrcoef(u[a], u[b])[0, 1]), float(np.std(boots))


def median_kl(p, q):
    return float(np.median(np.sum(p * (np.log(p + radial.EPS) - np.log(q + radial.EPS)), 1)))


def tilt_direction():
    mid = (radial_bins.RINGS[:-1] + radial_bins.RINGS[1:])[INNER] / 2
    g = np.log(mid) - np.log(mid).mean()
    return g / np.linalg.norm(g)


def fit_lowrank(d, B, M, eta, rng):
    return fit_variety(eta, d["i"], d["k"])


def fit_tilt(d, B, M, eta, rng):
    """lowrank with K = 1 and the direction fixed to the tilt in log ring radius."""
    mu, g = eta.mean(0), tilt_direction()
    a = (eta - mu) @ g
    sd = a.std()
    i, j = lag_pairs(d["i"], d["k"], 1)
    rho = np.corrcoef(a[i], a[j])[0, 1]
    return {"mu": mu, "L": (sd * g)[:, None], "rho": np.clip([rho], -0.99, 0.99),
            "var_share": np.array([np.sum(a ** 2) / np.sum((eta - mu) ** 2)])}


def draw_logit(v, d, B, M, rng):
    return shares(d["x"], B, M, draw_offsets(v, d["i"], d["k"], rng))


def draw_dirichlet(v, d, B, M, rng):
    q0 = softmax(d["x"] @ B, axis=1)
    q = rng.gamma(np.maximum(v["alpha"] * q0, 1e-6))
    q = np.maximum(q, 1e-300)
    return radial_bins.across(q / q.sum(1, keepdims=True), M)


def fit_dirichlet(d, B, M, eta, rng):
    """Concentration such that the median per-plane KL of projected draws to the law matches the planes' (bisection
    in log alpha, the same seed at every step)."""
    m = rng.choice(len(d["i"]), min(len(d["i"]), CHUNK), replace=False)
    sub = {key: val[m] for key, val in d.items()}
    p0 = shares(sub["x"], B, M)
    target = median_kl(sub["obs"], p0)
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        kl = median_kl(draw_dirichlet({"alpha": np.exp(mid)}, sub, B, M, np.random.default_rng(0)), p0)
        lo, hi = (mid, hi) if kl > target else (lo, mid)
    return {"alpha": np.array(np.exp((lo + hi) / 2)), "target KL": np.array(target)}


MODELS = {"lowrank": (fit_lowrank, draw_logit), "tilt": (fit_tilt, draw_logit),
          "dirichlet": (fit_dirichlet, draw_dirichlet)}


def judge(d, B, M, v, draw, rng):
    """Rules 1 and 2 and the reported checks on held-out planes d, for the fitted variety v of one candidate."""
    p0 = shares(d["x"], B, M)
    pm = np.concatenate([draw(v, d, B, M, rng) for _ in range(DRAWS)])
    i_m = np.concatenate([d["i"] * DRAWS + j for j in range(DRAWS)])
    k_m = np.tile(d["k"], DRAWS)
    s_obs, s_law, s_mod = cut_shares(d["obs"]), cut_shares(p0), cut_shares(pm)
    sp, sp0 = spread(s_obs, s_mod, d["i"], rng), spread(s_obs, s_law, d["i"], rng)
    u_obs, u_mod = s_obs[:, 1] - s_law[:, 1], s_mod[:, 1] - np.tile(s_law[:, 1], DRAWS)
    mem = {lag: (memory(u_obs, d["i"], d["k"], lag, rng), memory(u_mod, i_m, k_m, lag, rng)) for lag in LAGS}
    win = lambda p: radial.pooled(radial_bins.window(p))
    r = {"spread W1 model / noise per cut": {int(c): sp[j] for j, c in enumerate(CUTS_M)},
         "spread W1 law alone / noise per cut": {int(c): sp0[j] for j, c in enumerate(CUTS_M)},
         "memory corr CORSIKA (se) / model (se)": {lag: m for lag, m in mem.items()},
         "median per-plane KL to the law, CORSIKA / model": [median_kl(d["obs"], p0), median_kl(pm[: len(p0)], p0)],
         "2.3 pooled W1 with variety / law alone / noise": [radial.w1(win(pm), win(d["obs"])),
                                                           radial.w1(win(p0), win(d["obs"])),
                                                           radial.split_half(radial_bins.window(d["obs"]), d["i"], rng)]}
    r["rule 1 spread"] = all(w <= 2 * n for w, n in sp)
    r["rule 2 memory"] = all(abs(mo[0] - mc[0]) <= 2 * np.hypot(mc[1], mo[1]) for mc, mo in mem.values())
    r["sanity: law alone fails rule 1"] = not all(w <= 2 * n for w, n in sp0)
    r["verdict"] = "PASS" if r["rule 1 spread"] and r["rule 2 memory"] else "FAIL"
    return r, (s_obs[:, 1], s_mod[:, 1], s_law[:, 1]), u_obs, u_mod[: len(u_obs)]


def subsample(d, mask, rng):
    sh = np.unique(d["i"][mask])
    if len(sh) > MAX_SHOWERS:
        sh = rng.choice(sh, MAX_SHOWERS, replace=False)
    m = mask & np.isin(d["i"], sh)
    return {key: val[m] for key, val in d.items()}


COLORS = {"lowrank": "#2a78d6", "tilt": "#1baf7a", "dirichlet": "#9b59b6"}


def fig_shares(curves):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    bins = np.linspace(0, 1, 41)
    for (s, cn), (obs, law, mods) in curves.items():
        ax = axes[list(planes.CLASS.values()).index(cn), SPECIES_NAMES.index(s)]
        ax.hist(obs, bins, density=True, histtype="step", color="black", lw=1.6, label="CORSIKA")
        ax.hist(law, bins, density=True, histtype="step", color="#eb6834", ls="--", lw=1.2, label="law alone")
        for name, mod in mods.items():
            ax.hist(mod, bins, density=True, histtype="step", color=COLORS[name], lw=1.2, label=name)
        ax.set_title(f"{s}s, {cn}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("share of energy within 30 m across track")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def cross_species(offsets):
    """Correlation of the 30 m offsets of two species on the same held-out plane, CORSIKA and each candidate."""
    out = {}
    for cn in planes.CLASS.values():
        sp = [s for s in SPECIES_NAMES if (s, cn) in offsets]
        for a in range(len(sp)):
            for b in range(a + 1, len(sp)):
                (ka, oa, ma), (kb, ob, mb) = offsets[(sp[a], cn)], offsets[(sp[b], cn)]
                _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
                if len(ia) > 10:
                    out[f"{sp[a]}-{sp[b]} {cn}"] = {"planes": len(ia),
                                                    "CORSIKA": float(np.corrcoef(oa[ia], ob[ib])[0, 1]),
                                                    **{n: float(np.corrcoef(ma[n][ia], mb[n][ib])[0, 1]) for n in ma}}
    return out


def stage_run(out, args, log):
    M = radial_bins.ring_matrix()
    data = radial_bins.plane_table(args.out)
    rng = np.random.default_rng(args.seed)
    res = {"candidates": list(MODELS), "min cells": MIN_CELLS, "var kept": VAR_KEPT, "ridge": RIDGE, "draws": DRAWS,
           "rule": __doc__.split("Rule")[1].strip(), "results": {}}
    curves, offsets, fitted = {}, {}, {}
    for s, d in data.items():
        if args.species and s != args.species:
            continue
        laws = np.load(os.path.join(args.out, radial_bins.FOLDER, f"conv_{s}", "ring_laws.npz"))
        if not np.allclose(laws["rings"], radial_bins.RINGS):
            raise ValueError(f"ring grid of conv_{s} differs from radial_bins.RINGS")
        for c, cn in planes.CLASS.items():
            dense = (d["label"] == c) & (d["n"] >= MIN_CELLS)
            if not dense.any():
                log.info("%-8s %-8s no planes with >= %d cells", s, cn, MIN_CELLS)
                continue
            B = laws[f"{s}.{cn}.rings"]
            train = subsample(d, dense & (d["i"] % 5 != 0), rng)
            test = subsample(d, dense & (d["i"] % 5 == 0), rng)
            eta, conv = plane_offsets(train["x"], train["obs"], B, M)
            g = {"dense planes (all / train / held out)": [int(dense.sum()), len(train["i"]), len(test["i"])],
                 "held-out showers": len(np.unique(test["i"])), "offsets converged": conv,
                 "max |offset|": float(np.abs(eta).max())}
            mods, u_mods = {}, {}
            for name, (fit, draw) in MODELS.items():
                v = fit(train, B, M, eta, rng)
                r, (s_obs, s_mod, s_law), u_obs, u_mods[name] = judge(test, B, M, v, draw, rng)
                mods[name] = s_mod
                r = {**{key: np.atleast_1d(val).round(3).tolist() for key, val in v.items()
                        if key in ("rho", "var_share", "alpha", "target KL")}, **r}
                g[name] = r
                fitted.update({f"{s}.{cn}.{name}.{key}": val for key, val in v.items()})
                log.info("%-8s %-8s %-9s planes %6d/%6d %s | spread %s (law alone fails: %s) | memory %s | %s", s, cn,
                         name, len(train["i"]), len(test["i"]),
                         {key: val for key, val in r.items() if key in ("rho", "alpha")},
                         {cut: f"{w:.4f}/{n:.4f}" for cut, (w, n) in r["spread W1 model / noise per cut"].items()},
                         r["sanity: law alone fails rule 1"],
                         {lag: f"{mc[0]:.2f}+-{mc[1]:.2f} vs {mo[0]:.2f}+-{mo[1]:.2f}"
                          for lag, (mc, mo) in r["memory corr CORSIKA (se) / model (se)"].items()}, r["verdict"])
            curves[(s, cn)] = (s_obs, s_law, mods)
            offsets[(s, cn)] = (test["i"].astype(np.int64) * 64 + test["k"], u_obs, u_mods)
            res["results"][f"{s} {cn}"] = g
            logs.save(out, res, args, log)
    res["cross-species correlation of 30 m offsets"] = cross_species(offsets)
    log.info("cross-species %s", res["cross-species correlation of 30 m offsets"])
    np.savez(os.path.join(out, "variety.npz"), rings=radial_bins.RINGS, inner=INNER, **fitted)
    logs.save_fig(fig_shares(curves), out, "shares_30m", log)
    logs.save(out, res, args, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--species", choices=SPECIES_NAMES, help="one species only")
    ap.add_argument("--name", default="run", help="output folder under variety/")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    stage_run(out, args, log)


if __name__ == "__main__":
    main()

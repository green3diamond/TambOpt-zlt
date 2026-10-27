"""Symbolic shower generator: per-plane energies from the phase-1 sampler, cells within each plane from annulus laws.

In-plane laws, per species and class, fitted to repaired CORSIKA (annuli.py on M1's training slice), all on the stored
picture (cap and radius cut included, which is what the detector is compared against). x = annuli.features(age,
log10 E, cos z, ln E_k, r_M,ref) with age = 3 X / (X + 2 Xmax) clipped to AGE_RANGE.
  shares   the stored cell count per annulus of in-plane radius (annuli.EDGES) as a multinomial-logit law,
           log share_j = x . B_j, fitted to every plane's shares (each plane weight 1).
  cell     ln(energy per cell in annulus j / the plane's energy per cell) = x . C_j, least squares weighted by the
           cell count. Energy and count are tied per annulus: separately fitted energy and count shares disagree
           annulus by annulus and make bright cells that over-light the detector.
  pattern  the cell law's residuals across annuli as one pattern times one number per plane, AR(1) along a shower's
           planes (planes vary in how concentrated they are around the laws). Weighted rank-1 least squares, weights
           the cell counts, planes with >= PATTERN_MIN_ANN occupied annuli; the fit has several optima, so it starts
           from PATTERN_STARTS points and keeps the best training fit.
  count    ln(cells on the plane) = x . c, plus Gaussian noise (held-out residual sd).
  delay    median cell delay after the light travel time, per annulus (median over the training planes).
A draw for a primary (log10 E, direction, class): the sampler's E_k and Xmax per species; per plane with E_k > 0 the
cell count, its split over the annuli (multinomial on the count shares), cells uniform in area within their annulus
at a uniform angle in the plane, each cell's energy from the cell law and the plane's pattern number, all scaled to
sum to E_k, time = light travel time + the annulus delay. Cells in the native cloud format [x, y, plane, energy, time]
(ground offsets from the injection point).

Stages:
  fit    the laws, with held-out checks (share per annulus observed / law by |cos z| band, cell-law residuals,
         pattern variance removed) -> generator/MODEL
  draw   one cloud per primary of the scored test set (ROWS) -> generator/clouds[_oracle].npz; --oracle: CORSIKA's own
         plane energies and Xmax instead of the sampler's (isolates the in-plane laws)
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from analysis_core import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import annuli, atmosphere, latent, lateral, m1, m2, sampler, score

FOLDER = "generator"
ROWS = os.path.join("chain", "clouds_draw00.npz")   # the scored test primaries (their "rows"), shared with the chain
AGE_RANGE = (0.2, 2.5)
R_OPEN = 1.4                 # the open last annulus is drawn up to R_OPEN x its inner edge
MIN_CELL_PLANES = 50         # planes with cells in an annulus needed to fit its cell-energy law
PATTERN_MIN_ANN = 4          # occupied annuli a plane needs to enter the pattern fit
PATTERN_ITERS = 300
PATTERN_STARTS = 8           # the leading 3 singular vectors, a constant pattern, 4 random ones


def age(X, xmax):
    """Longitudinal age clipped to AGE_RANGE; 1 (the maximum) where Xmax is unknown (an unfit CORSIKA profile)."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(xmax), np.clip(3 * X / (X + 2 * xmax), *AGE_RANGE), 1.0)


def fit_shares(x, w):
    """Multinomial-logit coefficients (n_feat, n_bin) for plane shares w (rows sum to 1); the first bin is the base."""
    n_f, n_b = x.shape[1], w.shape[1]

    def f(b):
        B = np.column_stack([np.zeros(n_f), b.reshape(n_f, n_b - 1)])
        eta = x @ B
        lp = eta - logsumexp(eta, axis=1, keepdims=True)
        g = x.T @ (np.exp(lp) - w)
        return -np.sum(w * lp) / len(x), g[:, 1:].ravel() / len(x)

    res = minimize(f, np.zeros(n_f * (n_b - 1)), jac=True, method="L-BFGS-B", options={"maxiter": 10000})
    return np.column_stack([np.zeros(n_f), res.x.reshape(n_f, n_b - 1)]), res


def cell_target(e_ann, n_ann):
    """ln(energy per cell in each annulus / the plane's energy per cell); nan where an annulus has no cells."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e_ann / n_ann) - np.log(e_ann.sum(1) / n_ann.sum(1))[:, None]


def fit_cell_energy(x, e_ann, n_ann, train, test):
    """Per annulus j, cell_target = x . C_j by least squares weighted by the cell count, on the planes with cells in j;
    an annulus with fewer than MIN_CELL_PLANES planes takes the coefficients of the annulus inside it. Returns C
    (n_feat, n_bin) and the held-out residual [mean, sd] per annulus."""
    y = cell_target(e_ann, n_ann)
    C, held = np.zeros((x.shape[1], annuli.N_BIN)), []
    for j in range(annuli.N_BIN):
        m = train & (n_ann[:, j] > 0)
        if m.sum() >= MIN_CELL_PLANES:
            w = np.sqrt(n_ann[m, j])
            C[:, j] = np.linalg.lstsq(x[m] * w[:, None], y[m, j] * w, rcond=None)[0]
        elif j:
            C[:, j] = C[:, j - 1]
        mt = test & (n_ann[:, j] > 0)
        res = y[mt, j] - x[mt] @ C[:, j]
        held.append([round(float(res.mean()), 3), round(float(res.std()), 3)] if mt.sum() > 1 else None)
    return C, held


def pattern_scores(r, w, g):
    """Per row, the weighted least-squares number a of r ~ a g."""
    return (w * r) @ g / np.maximum(w @ g ** 2, 1e-12)


def removed(r, w, g):
    """Weighted share of the residual variance the pattern removes."""
    res = r - pattern_scores(r, w, g)[:, None] * g
    return float(1 - (w * res ** 2).sum() / (w * r ** 2).sum())


def lag1(a, unit, plane):
    """Correlation of a between planes k and k + 1 of a unit (a shower)."""
    order = np.lexsort((plane, unit))
    pair = (np.diff(unit[order]) == 0) & (np.diff(plane[order]) == 1)
    return float(np.corrcoef(a[order][:-1][pair], a[order][1:][pair])[0, 1])


def fit_pattern(r, w, rng):
    """Weighted rank-1 least squares r ~ a g by alternating least squares from PATTERN_STARTS starts (the leading
    singular vectors, a constant, random), keeping the best training fit; g scaled so that a has unit variance."""
    starts = list(np.linalg.svd(r[:: max(1, len(r) // 20000)], full_matrices=False)[2][:3])
    starts += [np.ones(r.shape[1])] + list(rng.normal(size=(PATTERN_STARTS - len(starts) - 1, r.shape[1])))
    best = None
    for g in starts:
        for _ in range(PATTERN_ITERS):
            a = pattern_scores(r, w, g)
            g = (w * r).T @ a / np.maximum(w.T @ a ** 2, 1e-12)
        if best is None or removed(r, w, g) > removed(r, w, best):
            best = g
    return best * pattern_scores(r, w, best).std()


def stage_fit(out, args, log):
    tab = annuli.law_table(os.path.join(args.out, annuli.FOLDER), args)
    m1f = np.load(os.path.join(args.out, m1.FOLDER, "fit", "fits_train.npz"))
    n_rows = len(m1f["meta.rows"])
    rng = np.random.default_rng(args.seed)
    model, res = {}, {s: {} for s in tab}
    for s, t in tab.items():
        x = annuli.features(age(m1f["X"][t["i"], t["k"]], m1f[f"{s}.{m2.NAME}.xmax"][t["i"]]),
                            m1f["meta.log10_E"][t["i"]], m1f["meta.cos_zenith"][t["i"]],
                            np.log(t["e_ann"].sum(1)), t["r_ref"])
        for c, cn in m1.CLASS.items():
            m = (t["label"] == c) & (((t["i"] // 5) % max(1, n_rows // args.smoke) == 0) if args.smoke else True)
            test = m & (t["i"] % 5 == 0)
            train = m & ~test
            r = {"planes train / test": [int(train.sum()), int(test.sum())]}
            sh = t["n_ann"] / t["n_ann"].sum(1, keepdims=True)
            B, opt = fit_shares(x[train], sh[train])
            pred = softmax(x[test] @ B, axis=1)
            r["shares law converged"] = bool(opt.success)
            r["shares held-out observed / law per annulus, by |cos z|"] = {
                f"{lo}-{hi}": [round(float(a / b), 3) if b > 1e-4 else None for a, b in
                               zip(sh[test & (t["cos_z"] >= lo) & (t["cos_z"] < hi)].mean(0),
                                   pred[(t["cos_z"][test] >= lo) & (t["cos_z"][test] < hi)].mean(0))]
                for lo, hi in zip(annuli.COS_BANDS[:-1], annuli.COS_BANDS[1:])}
            C, held = fit_cell_energy(x, t["e_ann"], t["n_ann"], train, test)
            r["cell law held-out residual mean, sd per annulus"] = held
            ln_n = np.log(t["n_ann"].sum(1))
            c_n = np.linalg.lstsq(x[train], ln_n[train], rcond=None)[0]
            res_n = (ln_n - x @ c_n)[test]
            r["ln cells held-out R2, sd"] = [round(float(1 - res_n.var() / ln_n[test].var()), 3), round(float(res_n.std()), 3)]
            # the cell law's residual pattern across annuli, one number per plane
            w = t["n_ann"] * (t["n_ann"] > 0)
            res_e = np.where(w > 0, cell_target(t["e_ann"], t["n_ann"]) - x @ C, 0.0)
            enough = (w > 0).sum(1) >= PATTERN_MIN_ANN
            fit, held_p = train & enough, test & enough
            g = fit_pattern(res_e[fit], w[fit], rng)
            a = pattern_scores(res_e[fit], w[fit], g)
            rho = lag1(a, t["i"][fit], t["k"][fit])
            r["pattern: lag-1 rho, variance removed train / held-out"] = [
                round(rho, 3), round(removed(res_e[fit], w[fit], g), 4), round(removed(res_e[held_p], w[held_p], g), 4)]
            model.update({f"{s}.{cn}.N": B, f"{s}.{cn}.cell": C, f"{s}.{cn}.count": c_n,
                          f"{s}.{cn}.count_sd": np.array(res_n.std()), f"{s}.{cn}.cell_mode": g,
                          f"{s}.{cn}.cell_rho": np.array(rho),
                          f"{s}.{cn}.delay": np.nan_to_num(np.nanmedian(t["delay"][m], 0))})
            res[s][cn] = r
            log.info("%s %s %s", s, cn, {k: v for k, v in r.items() if "per annulus" not in k})
    bad = [k for k, v in model.items() if not np.all(np.isfinite(v))]
    if bad:
        raise ValueError(f"laws not finite: {bad}")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, args.model), edges=annuli.EDGES, **model)
    return res


def plane_cells(k, d, e_k, x, law, rng, a):
    """Cells [x, y, plane, energy, time] of one plane; a: the plane's pattern number."""
    n = max(1, int(round(np.exp(x @ law["count"] + law["count_sd"] * rng.normal()))))
    counts = rng.multinomial(n, softmax(x @ law["N"]))
    w = counts * np.exp(x @ law["cell"] + a * law["cell_mode"])     # energy per annulus = cells x energy per cell
    e_ann = e_k * w / w.sum()
    j = np.repeat(np.arange(len(counts)), counts)
    lo, hi = annuli.EDGES[j], annuli.EDGES[j + 1]
    hi = np.where(np.isfinite(hi), hi, R_OPEN * lo)
    r = np.sqrt(rng.uniform(lo ** 2, hi ** 2))
    phi = rng.uniform(0, 2 * np.pi, len(r))
    h, perp, c = lateral.frame(d)
    ground = ((k + 1) * atmosphere.PLANE_SPACING * d[:2] + (r * np.cos(phi))[:, None] * perp
              + (r * np.sin(phi) * c)[:, None] * h)
    t = (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT + law["delay"][j]
    return np.column_stack([ground, np.full(len(r), k), (e_ann / np.maximum(counts, 1))[j], t])


def stage_draw(out, args, log):
    model = dict(np.load(os.path.join(out, args.model)))
    rows = np.load(os.path.join(args.out, ROWS))["rows"]
    rows = rows[:: max(1, len(rows) // args.smoke)] if args.smoke else rows
    zt = np.load(os.path.join(args.out, m1.FOLDER, "fit", "fits_test.npz"))
    l, u, lab, X = zt["meta.log10_E"][rows], zt["meta.cos_zenith"][rows], zt["meta.labels"][rows], zt["X"][rows]
    dirs = zt["meta.directions"][rows] / np.linalg.norm(zt["meta.directions"][rows], axis=1, keepdims=True)
    rng = np.random.default_rng(args.seed)
    E = {s: np.zeros((len(rows), N_PLANES)) for s in SPECIES_NAMES}
    xmax = {s: np.zeros(len(rows)) for s in SPECIES_NAMES}
    if args.oracle:      # each primary's own CORSIKA plane energies and fitted Xmax: isolates the in-plane laws
        for s in SPECIES_NAMES:
            E[s], xmax[s] = zt[f"{s}.E"][rows].astype(np.float64), zt[f"{s}.{m2.NAME}.xmax"][rows]
    else:
        tr, fits = score.load_train(args, log)
        for c in m1.CLASS:
            m = lab == c
            d, lat = sampler.sample(sampler.fit(tr, fits, c), l[m], u[m], X[m], rng, return_latent=True)
            for i, s in enumerate(SPECIES_NAMES):
                E[s][m] = d[s]
                xmax[s][m] = latent.from_latent(sampler.species_block(lat, i)[0], l[m])[0]
    clouds = {}
    for s in SPECIES_NAMES:
        per = []
        for i in range(len(rows)):
            law = {w: model[f"{s}.{m1.CLASS[lab[i]]}.{w}"]
                   for w in ("N", "cell", "cell_mode", "count", "count_sd", "delay")}
            a = sampler.ar1(1, N_PLANES, float(model[f"{s}.{m1.CLASS[lab[i]]}.cell_rho"]), rng)[0]
            ks = np.nonzero(E[s][i] > 0)[0]
            x = annuli.features(age(X[i, ks], xmax[s][i]), np.full(len(ks), l[i]), np.full(len(ks), u[i]),
                                np.log(E[s][i, ks]), np.array([lateral.r_m_ref(k, dirs[i, 2]) for k in ks]))
            cells = [plane_cells(k, dirs[i], E[s][i, k], x[j], law, rng, a[k]) for j, k in enumerate(ks)]
            per.append(np.concatenate(cells) if cells else np.zeros((0, 5)))
            if cells and not np.allclose(np.bincount(per[-1][:, 2].astype(int), per[-1][:, 3], N_PLANES), E[s][i]):
                raise ValueError("generated cells do not sum to the sampled plane energies")
        cap = max(len(p) for p in per)
        clouds[f"{s}.points"] = np.stack([np.pad(p, ((0, cap - len(p)), (0, 0))) for p in per]).astype(np.float32)
        clouds[f"{s}.E"] = E[s]
        log.info("%s: cells per shower median %d, max %d", s, int(np.median([len(p) for p in per])), cap)
    name = "clouds" + ("_oracle" if args.oracle else "") + (f"_{os.path.splitext(args.model)[0]}"
                                                           if args.model != "model.npz" else "")
    np.savez(os.path.join(out, name + ".npz"), rows=rows, labels=lab, **clouds)
    return {"n_primaries": len(rows), "seed": args.seed, "clouds": name + ".npz"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output folder of m1.py and annuli.py")
    ap.add_argument("--stage", required=True, choices=["fit", "draw"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", action="store_true", help="draw: CORSIKA's own plane energies and Xmax per primary")
    ap.add_argument("--model", default="model.npz", help="model file in the generator folder (fit writes, draw reads)")
    ap.add_argument("--smoke", type=int, default=0, help="fit on about SMOKE showers spread over the slice, draw SMOKE primaries")
    args = ap.parse_args()
    folder = FOLDER + ("_smoke" if args.smoke else "")
    name = args.stage + ("_oracle" if args.oracle else "") + (f"_{os.path.splitext(args.model)[0]}"
                                                             if args.model != "model.npz" else "")
    log_dir, log = logs.start(os.path.join(folder, "logs", name), args)
    stage = {"fit": stage_fit, "draw": stage_draw}[args.stage]
    logs.save(log_dir, stage(os.path.join(args.out, folder), args, log), args, log)


if __name__ == "__main__":
    main()

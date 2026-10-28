"""Extrapolation split: the per-plane energy sampler fitted below log10 E = CUT, scored on the test showers in
[CUT, TOP) against CORSIKA, next to the sampler fitted on the full range and the neural chain (which saw these
energies in training).

Metric: energy score of log10(1 + E_k) on planes 2-24 (score.py), paired per test primary. Rule (fixed before the
run): the laws extrapolate in a class if the restricted sampler minus the neural chain, joint score, is not worse by
more than 2 standard errors.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import chain, fit_profiles, planes, sampler, score

FOLDER = "extrapolate"
CUT = 5.9
N_DRAWS = 8
MIN_CHAIN = 100            # chain primaries per class in the band below which comparison (b) is underpowered
RESTRICTED, FULL, CHAIN = "sampler, restricted fit", "sampler", "neural chain"


def draws(model, l, u, X, M, rng):
    """M sampler draws per primary: {species: (n, M, planes)}."""
    out = {s: np.zeros((len(l), M, X.shape[1])) for s in SPECIES_NAMES}
    for j in range(M):
        d, _ = sampler.sample(model, l, u, X, rng)
        for s in SPECIES_NAMES:
            out[s][:, j] = d[s]
    return out


def scores(dr, cor):
    """Energy score per primary, per species and joint."""
    y = {s: score.obs(cor[s]) for s in SPECIES_NAMES}
    sc = {s: score.energy_score(score.obs(dr[s]), y[s]) for s in SPECIES_NAMES}
    sc["joint"] = score.energy_score(np.concatenate([score.obs(dr[s]) for s in SPECIES_NAMES], 2),
                                     np.concatenate([y[s] for s in SPECIES_NAMES], 1))
    return sc


def paired(a, b, m=slice(None)):
    return {s: score.paired(a[s][m], b[s][m]) for s in a}


def verdict(diff, n):
    mean, se = diff["joint"]
    if n < MIN_CHAIN:
        return "UNDERPOWERED"
    return "FAIL" if mean > 2 * se else "PASS"


def stage_run(out, args, log):
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    if not np.array_equal(zt["meta.rows"], np.arange(len(zt["meta.rows"]))):
        raise ValueError("scoring needs the profile fits of the whole test file")
    l_all, u_all, lab_all = (zt[f"meta.{k}"] for k in ("log10_E", "cos_zenith", "labels"))
    X_all = zt["X"]
    chain_rows, nn = score.load_draws(args.out, chain.DRAWS_BAND)
    if np.any(np.diff(chain_rows) <= 0):
        raise ValueError("neural chain rows are not sorted and unique")
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    rng = np.random.default_rng(args.seed)
    res = {"cut": args.cut, "top": sampler.TOP, "n_draws": args.draws, "rule": __doc__.split("Rule")[1].strip(),
           "classes": {}}
    models_fig, cor_fig, lab_fig = {FULL: [], RESTRICTED: []}, [], []
    for c, cn in planes.CLASS.items():
        full, restr = sampler.fit(tr, c), sampler.fit(tr, c, top=args.cut)
        used = tr["meta"]["log10_E"][sampler.class_rows(tr, c, args.cut)["rows"]]
        if used.max() >= args.cut or restr["n_train"] != len(used):
            raise ValueError(f"{cn}: the restricted fit saw showers at or above log10 E = {args.cut}")
        log.info("%s: full fit %d training showers, restricted fit %d (log10 E %.3f-%.3f)", cn, full["n_train"],
                 restr["n_train"], used.min(), used.max())

        band = np.nonzero((lab_all == c) & (l_all >= args.cut) & (l_all < sampler.TOP))[0]
        l, u, X = l_all[band], u_all[band], X_all[band]
        cor = {s: zt[f"{s}.E"][band].astype(np.float64) for s in SPECIES_NAMES}
        dr = {k: draws(m, l, u, X, args.draws, rng) for k, m in ((FULL, full), (RESTRICTED, restr))}
        sc = {k: scores(v, cor) for k, v in dr.items()}
        cost = paired(sc[RESTRICTED], sc[FULL])
        log.info("%s: %d test showers in [%.3f, %.3f) | restricted minus full %s", cn, len(band), args.cut,
                 sampler.TOP, {s: [round(x, 4) for x in v] for s, v in cost.items()})

        in_chain = np.isin(band, chain_rows)
        pos = np.searchsorted(chain_rows, band[in_chain])
        sc_nn = scores({s: nn[s][pos] for s in SPECIES_NAMES}, {s: cor[s][in_chain] for s in SPECIES_NAMES})
        sub = {k: {s: v[s][in_chain] for s in v} for k, v in sc.items()}
        vs_chain = {k: paired(sub[k], sc_nn) for k in (RESTRICTED, FULL)}
        cost_sub = paired(sub[RESTRICTED], sub[FULL])["joint"][0]
        adv = -vs_chain[FULL]["joint"][0]
        n_chain = int(in_chain.sum())
        v = verdict(vs_chain[RESTRICTED], n_chain)
        log.info("%s: %d chain primaries in the band | restricted minus chain %s | full minus chain %s", cn, n_chain,
                 {s: [round(x, 4) for x in w] for s, w in vs_chain[RESTRICTED].items()},
                 {s: [round(x, 4) for x in w] for s, w in vs_chain[FULL].items()})
        log.info("%s: extrapolation cost / full-fit advantage over the chain (joint, chain primaries) = %.4f / %.4f"
                 " | VERDICT %s", cn, cost_sub, adv, v)

        res["classes"][cn] = {
            "n_train": {"full": full["n_train"], "restricted": restr["n_train"]},
            "n_band": len(band), "n_band_chain": n_chain,
            "redrawn_frac": {k: m["n_redrawn"] / m["n_drawn"] for k, m in ((FULL, full), (RESTRICTED, restr))},
            "energy_score_mean": {k: {s: float(w.mean()) for s, w in p.items()} for k, p in sc.items()},
            "restricted minus full": cost,
            "restricted minus neural chain": vs_chain[RESTRICTED], "full minus neural chain": vs_chain[FULL],
            "cost_over_advantage": cost_sub / adv if adv > 0 else None, "verdict": v}
        for k in models_fig:
            models_fig[k].append(dr[k])
        cor_fig.append(cor)
        lab_fig.append(np.full(len(band), c))

    cat = lambda parts: {s: np.concatenate([p[s] for p in parts]) for s in SPECIES_NAMES}
    fig = score.fig_medians(cat(cor_fig), {k: cat(v) for k, v in models_fig.items()}, np.concatenate(lab_fig))
    logs.save_fig(fig, out, "band_medians", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", type=float, default=CUT)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()

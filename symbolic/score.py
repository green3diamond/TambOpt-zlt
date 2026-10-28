"""Scores the per-plane energy sampler against CORSIKA, paired per test primary with the neural chain (chain.py draws).

Metric: energy score (lower is better) of log10(1 + E_k) on planes 2-24, per species and for all three together.
"""
import argparse
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import chain, fit_profiles, planes, sampler

FOLDER = "score"
PLANES = slice(1, None)
COLORS = {"CORSIKA": "black", "sampler": "steelblue", "neural chain": "#eb6834", "sampler, restricted fit": "#2ca02c"}


def obs(E):
    return np.log10(1.0 + E[..., PLANES])


def energy_score(Xs, y):
    """Unbiased energy score per primary of M draws Xs (n, M, d) against the observation y (n, d)."""
    M = Xs.shape[1]
    t1 = np.linalg.norm(Xs - y[:, None], axis=2).mean(1)
    pair = np.linalg.norm(Xs[:, :, None] - Xs[:, None], axis=3)
    return t1 - pair.sum((1, 2)) / (2 * M * (M - 1))


def paired(a, b):
    d = a - b
    return [float(d.mean()), float(d.std() / np.sqrt(len(d)))]


def load_draws(root, folder=chain.DRAWS):
    paths = sorted(glob.glob(os.path.join(root, folder, "draw_*.npz")))
    if len(paths) < 2:
        raise ValueError(f"need at least 2 neural chain draws, found {len(paths)}")
    zs = [np.load(p) for p in paths]
    rows = zs[0]["rows"]
    if any(not np.array_equal(z["rows"], rows) for z in zs):
        raise ValueError("neural chain draws were made on different primaries")
    return rows, {s: np.stack([z[f"{s}.E"] for z in zs], 1) for s in SPECIES_NAMES}


def fig_medians(cor, models, labels):
    k_planes = np.arange(1, cor[SPECIES_NAMES[0]].shape[1] + 1)
    fig = plt.figure(figsize=(len(SPECIES_NAMES) * 4.2, len(planes.CLASS) * 4.8))
    outer = gridspec.GridSpec(len(planes.CLASS), len(SPECIES_NAMES), figure=fig, hspace=0.45, wspace=0.38)
    for r, (c, cn) in enumerate(planes.CLASS.items()):
        m = labels == c
        for j, s in enumerate(SPECIES_NAMES):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, j], height_ratios=[3, 1], hspace=0.08)
            ax, ar = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            ref = np.median(cor[s][m], 0)
            ax.plot(k_planes, ref, color=COLORS["CORSIKA"], lw=1.5, drawstyle="steps-mid", label="CORSIKA")
            for k, v in models.items():
                med = np.median(v[s][m].reshape(-1, len(k_planes)), 0)
                ax.plot(k_planes, med, color=COLORS[k], lw=1.5, drawstyle="steps-mid", label=k)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ar.plot(k_planes, med / ref, color=COLORS[k], lw=1.2, drawstyle="steps-mid")
            ar.axhline(1, color="gray", lw=0.8, ls="--")
            ar.set_ylim(0.5, 1.5)
            ar.set_xlabel("Plane", fontsize=8)
            ax.tick_params(labelbottom=False)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3))
            ax.set_ylim(bottom=0)
            ax.set_title(f"{s}, {cn} ({m.sum()} test primaries)", fontsize=9)
            if j == 0:
                ax.set_ylabel("Median energy [GeV]", fontsize=8)
                ar.set_ylabel("ratio to CORSIKA", fontsize=7)
            if r == 0 and j == 0:
                ax.legend(fontsize=7)
    return fig


def stage_run(out, args, log):
    rows, nn = load_draws(args.out)
    M = next(iter(nn.values())).shape[1]
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    if not np.array_equal(zt["meta.rows"], np.arange(len(zt["meta.rows"]))):
        raise ValueError("scoring needs the profile fits of the whole test file")
    l, u, lab, X = zt["meta.log10_E"][rows], zt["meta.cos_zenith"][rows], zt["meta.labels"][rows], zt["X"][rows]
    if l.max() >= sampler.TOP:
        raise ValueError("scored primaries reach the left-out top energy bin")
    cor = {s: zt[f"{s}.E"][rows].astype(np.float64) for s in SPECIES_NAMES}
    log.info("%d primaries (EM %d, hadronic %d), %d draws each", len(rows), (lab == 0).sum(), (lab == 1).sum(), M)
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    te = sampler.load(os.path.join(args.out, fit_profiles.TEST))
    sym = {s: np.zeros_like(nn[s]) for s in SPECIES_NAMES}
    rng = np.random.default_rng(args.seed)
    res = {"n_primaries": len(rows), "n_draws": M, "top": sampler.TOP, "classes": {}}
    for c, cn in planes.CLASS.items():
        model = sampler.fit(tr, c)
        chk = sampler.check(model, te, c)
        log.info("%s sampler: %d training showers | held-out tail rates %s | log c per shower %s", cn,
                 model["n_train"], chk["tail_rate"], {k: np.round(v, 3).tolist() for k, v in chk["logc_per_shower"].items()})
        m = lab == c
        for j in range(M):
            d, _ = sampler.sample(model, l[m], u[m], X[m], rng)
            for s in SPECIES_NAMES:
                sym[s][m, j] = d[s]
        redrawn = model["n_redrawn"] / model["n_drawn"]
        log.info("%s draws ending inside the planes, drawn again: %.4f", cn, redrawn)
        y = {s: obs(cor[s][m]) for s in SPECIES_NAMES}
        sc = {}
        for k, dr in (("sampler", sym), ("neural chain", nn)):
            sc[k] = {s: energy_score(obs(dr[s][m]), y[s]) for s in SPECIES_NAMES}
            sc[k]["joint"] = energy_score(np.concatenate([obs(dr[s][m]) for s in SPECIES_NAMES], 2),
                                          np.concatenate([y[s] for s in SPECIES_NAMES], 1))
        diff = {s: paired(sc["sampler"][s], sc["neural chain"][s]) for s in sc["sampler"]}
        res["classes"][cn] = {"n": int(m.sum()), "n_train": model["n_train"], "check": chk, "redrawn_frac": redrawn,
                              "energy_score_mean": {k: {s: float(v.mean()) for s, v in p.items()} for k, p in sc.items()},
                              "sampler minus neural chain": diff}
        log.info("%-8s sampler minus neural chain, energy score mean +- se %s", cn,
                 {s: [round(x, 4) for x in v] for s, v in diff.items()})
    logs.save_fig(fig_medians(cor, {"sampler": sym, "neural chain": nn}, lab), out, "score_medians", log)
    np.savez(os.path.join(out, "sampler_draws.npz"), rows=rows, **{f"{s}.E": sym[s] for s in SPECIES_NAMES})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()

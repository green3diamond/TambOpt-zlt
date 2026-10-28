"""Scores the per-plane energy sampler against CORSIKA, paired per test primary with the neural generator.

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

from analysis_core import logs, plots
from modules.constants import SPECIES_NAMES
from symbolic import m1, m2, sampler

FOLDER = "score"
PLANES = slice(1, None)
DRAWS = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic/score/draws"
COLORS = {"CORSIKA": "black", "sampler": "steelblue", "generator": "#eb6834"}
VARIANTS = {"sampler": {}, "sampler, curve only": {"planes": False}, "sampler, species independent": {"independent": True}}


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


def load_draws():
    paths = sorted(glob.glob(os.path.join(DRAWS, "draw_*.npz")))
    if len(paths) < 2:
        raise ValueError(f"need at least 2 generator draws, found {len(paths)}")
    zs = [np.load(p) for p in paths]
    rows = zs[0]["rows"]
    if any(not np.array_equal(z["rows"], rows) for z in zs):
        raise ValueError("generator draws were made on different primaries")
    return rows, {s: np.stack([z[f"{s}.E"] for z in zs], 1) for s in SPECIES_NAMES}


def load_train(args, log):
    """Profile numbers and fit arrays of the training showers."""
    path = os.path.join(args.out, m2.FOLDER, "fit", "fits_train_full.npz")
    tr = m2.load_latent(path, log)
    z = np.load(path)
    keep = z["meta.log10_E"] <= m2.LOG10_E_MAX
    fits = {k: z[k][keep] for k in z.files if k == "X" or k.split(".")[0] in SPECIES_NAMES}
    return tr, fits


def fig_medians(cor, models, labels):
    planes = np.arange(1, cor[SPECIES_NAMES[0]].shape[1] + 1)
    fig = plt.figure(figsize=(len(SPECIES_NAMES) * 4.2, len(m1.CLASS) * 4.8))
    outer = gridspec.GridSpec(len(m1.CLASS), len(SPECIES_NAMES), figure=fig, hspace=0.45, wspace=0.38)
    for r, (c, cn) in enumerate(m1.CLASS.items()):
        m = labels == c
        for j, s in enumerate(SPECIES_NAMES):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, j], height_ratios=[3, 1], hspace=0.08)
            ax, ar = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            ref = np.median(cor[s][m], 0)
            ax.plot(planes, ref, color=COLORS["CORSIKA"], lw=1.5, drawstyle="steps-mid", label="CORSIKA")
            for k in ("sampler", "generator"):
                med = np.median(models[k][s][m].reshape(-1, len(planes)), 0)
                ax.plot(planes, med, color=COLORS[k], lw=1.5, drawstyle="steps-mid", label=k)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ar.plot(planes, med / ref, color=COLORS[k], lw=1.2, drawstyle="steps-mid")
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
    rows, gen = load_draws()
    M = next(iter(gen.values())).shape[1]
    zt = np.load(os.path.join(args.out, m1.FOLDER, "fit", "fits_test.npz"))
    if not np.array_equal(zt["meta.rows"], np.arange(len(zt["meta.rows"]))):
        raise ValueError("fits_test.npz rows are not the test file in order")
    l, u, lab, X = zt["meta.log10_E"][rows], zt["meta.cos_zenith"][rows], zt["meta.labels"][rows], zt["X"][rows]
    if l.max() >= sampler.TOP:
        raise ValueError("scored primaries reach the left-out top energy bin")
    cor = {s: zt[f"{s}.E"][rows].astype(np.float64) for s in SPECIES_NAMES}
    log.info("%d primaries (EM %d, hadronic %d), %d generator draws each", len(rows), (lab == 0).sum(), (lab == 1).sum(), M)
    tr, fits = load_train(args, log)
    te = m2.load_latent(os.path.join(args.out, m1.FOLDER, "fit", "fits_test.npz"), log)
    draws = {k: {s: np.zeros_like(gen[s]) for s in SPECIES_NAMES} for k in VARIANTS}
    rng = np.random.default_rng(args.seed)
    res = {"n_primaries": len(rows), "n_draws": M, "top": sampler.TOP, "curve": m2.NAME, "classes": {}}
    for c, cn in m1.CLASS.items():
        model = sampler.fit(tr, fits, c)
        chk = sampler.check(model, te, c)
        p = model["planes"]
        log.info("%s sampler: %d training showers | held-out tail rates %s | log c per shower %s | plane noise lag-1 %s",
                 cn, model["n_train"], chk["tail_rate"], {k: np.round(v, 3).tolist() for k, v in chk["logc_per_shower"].items()},
                 {s: round(v["rho"], 2) for s, v in p.items()})
        m = lab == c
        for k, kw in VARIANTS.items():
            for j in range(M):
                d = sampler.sample(model, l[m], u[m], X[m], rng, **kw)
                for s in SPECIES_NAMES:
                    draws[k][s][m, j] = d[s]
        redrawn = model["n_redrawn"] / model["n_drawn"]
        log.info("%s draws ending inside the planes, drawn again: %.4f", cn, redrawn)
        res["classes"][cn] = {"n_train": model["n_train"], "check": chk, "redrawn_frac": redrawn,
                              "planes": {s: {k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v)
                                             for k, v in pp.items()} for s, pp in p.items()}}
    models = {"generator": gen, **draws}
    y = {s: obs(cor[s]) for s in SPECIES_NAMES}
    for c, cn in m1.CLASS.items():
        m = lab == c
        sc = {}
        for k, dr in models.items():
            sc[k] = {s: energy_score(obs(dr[s][m]), y[s][m]) for s in SPECIES_NAMES}
            sc[k]["joint"] = energy_score(np.concatenate([obs(dr[s][m]) for s in SPECIES_NAMES], 2),
                                          np.concatenate([y[s][m] for s in SPECIES_NAMES], 1))
        r = res["classes"][cn]
        r["n"] = int(m.sum())
        r["energy_score_mean"] = {k: {s: float(v.mean()) for s, v in per.items()} for k, per in sc.items()}
        r["vs_generator"] = {k: {s: paired(v, sc["generator"][s]) for s, v in per.items()}
                             for k, per in sc.items() if k != "generator"}
        r["plane1_empty_frac"] = {"CORSIKA": {s: float(np.mean(cor[s][m, 0] == 0)) for s in SPECIES_NAMES},
                                  **{k: {s: float(np.mean(dr[s][m, :, 0] == 0)) for s in SPECIES_NAMES}
                                     for k, dr in models.items()}}
        for k, per in sc.items():
            log.info("%-8s %-30s energy score %s", cn, k, {s: round(float(v.mean()), 4) for s, v in per.items()})
        for k, v in r["vs_generator"].items():
            log.info("%-8s %-30s minus generator %s", cn, k, {s: [round(x, 4) for x in d] for s, d in v.items()})
    plots.save(fig_medians(cor, {"sampler": draws["sampler"], "generator": gen}, lab), out, "score_medians", log)
    np.savez(os.path.join(out, "sampler_draws.npz"), rows=rows, **{f"{s}.E": draws["sampler"][s] for s in SPECIES_NAMES})
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

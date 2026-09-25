"""Utility map for one detector added to, or moved within, a fixed layout."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from analysis_core import events, logs, plots
from analysis_core.run import LAYOUTS, Run, add_run_args
from analysis_core.scoring import PARTS, Scorer, paired


def scan_grid(run, spacing):
    """East and North axes and the on-mountain mask of a regular grid over the mesh."""
    m = run.mountain
    E = torch.arange(m.east_lo, m.east_hi + spacing, spacing)
    N = torch.arange(m.n_min, m.n_max + spacing, spacing)
    EE, NN = torch.meshgrid(E, N, indexing="xy")
    on = run.on_mountain(EE.reshape(-1), NN.reshape(-1)).reshape(EE.shape)
    return E, N, EE, NN, on


def pick_detector(e, n, which):
    if which not in ("center", "edge"):
        return int(which)
    d = (e - e.mean()) ** 2 + (n - n.mean()) ** 2
    return int(d.argmin() if which == "center" else d.argmax())


def place(e, n, pe, pn, detector):
    """Layout with a detector at (pe, pn): appended when detector is None, else moved there."""
    if detector is None:
        return torch.cat([e, e.new_tensor([pe])]), torch.cat([n, n.new_tensor([pn])])
    e, n = e.clone(), n.clone()
    e[detector], n[detector] = pe, pn
    return e, n


def spot_check(run, ev, args, e0, n0, points, detector, log):
    """Kernel paired differences at chosen grid points, beside the surrogate ones."""
    kernel = Scorer(run, ev, "kernel", args.chunk, args.kernel_chunk)
    base = kernel.score(e0, n0)
    rows = []
    for p in points:
        d, se = paired(kernel.score(*place(e0, n0, p["E"], p["N"], detector)), base)
        rows.append({**p, "kernel_dU": d, "kernel_se": se})
        log.info("spot E %7.1f N %7.1f  surrogate %+.4f +- %.4f  kernel %+.4f +- %.4f",
                 p["E"], p["N"], p["dU"], p["se"], d, se)
    return rows


def map_figure(run, E, N, dU, se, e0, n0, detector, best, checks):
    fig, axes = plots.plt.subplots(1, 2, figsize=(13, 5.5))
    lim = np.nanmax(np.abs(dU))
    panels = ((dU, "RdBu_r", TwoSlopeNorm(0, -lim, lim), "ΔU vs fixed layout"),
              (dU / se, "RdBu_r", TwoSlopeNorm(0, -5, 5), "ΔU / standard error"))
    for ax, (Z, cmap, norm, label) in zip(axes, panels):
        plots.footprint(ax, run.mountain)
        im = ax.pcolormesh(E, N, Z, cmap=cmap, norm=norm, shading="nearest")
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.scatter(e0, n0, s=8, c="k", label="fixed detectors")
        if detector is not None:
            ax.scatter(e0[detector], n0[detector], marker="*", s=200, c="gold", edgecolor="k", label="moved detector")
        ax.scatter(best["E"], best["N"], marker="X", s=120, c="cyan", edgecolor="k", label="best point")
        ax.scatter([c["E"] for c in checks], [c["N"] for c in checks], s=60, facecolor="none", edgecolor="m",
                   label="kernel spot checks")
    axes[0].legend(fontsize=7, loc="lower left")
    return fig


def terms_figure(run, E, N, terms):
    fig, axes = plots.plt.subplots(1, len(PARTS), figsize=(6 * len(PARTS), 5))
    for ax, k in zip(axes, PARTS):
        lim = np.nanmax(np.abs(terms[k])) or 1.0
        plots.footprint(ax, run.mountain)
        im = ax.pcolormesh(E, N, terms[k], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
        fig.colorbar(im, ax=ax, label=f"Δ{k}", shrink=0.8)
    return fig


def profile_figure(E, N, dU, se, e0, n0):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 3.8))
    i = int(torch.argmin((N - n0.mean()).abs()))
    j = int(torch.argmin((E - e0.mean()).abs()))
    for ax, x, y, s, label in ((axes[0], E, dU[i], se[i], f"East, at North {N[i]:.0f} m"),
                               (axes[1], N, dU[:, j], se[:, j], f"North, at East {E[j]:.0f} m")):
        ax.plot(x, y, color="C0")
        ax.fill_between(x, y - s, y + s, color="C0", alpha=0.3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{label} [m]")
        ax.set_ylabel("ΔU")
    return fig


def check_figure(checks):
    fig, ax = plots.plt.subplots(figsize=(4.5, 4.5))
    x, y = [c["dU"] for c in checks], [c["kernel_dU"] for c in checks]
    ax.errorbar(x, y, xerr=[c["se"] for c in checks], yerr=[c["kernel_se"] for c in checks], fmt="o")
    lim = max(map(abs, x + y)) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("surrogate ΔU")
    ax.set_ylabel("kernel ΔU")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layout", default="opt_grid", choices=LAYOUTS)
    ap.add_argument("--mode", default="add", choices=("add", "move"))
    ap.add_argument("--detector", default="center", help="move mode: center, edge or an index")
    ap.add_argument("--spacing", type=float, default=50.0)
    ap.add_argument("--kernel-points", type=int, default=10, help="best points and random points each")
    args = ap.parse_args()

    tag = args.mode if args.mode == "add" else f"move_{args.detector}"
    out, log = logs.start(f"single_detector_scan_{args.layout}_{tag}", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    sc = Scorer(run, ev, "surrogate", args.chunk)

    e0, n0 = run.layout(args.layout)
    detector = None if args.mode == "add" else pick_detector(e0, n0, args.detector)
    E, N, EE, NN, on = scan_grid(run, args.spacing)
    log.info("grid %d x %d, %d on the mountain, detector %s", len(N), len(E), int(on.sum()), detector)

    base = sc.score(e0, n0)
    dU, se = np.full(on.shape, np.nan), np.full(on.shape, np.nan)
    terms = {k: np.full(on.shape, np.nan) for k in PARTS}
    cells = torch.nonzero(on).tolist()
    for c, (i, j) in enumerate(cells):
        s = sc.score(*place(e0, n0, float(EE[i, j]), float(NN[i, j]), detector))
        dU[i, j], se[i, j] = paired(s, base)
        for k in PARTS:
            terms[k][i, j] = s.parts[k].mean() - base.parts[k].mean()
        if (c + 1) % 100 == 0:
            log.info("%d / %d points", c + 1, len(cells))

    flat = [{"E": float(EE[i, j]), "N": float(NN[i, j]), "dU": float(dU[i, j]), "se": float(se[i, j])}
            for i, j in cells]
    ranked = sorted(flat, key=lambda p: -p["dU"])
    rng = np.random.default_rng(args.seed)
    picks = ranked[: args.kernel_points] + [flat[k] for k in rng.choice(len(flat), args.kernel_points, replace=False)]
    checks = spot_check(run, ev, args, e0, n0, picks, detector, log)
    best = ranked[0]
    log.info("base U %.4f +- %.4f  best ΔU %+.4f +- %.4f at E %.1f N %.1f  worst ΔU %+.4f",
             base.U, base.se, best["dU"], best["se"], best["E"], best["N"], ranked[-1]["dU"])

    plots.save(map_figure(run, E, N, dU, se, e0, n0, detector, best, checks), out, "map", log)
    plots.save(terms_figure(run, E, N, terms), out, "terms", log)
    plots.save(profile_figure(E, N, dU, se, e0, n0), out, "profiles", log)
    plots.save(check_figure(checks), out, "kernel_spot_checks", log)
    logs.save(out, {"layout": args.layout, "mode": args.mode, "detector": detector, "base": base.summary(),
                    "E": E.tolist(), "N": N.tolist(), "dU": dU.tolist(), "se": se.tolist(),
                    "terms": {k: v.tolist() for k, v in terms.items()}, "best": best,
                    "kernel_spot_checks": checks}, args, log)


if __name__ == "__main__":
    main()

"""Which detectors and areas matter: leave-one-out, region knockout, geometry, and consistency across optima."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from analysis_core import events, logs, plots, stats
from analysis_core.run import LAYOUTS, SCHEMES, Run, add_run_args
from analysis_core.scoring import Scorer, paired

FEATURES = ("elevation", "depth", "nearest_neighbour", "from_centroid", "from_edge")


def knockout(scorer, e, n, groups):
    """ΔU (full minus without) and standard error for removing each group of detectors."""
    full = scorer.score(e, n)
    rows = []
    for g in groups:
        keep = torch.ones(len(e), dtype=torch.bool)
        keep[g] = False
        rows.append(paired(full, scorer.score(e[keep], n[keep])))
    return np.array(rows)


def cell_ids(run, e, n, size):
    """Square cell of each detector on a grid anchored at the mesh bounding box."""
    m = run.mountain
    return (((n - m.n_min) // size).long() * 10000 + ((e - m.east_lo) // size).long()).numpy()


def cell_centre(run, cid, size):
    return run.mountain.east_lo + (cid % 10000 + 0.5) * size, run.mountain.n_min + (cid // 10000 + 0.5) * size


def edge_distance(run, e, n, spacing=25.0, pad=200.0):
    """Distance from each detector to the nearest off-mountain point of a fine grid."""
    m = run.mountain
    E = torch.arange(m.east_lo - pad, m.east_hi + pad, spacing)
    N = torch.arange(m.n_min - pad, m.n_max + pad, spacing)
    EE, NN = torch.meshgrid(E, N, indexing="xy")
    off = ~run.on_mountain(EE.reshape(-1), NN.reshape(-1))
    pts = torch.stack([EE.reshape(-1)[off], NN.reshape(-1)[off]], -1)
    return torch.cdist(torch.stack([e, n], -1), pts).min(1).values.numpy()


def geometry(run, e, n):
    d = torch.cdist(torch.stack([e, n], -1), torch.stack([e, n], -1))
    d.fill_diagonal_(float("inf"))
    up = run.surface(n.to(run.device), e.to(run.device)).cpu().numpy()
    return {"elevation": up, "depth": run.mountain.east_to_z_cont(e).numpy(),
            "nearest_neighbour": d.min(1).values.numpy(),
            "from_centroid": torch.hypot(e - e.mean(), n - n.mean()).numpy(),
            "from_edge": edge_distance(run, e, n)}


def corr_ci(x, y, n_boot):
    """Pearson r and its bootstrap interval over detectors."""
    r = lambda i: np.corrcoef(x[i], y[i])[0, 1]
    return float(r(np.arange(len(x)))), stats.bootstrap_ci(np.arange(len(x)), stat=r, n_boot=n_boot)


def analyse(run, scorer, e, n, size, n_boot):
    loo = knockout(scorer, e, n, [[i] for i in range(len(e))])
    ids = cell_ids(run, e, n, size)
    cells = sorted(set(ids.tolist()))
    region = knockout(scorer, e, n, [np.nonzero(ids == c)[0].tolist() for c in cells])
    geo = geometry(run, e, n)
    corr = {f: corr_ci(loo[:, 0], geo[f], n_boot) for f in FEATURES}
    return {"loo": loo, "cells": cells, "cell_counts": [int((ids == c).sum()) for c in cells],
            "region": region, "geometry": geo, "correlations": corr}


def loo_figure(run, e, n, loo):
    fig, ax = plots.plt.subplots(figsize=(6.5, 6))
    plots.footprint(ax, run.mountain)
    lim = np.abs(loo[:, 0]).max()
    sc = ax.scatter(e, n, c=loo[:, 0], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                    s=20 + 60 * np.clip(np.abs(loo[:, 0] / loo[:, 1]), 0, 5) / 5, edgecolor="k", lw=0.3)
    fig.colorbar(sc, ax=ax, label="U lost when removed (size: significance up to 5 SE)", shrink=0.8)
    return fig


def draw_regions(ax, run, size, res, title):
    """Cells coloured by U lost when their detectors are removed, labelled ΔU, SE and detector count."""
    plots.footprint(ax, run.mountain)
    lim = np.abs(res["region"][:, 0]).max()
    cmap, norm = plots.plt.get_cmap("RdBu_r"), TwoSlopeNorm(0, -lim, lim)
    for cid, (d, se), k in zip(res["cells"], res["region"], res["cell_counts"]):
        ce, cn = cell_centre(run, cid, size)
        ax.add_patch(plots.plt.Rectangle((ce - size / 2, cn - size / 2), size, size, color=cmap(norm(d)), alpha=0.85))
        ax.text(ce, cn, f"{d:+.2f}\n±{se:.2f}\nn={k}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    ax.autoscale_view()


def region_figure(run, size, panels):
    fig, axes = plots.plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), squeeze=False)
    for ax, (title, res) in zip(axes[0], panels):
        draw_regions(ax, run, size, res, title)
    return fig


def geometry_figure(res):
    fig, axes = plots.plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 3.5), sharey=True)
    for ax, f in zip(axes, FEATURES):
        r, (lo, hi) = res["correlations"][f]
        ax.errorbar(res["geometry"][f], res["loo"][:, 0], yerr=res["loo"][:, 1], fmt="o", ms=3, alpha=0.7)
        ax.set_title(f"r = {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        ax.set_xlabel(f)
    axes[0].set_ylabel("U lost when removed")
    return fig


def consistency_figure(names, matrix, per_optimum):
    fig, axes = plots.plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(im, ax=axes[0], label="correlation of region knockout ΔU", shrink=0.8)
    for i, f in enumerate(FEATURES):
        axes[1].scatter(np.full(len(per_optimum), i), [p[f] for p in per_optimum], s=12)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(range(len(FEATURES)), FEATURES, rotation=30)
    axes[1].set_ylabel("r with leave-one-out ΔU, one point per optimum")
    return fig


def scan_figure(run, size, scan_path, source, res):
    scan = json.load(open(scan_path))
    fig, axes = plots.plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    dU = np.array(scan["dU"], dtype=float)
    lim = np.nanmax(np.abs(dU))
    plots.footprint(axes[0], run.mountain)
    im = axes[0].pcolormesh(scan["E"], scan["N"], dU, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
    fig.colorbar(im, ax=axes[0], label="ΔU from adding one detector here", shrink=0.8)
    axes[0].set_title("value of adding a detector (surrogate)")
    draw_regions(axes[1], run, size, res, f"U lost by removing each cell's detectors ({source})")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layout", default="opt_grid", choices=LAYOUTS)
    ap.add_argument("--cell", type=float, default=250.0, help="region knockout cell size [m]")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--no-chains", action="store_true", help="skip the 12 chain optima")
    ap.add_argument("--kernel-regions", action="store_true", help="region knockout with kernel labels too")
    ap.add_argument("--scan", default=None, help="results.json of the add-mode single-detector scan")
    args = ap.parse_args()

    out, log = logs.start(f"detector_importance_{args.layout}", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    sc = Scorer(run, ev, "surrogate", args.chunk)

    e, n = run.layout(args.layout)
    main_res = analyse(run, sc, e, n, args.cell, args.n_boot)
    loo = main_res["loo"]
    log.info("%s leave-one-out ΔU: max %.4f  mean %.4f  %d of %d above 2 SE  %d negative",
             args.layout, loo[:, 0].max(), loo[:, 0].mean(), int((loo[:, 0] > 2 * loo[:, 1]).sum()), len(loo),
             int((loo[:, 0] < 0).sum()))
    for f, (r, ci) in main_res["correlations"].items():
        log.info("correlation with %-18s r %+.3f  [%+.3f, %+.3f]", f, r, *ci)
    panels = [("surrogate", main_res)]
    if args.kernel_regions:
        kernel = Scorer(run, ev, "kernel", args.chunk, args.kernel_chunk)
        ids = cell_ids(run, e, n, args.cell)
        k_region = knockout(kernel, e, n, [np.nonzero(ids == c)[0].tolist() for c in main_res["cells"]])
        panels.append(("kernel", {**main_res, "region": k_region}))
        r = np.corrcoef(main_res["region"][:, 0], k_region[:, 0])[0, 1]
        log.info("region knockout surrogate vs kernel correlation %.3f over %d cells", r, len(k_region))

    chains = []
    if not args.no_chains:
        for scheme in SCHEMES:
            for i, (ce, cn) in enumerate(run.chains(scheme)):
                res = analyse(run, sc, ce, cn, args.cell, args.n_boot)
                chains.append((f"{scheme}_{i}", res))
                log.info("chain %s_%d: max ΔU %.4f  nearest-neighbour r %+.3f", scheme, i,
                         res["loo"][:, 0].max(), res["correlations"]["nearest_neighbour"][0])
        cells = sorted(set().union(*(r["cells"] for _, r in chains)))
        vec = np.array([[dict(zip(r["cells"], r["region"][:, 0])).get(c, np.nan) for c in cells] for _, r in chains])
        matrix = np.array([[np.corrcoef(a[~np.isnan(a) & ~np.isnan(b)], b[~np.isnan(a) & ~np.isnan(b)])[0, 1]
                            for b in vec] for a in vec])
        log.info("mean region knockout correlation between optima %.3f", np.nanmean(matrix[~np.eye(len(matrix), dtype=bool)]))
        plots.save(consistency_figure([c for c, _ in chains], matrix,
                                      [{f: r["correlations"][f][0] for f in FEATURES} for _, r in chains]),
                   out, "consistency_across_optima", log)

    plots.save(loo_figure(run, e, n, loo), out, "leave_one_out_map", log)
    plots.save(region_figure(run, args.cell, panels), out, "region_knockout_map", log)
    plots.save(geometry_figure(main_res), out, "geometry_correlations", log)
    if args.scan:
        plots.save(scan_figure(run, args.cell, args.scan, *panels[-1]), out, "adding_vs_removing", log)

    to_json = lambda r: {"loo": r["loo"].tolist(), "cells": r["cells"], "cell_counts": r["cell_counts"],
                         "region": r["region"].tolist(), "correlations": r["correlations"],
                         "geometry": {f: v.tolist() for f, v in r["geometry"].items()}}
    logs.save(out, {"layout": args.layout, "cell": args.cell, "main": to_json(main_res),
                    "kernel_region": panels[1][1]["region"].tolist() if args.kernel_regions else None,
                    "chains": {c: to_json(r) for c, r in chains}}, args, log)


if __name__ == "__main__":
    main()

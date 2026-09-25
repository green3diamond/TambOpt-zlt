"""U along the line from start to optimum and on 2D slices around the optimum, after Li et al. (arXiv 1712.09913)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from analysis_core import events, logs, plots
from analysis_core.run import ENSEMBLE_DIR, SCHEMES, Run, add_run_args
from analysis_core.scoring import LABELS, Scorer


def to_layout(flat):
    """Split a flat [East..., North...] vector into (East, North)."""
    k = flat.shape[-1] // 2
    return flat[..., :k], flat[..., k:]


def detector_direction(n_det, generator):
    """Random direction in which every detector moves one metre per unit step."""
    d = torch.randn(n_det, 2, generator=generator)
    d = d / d.norm(dim=1, keepdim=True)
    return torch.cat([d[:, 0], d[:, 1]])


def principal_directions(path, centre):
    """First two PCA directions of the path relative to the optimum, explained variance, and the path's coordinates."""
    M = (path - centre).double()
    _, s, vt = torch.linalg.svd(M, full_matrices=False)
    return vt[:2].float(), (s ** 2 / (s ** 2).sum()).float(), (M @ vt[:2].T).float()


def line(scorer, run, a, b, ts):
    rows = []
    for t in ts:
        e, n = to_layout((1 - t) * a + t * b)
        s = scorer.score(e, n)
        rows.append((s.U, s.se, float((~run.on_mountain(e, n)).float().mean())))
    return np.array(rows)


def plane(scorer, run, centre, u, v, alphas, betas):
    """U, standard error and off-mountain fraction on centre + alpha u + beta v; rows are beta."""
    out = np.zeros((3, len(betas), len(alphas)))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            e, n = to_layout(centre + a * u + b * v)
            s = scorer.score(e, n)
            out[:, i, j] = s.U, s.se, float((~run.on_mountain(e, n)).float().mean())
    return out


def line_figure(ts, rows):
    fig, ax = plots.plt.subplots(figsize=(7, 4))
    for src, r in rows.items():
        color = plots.KERNEL if src == "kernel" else plots.SURROGATE
        ax.plot(ts, r[:, 0], color=color, label=src)
        ax.fill_between(ts, r[:, 0] - r[:, 1], r[:, 0] + r[:, 1], color=color, alpha=0.3)
    twin = ax.twinx()
    twin.plot(ts, rows["surrogate"][:, 2], "k:")
    twin.set_ylabel("fraction of detectors off the mountain (dotted)")
    for t in (0, 1):
        ax.axvline(t, color="0.6", lw=0.8)
    ax.set_xlabel("t (0 = starting layout, 1 = optimum)")
    ax.set_ylabel("U")
    ax.legend()
    return fig


def plane_figure(alphas, betas, grid, max_off, xlabel, ylabel, path=None, n_adam=None):
    fig, ax = plots.plt.subplots(figsize=(6.5, 5.5))
    U = np.where(grid[2] > max_off, np.nan, grid[0])
    cs = ax.contourf(alphas, betas, U, levels=20, cmap="viridis")
    ax.contour(alphas, betas, U, levels=20, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, label=f"U (blank where more than {max_off:.0%} of detectors are off the mountain)")
    if path is not None:
        ax.plot(path[:n_adam, 0], path[:n_adam, 1], color="w", lw=1, label="Adam")
        ax.plot(path[n_adam:, 0], path[n_adam:, 1], color="r", lw=1, label="L-BFGS")
        ax.plot(path[0, 0], path[0, 1], "wo", ms=5)
        ax.legend(fontsize=8)
    ax.plot(0, 0, "*", color="gold", ms=14, mec="k")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def variance_figure(var):
    fig, ax = plots.plt.subplots(figsize=(5, 3.5))
    ax.bar(np.arange(1, len(var) + 1), var)
    ax.set_xlabel("PCA component of the optimizer path")
    ax.set_ylabel("explained variance fraction")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--scheme", default="grid", choices=SCHEMES)
    ap.add_argument("--line-points", type=int, default=29)
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--span", type=float, default=200.0, help="random slice half-width, metres per detector")
    ap.add_argument("--margin", type=float, default=0.2, help="PCA slice padding beyond the path")
    ap.add_argument("--max-off", type=float, default=0.1)
    ap.add_argument("--kernel", action="store_true", help="also score the 1D line with the kernel")
    args = ap.parse_args()

    out, log = logs.start(f"loss_slices_{args.scheme}", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    sc = Scorer(run, ev, "surrogate", args.chunk)

    traj = torch.load(os.path.join(args.run_dir, ENSEMBLE_DIR.format(args.scheme), "trajectory.pt"),
                      map_location="cpu", weights_only=False)
    # layout_best is the best-scoring L-BFGS iterate, not the last one, so it is the centre.
    k = traj["ref_idx"]
    n_adam = len(traj["adam"][k])
    path = torch.cat([traj["adam"][k], traj["lbfgs"][k]]).float()
    start, optimum = path[0], torch.cat(run.layout(f"opt_{args.scheme}"))
    offsets_m = torch.stack(to_layout(path - optimum), -1).norm(dim=-1).mean(-1)
    log.info("chain %d  path %d steps (%d Adam)  path end %.2f m per detector from layout_best, closest step %d (%.2f m)",
             k, len(path), n_adam, float(offsets_m[-1]), int(offsets_m.argmin()), float(offsets_m.min()))

    ts = np.linspace(-0.2, 1.2, args.line_points)
    rows = {"surrogate": line(sc, run, start, optimum, ts)}
    if args.kernel:
        rows["kernel"] = line(Scorer(run, ev, "kernel", args.chunk, args.kernel_chunk), run, start, optimum, ts)
    for src, r in rows.items():
        log.info("line %-9s U at start %.4f  at optimum %.4f  max %.4f at t %.2f", src,
                 r[np.argmin(np.abs(ts))][0], r[np.argmin(np.abs(ts - 1))][0], r[:, 0].max(), ts[r[:, 0].argmax()])

    g = torch.Generator().manual_seed(args.seed)
    n_det = len(optimum) // 2
    u, v = detector_direction(n_det, g), detector_direction(n_det, g)
    offsets = np.linspace(-args.span, args.span, args.grid)
    random_grid = plane(sc, run, optimum, u, v, offsets, offsets)
    log.info("random slice: U range %.4f to %.4f within ±%.0f m per detector", np.nanmin(random_grid[0]),
             np.nanmax(random_grid[0]), args.span)

    (p1, p2), var, coords = principal_directions(path, optimum)
    lo, hi = coords.min(0).values.numpy(), coords.max(0).values.numpy()
    pad = args.margin * (hi - lo)
    a_axis = np.linspace(min(lo[0] - pad[0], 0), max(hi[0] + pad[0], 0), args.grid)
    b_axis = np.linspace(min(lo[1] - pad[1], 0), max(hi[1] + pad[1], 0), args.grid)
    pca_grid = plane(sc, run, optimum, p1, p2, a_axis, b_axis)
    log.info("path PCA explained variance: first %.3f  second %.3f  first two %.3f", var[0], var[1], var[:2].sum())

    plots.save(line_figure(ts, rows), out, "line_start_to_optimum", log)
    plots.save(plane_figure(offsets, offsets, random_grid, args.max_off, "α [m per detector]", "β [m per detector]"),
               out, "random_slice", log)
    plots.save(plane_figure(a_axis, b_axis, pca_grid, args.max_off, f"PC1 [m], {var[0]:.1%} of path variance",
                            f"PC2 [m], {var[1]:.1%}", coords.numpy(), n_adam), out, "pca_slice", log)
    plots.save(variance_figure(var[:10].numpy()), out, "pca_explained_variance", log)
    logs.save(out, {"scheme": args.scheme, "chain": k, "t": ts.tolist(),
                    "line": {src: r.tolist() for src, r in rows.items()},
                    "random_slice": {"offsets": offsets.tolist(), "U_se_off": random_grid.tolist()},
                    "pca_slice": {"alpha": a_axis.tolist(), "beta": b_axis.tolist(), "U_se_off": pca_grid.tolist(),
                                  "explained_variance": var[:10].tolist(), "path_coords": coords[::10].tolist()}},
              args, log)


if __name__ == "__main__":
    main()

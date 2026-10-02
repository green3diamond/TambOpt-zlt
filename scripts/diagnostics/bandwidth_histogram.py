"""Histogram of the interpolated kernel's Gaussian widths h per species.

h = max(sigma, distance to the k-th nearest in-plane neighbour), computed on the
placed clouds of Stage 4's event pool (the same widths --pool main uses). Two
curves per species: share of points, and share of deposited energy, so it shows
both how often a point is widened and how much of the signal that affects.

Usage:
    python scripts/diagnostics/bandwidth_histogram.py [--n-events 1000] [--start 0]
"""
import argparse
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "kopt", os.path.join(_REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

from modules.constants import GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, N_PLANES  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

COLORS = ("#2a78d6", "#eb6834")        # points, deposit
INK, INK2 = "#0b0b0b", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(K.TRAINING_DATASET_FOLDER,
                                                  f"kernel_bw_k{K.KERNEL_INTERP_K}_hist.png"))
    args = ap.parse_args()

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=K.EAST_ENTRY, layer_east_dx=K.LAYER_EAST_DX,
                                n_planes=N_PLANES)
    pool = K.MainPool.__new__(K.MainPool)
    pool.meta = K._load_corpus_metadata(mountain, K.DUAL_SHOWER_CACHE_PATH, K.MAX_SHOWERS)
    pool.n, pool.bw = pool.meta.k_sp, None
    a, b = args.start, min(args.start + args.n_events, pool.n)
    sigma = float(K.SIGMA_SPATIAL)

    data = {}
    for s, name in enumerate(K.SPECIES_NAMES):
        h_all, e_all = [], []
        for lo in range(a, b, 250):
            hi = min(lo + 250, b)
            c = pool.clouds(s, lo, hi)
            ok = torch.isfinite(c).all(dim=(1, 2))
            c = c[ok].to(K.DEVICE)
            h = K.adaptive_bandwidths(c, K.KERNEL_INTERP_K, sigma)
            real = c[..., 3] > 0
            h_all.append(h[real].cpu().numpy())
            e_all.append(c[..., 3][real].cpu().numpy())
        h, e = np.concatenate(h_all), np.concatenate(e_all).astype(np.float64)
        wide = h > sigma * 1.001
        data[name] = (h, e)
        print(f"[{name}] points={h.size}  frac widened={wide.mean():.3f}  "
              f"deposit frac widened={e[wide].sum() / e.sum():.3f}  "
              f"median h={np.median(h):.1f} m  p90={np.percentile(h, 90):.1f}  "
              f"p99={np.percentile(h, 99):.1f}  "
              f"deposit-weighted median h="
              f"{h[np.argsort(h)][np.searchsorted(np.cumsum(e[np.argsort(h)]), 0.5 * e.sum())]:.1f} m")

    h_max = max(float(np.percentile(h, 99.9)) for h, _ in data.values())
    edges = np.logspace(np.log10(sigma * 0.9), np.log10(max(h_max, sigma * 10)), 60)
    fig, axes2 = plt.subplots(2, len(data), figsize=(4.6 * len(data), 7.4), sharey="row")
    axes, zoom = axes2[0], axes2[1]
    for ax, (name, (h, e)) in zip(np.atleast_1d(axes), data.items()):
        hc = np.clip(h, edges[0], edges[-1])
        pts, _ = np.histogram(hc, bins=edges)
        dep, _ = np.histogram(hc, bins=edges, weights=e)
        ax.stairs(pts / pts.sum(), edges, color=COLORS[0], lw=2, label="share of points")
        ax.stairs(dep / dep.sum(), edges, color=COLORS[1], lw=2, label="share of deposit")
        ax.axvline(sigma, color=INK2, ls="--", lw=1)
        ax.text(sigma * 1.08, 0.97, f"σ = {sigma:g} m", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=INK2)
        wide = h > sigma * 1.001
        ax.set_title(f"{name}: {wide.mean():.0%} of points, "
                     f"{e[wide].sum() / e.sum():.0%} of deposit widened",
                     fontsize=10, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Gaussian width h [m]", color=INK2)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("fraction per bin", color=INK2)
    axes[0].legend(fontsize=8, frameon=False)

    # Zoom: 0-30 m, linear. Fractions are of ALL points / deposit, so the part
    # of each curve above 30 m is simply off-panel (stated in the title).
    lin = np.arange(0.0, 30.5, 0.5)
    for ax, (name, (h, e)) in zip(np.atleast_1d(zoom), data.items()):
        pts, _ = np.histogram(h, bins=lin)
        dep, _ = np.histogram(h, bins=lin, weights=e)
        ax.stairs(pts / h.size, lin, color=COLORS[0], lw=2, label="share of points")
        ax.stairs(dep / e.sum(), lin, color=COLORS[1], lw=2, label="share of deposit")
        ax.axvline(sigma, color=INK2, ls="--", lw=1)
        above = h > 30.0
        ax.set_title(f"{name}, 0–30 m: {above.mean():.0%} of points, "
                     f"{e[above].sum() / e.sum():.0%} of deposit above 30 m",
                     fontsize=10, color=INK)
        ax.set_xlim(0, 30)
        ax.set_xlabel("Gaussian width h [m]", color=INK2)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    zoom[0].set_ylabel("fraction per 0.5 m bin", color=INK2)
    for ax in np.ravel(axes2):
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
    fig.suptitle(f"Interpolated-kernel widths, k = {K.KERNEL_INTERP_K}, "
                 f"{b - a} Stage-4 pool events", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()

"""Run 20 LHS repeats: the final layouts and their kernel validation (diagnostics/lhs, slurm/run20_quick_val.sh with
QTAG=lhs). Figure: one row per LHS seed (1-5); columns: the start, then the 8 5 deg-utility runs and the 4 penalty-only
runs; New U (drawn-count recon, no background, all read out, all energies) in each panel. Table: per run, New U for each
seed, mean and spread, the start's New U, and the construction metrics (seed mean), against the 100-detector grid.
Writes <run 20>/diagnostics/lhs/lhs_layouts.png and lhs_summary.md.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_lhs_summary.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402
from modules.optimize.cable import cable_length  # noqa: E402
from modules.optimize.construction import describe  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "lhs")
PATHS = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if "=" in l)
SEEDS = (1, 2, 3, 4, 5)
COLS = [("start", "LHS start", "#8a8a8a")] + [
    (f"sur5_{{s}}_{k}", f"5°: {v}", "#2a6fb0") for k, v in (
        ("e500_567", "E 5.00–5.67"), ("e567_633", "E 5.67–6.33"), ("e633_700", "E 6.33–7.00"), ("eall", "all E"),
        ("slope", "+ slope"), ("access", "+ access"), ("pads", "+ pads"), ("contour", "+ contour"))] + [
    (f"pen_{{s}}_{k}", f"penalty only: {k}", "#5b8c3a") for k in ("slope", "access", "pads", "contour")]


def key(pat, s):
    return f"lhs{s}_start" if pat == "start" else pat.format(s=f"lhs{s}")


num, n = {}, {}
for f in glob.glob(os.path.join(D, "val_deepsets_count_*.json")):
    for k, by in json.load(open(f))["scores"].items():
        r = by["none"]["0"]
        num[k] = num.get(k, 0.0) + r["n_showers"] * r["u_swgo_drawn"]
        n[k] = n.get(k, 0.0) + r["n_showers"]
U = {k: num[k] / n[k] for k in num}

verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surf = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256)
g = torch.load(os.path.join(C.RUN_LOCATION, "diagnostics", "triplets", "layouts", "grid_n99.pt"), map_location="cpu")
grid_xy = (g["x"].float(), g["y"].float())

fig, axs = plt.subplots(len(SEEDS), len(COLS), figsize=(2.6 * len(COLS), 3.5 * len(SEEDS)), constrained_layout=True)
geo = {}
for r, s in enumerate(SEEDS):
    for c, (pat, name, col) in enumerate(COLS):
        ax = axs[r, c]
        ax.add_collection(PolyCollection(tri[:, :, :2], facecolors="#f2f2f2", edgecolors="#d6d6d6", linewidths=0.3))
        ax.set_xlim(tri[..., 0].min() - 60, tri[..., 0].max() + 60)
        ax.set_ylim(tri[..., 1].min() - 60, tri[..., 1].max() + 60)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        k = key(pat, s)
        L = torch.load(PATHS[k], map_location="cpu")
        x, y = L["x"].float(), L["y"].float()
        with torch.no_grad():
            geo[k] = dict(describe(x, y, surf, m, grid_xy), cable_km=float(cable_length(x, y, surf)[0]) / 1e3)
        ax.scatter(x.numpy(), y.numpy(), s=6, color=col, lw=0)
        ax.text(0.03, 0.03, f"New U {U.get(k, float('nan')):.3f}", transform=ax.transAxes, fontsize=8)
        if r == 0:
            ax.set_title(name, fontsize=9)
        if c == 0:
            ax.text(-0.08, 0.5, f"LHS seed {s}", transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=10)
fig.suptitle("Run 20: surrogate optimisations from 5 Latin-hypercube starts (SGD 1,000 steps, Poisson trigger, no "
             f"background). New U = kernel validation, all energies; 100-detector grid {U.get('grid_n100', float('nan')):.3f}",
             fontsize=12)
fig.savefig(os.path.join(D, "lhs_layouts.png"), dpi=140, facecolor="white")

md = ["## Run 20: optimisations from 5 Latin-hypercube starts (surrogate, SGD 1,000 steps)", "",
      f"New U = kernel validation (drawn-count recon, no background, all read out, all energies); 100-detector grid "
      f"{U.get('grid_n100', float('nan')):.3f}. Start = the LHS starts' New U. Construction metrics: seed means.", "",
      "| Run | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean ± std | Slope [°] | Access [m] | Pad RMS [m] | "
      "Row RMS [m] | Cable [km] |", "|---" * 13 + "|"]
for pat, name, _ in COLS:
    us = np.array([U.get(key(pat, s), np.nan) for s in SEEDS])
    q = {t: np.mean([geo[key(pat, s)][t] for s in SEEDS]) for t in ("slope_mean", "access_m", "pad_rms", "row_rms",
                                                                     "cable_km")}
    md.append(f"| {name} | " + " | ".join(f"{u:.3f}" for u in us) + f" | {np.nanmean(us):.3f} ± {np.nanstd(us):.3f} | "
              f"{q['slope_mean']:.1f} | {q['access_m']:.0f} | {q['pad_rms']:.0f} | {q['row_rms']:.1f} | "
              f"{q['cable_km']:.2f} |")
open(os.path.join(D, "lhs_summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("[write]", os.path.join(D, "lhs_layouts.png"), os.path.join(D, "lhs_summary.md"))

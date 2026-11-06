"""Run 20: every surrogate-optimised layout in one figure (final positions, 1,000 SGD steps from the 100-detector grid).
Rows: objective (5 deg Poisson utility; the same at 1 us background; trigger only; penalty only). Columns: the energy-band
runs (no penalty) and the construction-penalty runs. Each panel: final detector positions on the detector region, and the
kernel validation New U (drawn-count recon, no background, all read out, all energies; start grid in the title). Runs
whose folder cannot be read are left empty (SKIP).
Writes <run 20>/diagnostics/r20b/all_layouts.png.

    source slurm/run20_guard.sh; [SKIP=trg_pads] python plots/analysis/run20/plot_all_layouts.py
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
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402

PRE = os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_")
DG = os.path.join(C.RUN_LOCATION, "diagnostics")
ROWS = [("sur5", "5° utility", "#2a6fb0"), ("sur5_1us", "5° utility, 1 µs bg", "#4f9bd9"),
        ("trg", "trigger only", "#e07b39"), ("pen", "penalty only", "#5b8c3a")]
COLS = [("e500_567", "E 5.00–5.67"), ("e567_633", "E 5.67–6.33"), ("e633_700", "E 6.33–7.00"), ("eall", "all E"),
        ("slope", "+ slope limit"), ("access", "+ access"), ("pads", "+ pads (20)"), ("contour", "+ contour rows (8)")]
SKIP = set(filter(None, os.environ.get("SKIP", "").split(",")))


def new_u(tag):
    num, n = {}, {}
    for f in glob.glob(os.path.join(DG, tag, "val_deepsets_count_*.json")):
        for k, by in json.load(open(f))["scores"].items():
            r = by["none"]["0"]
            num[k] = num.get(k, 0.0) + r["n_showers"] * r["u_swgo_drawn"]
            n[k] = n.get(k, 0.0) + r["n_showers"]
    return {k: num[k] / n[k] for k in num}


verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)
fig, axs = plt.subplots(len(ROWS), len(COLS), figsize=(3.3 * len(COLS), 4.3 * len(ROWS)), constrained_layout=True)
u0 = None
for r, (tag, rname, col) in enumerate(ROWS):
    U = new_u(tag)
    u0 = U.get("grid_n100", u0)
    for c, (key, cname) in enumerate(COLS):
        ax = axs[r, c]
        ax.add_collection(PolyCollection(tri[:, :, :2], facecolors="#f2f2f2", edgecolors="#d6d6d6", linewidths=0.3))
        ax.set_xlim(tri[..., 0].min() - 60, tri[..., 0].max() + 60)
        ax.set_ylim(tri[..., 1].min() - 60, tri[..., 1].max() + 60)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        name = f"{tag}_{key}"
        f = os.path.join(PRE + name, "opt_state.pt")
        if name in SKIP or not os.path.exists(f):
            ax.text(0.5, 0.5, "not run" if not os.path.exists(f) and name not in SKIP else "file locked",
                    transform=ax.transAxes, ha="center", va="center", color="#888888", fontsize=10)
        else:
            st = torch.load(f, map_location="cpu", weights_only=False)
            x, y = st["xy"]["x"].float().numpy(), st["xy"]["y"].float().numpy()
            ax.scatter(x, y, s=9, color=col, lw=0)
            u = U.get(name)
            ax.text(0.03, 0.03, f"New U {u:.3f}" if u is not None else "", transform=ax.transAxes, fontsize=9)
        if r == 0:
            ax.set_title(cname, fontsize=11)
for ax, (_, rname, _) in zip(axs[:, 0], ROWS):
    ax.text(-0.08, 0.5, rname, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=11)
fig.suptitle("Run 20: final layouts of the Stage 2 surrogate optimisations (SGD, 1,000 steps from the 100-detector grid, "
             f"Poisson trigger). New U = kernel validation, no background, all energies; start grid {u0:.3f}",
             fontsize=12)
out = os.path.join(DG, "r20b", "all_layouts.png")
fig.savefig(out, dpi=150, facecolor="white")
print("[write]", out)

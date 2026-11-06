"""Run 20: do the kernel and the Stage 2 surrogate optimisations converge to the same layouts? The 1,000-step SGD runs
from the 100-detector grid (5 deg Poisson utility, no background): sur5_<name> (surrogate) against ker5_<name> (kernel),
for the 4 energy bands and the 4 construction penalties. Detectors keep their index from the shared start, so the
layouts compare detector by detector:
  move     median distance each run moved its detectors from the start [m]
  apart    median distance between the two runs' positions of the same detector [m]
  cos      cosine between the two runs' displacement fields (start -> final, all detectors): 1 = same direction
  U        kernel validation New U, drawn-count recon, no background, all read out, all energies (quick tables)
Writes <run 20>/diagnostics/ker5/compare_sur_kernel.md and compare_sur_kernel.png (start, surrogate and kernel final
positions per run).

    source slurm/run20_guard.sh; python plots/analysis/run20/compare_sur_kernel.py
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
NAMES = [("e500_567", "E 5.00-5.67"), ("e567_633", "E 5.67-6.33"), ("e633_700", "E 6.33-7.00"), ("eall", "all E"),
         ("slope", "slope limit"), ("access", "access"), ("pads", "pads"), ("contour", "contour rows")]


def final(tag, k):
    st = torch.load(os.path.join(PRE + f"{tag}_{k}", "opt_state.pt"), map_location="cpu", weights_only=False)
    return st["xy"]["x"].float().numpy(), st["xy"]["y"].float().numpy(), st


def new_u(tag):
    """{layout: New U}, drawn-count recon, no background, all read out, all energies, chunks pooled."""
    num, n = {}, {}
    for f in glob.glob(os.path.join(DG, tag, "val_deepsets_count_*.json")):
        for k, by in json.load(open(f))["scores"].items():
            r = by["none"]["0"]
            num[k] = num.get(k, 0.0) + r["n_showers"] * r["u_swgo_drawn"]
            n[k] = n.get(k, 0.0) + r["n_showers"]
    return {k: num[k] / n[k] for k in num}


g = torch.load(os.path.join(DG, "ker5", "grid_n100.pt"), map_location="cpu")
e0, n0 = g["x"].numpy(), g["y"].numpy()
US, UK = new_u("sur5"), new_u("ker5")
verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)

md = ["## Run 20: surrogate vs kernel optimisation (5° Poisson utility, SGD lr 1 m, 1,000 steps, 100-detector grid start, "
      "no background)", "",
      "Move = median distance from the start; apart = median distance between the two runs' positions of the same "
      "detector; cos = cosine between the two displacement fields (1 = same direction, 0 = unrelated); New U = kernel "
      "validation (drawn-count recon, no background, all read out, all energies; start grid "
      f"{UK.get('grid_n100', float('nan')):.3f}).", "",
      "| Run | Move sur. [m] | Move ker. [m] | Apart [m] | cos | New U sur. | New U ker. |", "|---|---|---|---|---|---|---|"]
fig, axs = plt.subplots(2, 4, figsize=(19, 12), constrained_layout=True)
for ax, (k, name) in zip(axs.flat, NAMES):
    xs, ys, _ = final("sur5", k)
    xk, yk, stk = final("ker5", k)
    ds = np.c_[xs - e0, ys - n0]
    dk = np.c_[xk - e0, yk - n0]
    cos = float((ds * dk).sum() / (np.linalg.norm(ds) * np.linalg.norm(dk) + 1e-12))
    ms, mk = np.median(np.hypot(*ds.T)), np.median(np.hypot(*dk.T))
    ap = np.median(np.hypot(xs - xk, ys - yk))
    us, uk = US.get(f"sur5_{k}", float("nan")), UK.get(f"ker5_{k}", float("nan"))
    md.append(f"| {name} | {ms:.1f} | {mk:.1f} | {ap:.1f} | {cos:+.2f} | {us:.3f} | {uk:.3f} |")
    ax.add_collection(PolyCollection(tri[:, :, :2], facecolors="#f2f2f2", edgecolors="#d0d0d0", linewidths=0.3))
    ax.scatter(e0, n0, s=14, facecolors="none", edgecolors="#9a9a9a", lw=0.7, label="start")
    ax.scatter(xs, ys, s=12, color="#2a6fb0", lw=0, label="surrogate")
    ax.scatter(xk, yk, s=12, color="#e07b39", lw=0, label=f"kernel (step {stk['epoch']})")
    ax.set_xlim(tri[..., 0].min() - 60, tri[..., 0].max() + 60)
    ax.set_ylim(tri[..., 1].min() - 60, tri[..., 1].max() + 60)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: moved {ms:.0f} / {mk:.0f} m (sur. / ker.), apart {ap:.0f} m, cos {cos:+.2f}\n"
                 f"New U {us:.3f} / {uk:.3f}", fontsize=9)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Run 20: final layouts from the surrogate (blue) and the kernel (orange), same start and settings",
             fontsize=12)
out = os.path.join(DG, "ker5", "compare_sur_kernel")
fig.savefig(out + ".png", dpi=150, facecolor="white")
open(out + ".md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("[write]", out + ".png / .md")

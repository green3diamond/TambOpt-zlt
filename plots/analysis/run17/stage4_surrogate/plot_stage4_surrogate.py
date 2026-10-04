"""Run 17 Stage 4 through the surrogate (FNN + recon): layouts (top) and held-out kernel metrics
(bottom), as plot_det20.py does for run 15. Grid and center starts; current utility and two-tier
lambda 0.3 / 0. Scores: diagnostics/stage4_scores/scores.json (1,536 held-out kernel showers)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-plane (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(D, "diagnostics", "stage4_scores")
INK, GREY = "#1f2328", "#52514e"
COL = {"start": "#b9b7af", "current": "#2a78d6", "tt03": "#eb6834", "tt0": "#1baf7a"}
LAB = {"start": "start layout", "current": "current utility", "tt03": "two-tier λ = 0.3", "tt0": "two-tier λ = 0"}
TAG = {"current": "current", "tt03": "twotier_l0p3", "tt0": "twotier_l0"}

sc = json.load(open(os.path.join(S, "scores.json")))
runs = {(g, v): (sc[f"start_{g}"], sc[f"{TAG[v]}_{g}"],
                 os.path.join(D, f"test_v6_run_04_optimize_lbfgs_ensemble_{TAG[v]}_{g}", "layout_best.pt"))
        for g in ("grid", "center") for v in ("current", "tt03", "tt0")}
starts = {g: torch.load(os.path.join(S, f"start_{g}.pt"), map_location="cpu") for g in ("grid", "center")}

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)

keys = [(g, v) for g in ("grid", "center") for v in ("current", "tt03", "tt0")]
fig = plt.figure(figsize=(17, 8.6))
gs = fig.add_gridspec(2, len(keys), height_ratios=[1.35, 1], hspace=0.28, wspace=0.05)
for i, k in enumerate(keys):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(runs[k][2], map_location="cpu")
    s0 = starts[k[0]]
    a.scatter(mesh[:, 0], mesh[:, 1], s=2, color="#e8e6df", lw=0)
    a.scatter(np.asarray(s0["x"]), np.asarray(s0["y"]), s=9, facecolor="none", edgecolor=COL["start"], lw=0.6)
    a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=10, color=COL[k[1]], lw=0)
    o = runs[k][1]
    a.set_title(f"{k[0]} start · {LAB[k[1]]}\ngood {o['frac_good_5']:.1%} · angle {o['ang_med']:.1f}°",
                fontsize=10, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

metrics = [("frac_good_5", "showers reconstructed well (5°, 0.2)", 100, "%"),
           ("ang_med", "median angle error [deg]", 1, "°"),
           ("U_kernel", "current utility on the kernel", 1, "")]
sub = gs[1, :].subgridspec(1, len(metrics), wspace=0.3)
groups, variants, w = ["grid", "center"], ["start", "current", "tt03", "tt0"], 0.2
for m, (key, title, scale, unit) in enumerate(metrics):
    a = fig.add_subplot(sub[0, m])
    for gi, g in enumerate(groups):
        for vi, v in enumerate(variants):
            val = (runs[(g, "current")][0] if v == "start" else runs[(g, v)][1])[key] * scale
            xx = gi + (vi - (len(variants) - 1) / 2) * w
            a.bar(xx, val, width=w - 0.03, color=COL[v], label=LAB[v] if gi == 0 else None)
            a.text(xx, val, f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    a.set_xticks(range(len(groups))); a.set_xticklabels([f"{g} start" for g in groups])
    a.set_title(title, fontsize=10, color=INK)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if m == 0:
        handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("100 detectors (run 17, 45° plane): 10,000 Adam steps through the surrogate (FNN + recon), "
             "judged on 1,536 held-out kernel showers", fontsize=12, color=INK)
out = os.path.join(D, "stage4_surrogate_summary.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

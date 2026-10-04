"""det20_summary_2p5deg.png, 100-detector version: current utility and two-tier (lambda 0.3, 0)
optimized at 'good' = angle < 2.5 deg, grid and center starts, scored on held-out kernel showers."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

N = 100
DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5          # 2.5 or 5
TAG = "2p5deg" if DEG == 2.5 else "5deg"
GOOD = "frac_good_t" if DEG == 2.5 else "frac_good_5"
D = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
COL = {"start": "#b9b7af", "current": "#2a78d6", "tt03": "#eb6834", "tt0": "#1baf7a"}
LAB = {"start": "start layout", "current": "current utility", "tt03": "two-tier λ = 0.3", "tt0": "two-tier λ = 0"}
# the current-utility runs do not depend on the threshold: both versions take them from det100_2p5deg
SRC = [("det100_2p5deg", "current", "current"), (f"det100_{TAG}", "two_tier", "tt03"),
       (f"det100_{TAG}_lam0", "two_tier", "tt0")]

runs = {}
for folder, util, var in SRC:
    p = os.path.join(D, folder, "detector_count_scan.json")
    if not os.path.exists(p):
        continue
    for k, r in json.load(open(p)).items():
        if r["n"] == N and r["utility"] == util:
            runs[(r["start_scheme"], var)] = (r["start"], r["optimized"],
                                              os.path.join(D, folder, f"det_count_layout_{k}.pt"))

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)
starts = {}
for scheme in ("grid", "center"):
    torch.manual_seed(0)
    e0, n0 = sample_initial_layout_ne(mountain, n_units=N, scheme=scheme)
    starts[scheme] = project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())

keys = [k for k in [("grid", "current"), ("grid", "tt03"), ("grid", "tt0"),
                    ("center", "current"), ("center", "tt03"), ("center", "tt0")] if k in runs]
fig = plt.figure(figsize=(17, 8.2))
gs = fig.add_gridspec(2, max(len(keys), 1), height_ratios=[1.25, 1], hspace=0.28, wspace=0.05)
for i, k in enumerate(keys):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(runs[k][2], map_location="cpu")
    e0, n0 = starts[k[0]]
    a.scatter(mesh[:, 0], mesh[:, 1], s=3, color="#e8e6df", lw=0)
    a.scatter(e0.numpy(), n0.numpy(), s=12, facecolor="none", edgecolor=COL["start"], lw=0.7)
    a.scatter(L["x"].numpy(), L["y"].numpy(), s=14, color=COL[k[1]], lw=0)
    o = runs[k][1]
    a.set_title(f"{k[0]} start · {LAB[k[1]]}\ngood {o[GOOD]:.1%} · angle {o['ang_med']:.1f}°",
                fontsize=10, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

metrics = [(GOOD, f"showers reconstructed well ({DEG:g}°, 0.2)", 100, "%"),
           ("ang_med", "median angle error [deg]", 1, "°"),
           ("U_kernel", "current utility on the kernel", 1, "")]
sub = gs[1, :].subgridspec(1, len(metrics), wspace=0.3)
w = 0.2
handles, labels = [], []
for m, (key, title, scale, unit) in enumerate(metrics):
    a = fig.add_subplot(sub[0, m])
    for gi, g in enumerate(("grid", "center")):
        present = [v for v in ("current", "tt03", "tt0") if (g, v) in runs]
        if not present:
            continue
        present = ["start"] + present
        for vi, v in enumerate(present):
            val = (runs[(g, present[1])][0] if v == "start" else runs[(g, v)][1])[key] * scale
            xx = gi + (vi - (len(present) - 1) / 2) * w
            a.bar(xx, val, width=w - 0.03, color=COL[v], label=LAB[v])
            a.text(xx, val, f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    a.set_xticks([0, 1]); a.set_xticklabels(["grid start", "center start"])
    a.set_title(title, fontsize=10, color=INK)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for h, l in zip(*a.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
order = [LAB[v] for v in ("start", "current", "tt03", "tt0") if LAB[v] in labels]
handles = [handles[labels.index(l)] for l in order]
fig.legend(handles, order, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"100 detectors (run 15), 'good' = angle < {DEG:g}° and |Δlog10 E| < 0.2: two-tier optimized at {DEG:g}°",
             fontsize=12, color=INK)
out = os.path.join(D, f"det100_summary_{TAG}.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}  ({len(keys)} runs)")

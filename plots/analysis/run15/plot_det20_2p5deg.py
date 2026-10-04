"""det20_summary.png, 2.5-degree version: every two-tier run optimized with 'good' = angle < 2.5 deg,
and the current-utility layouts re-scored at 2.5 deg; all judged on held-out kernel showers."""
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

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
COL = {"start": "#b9b7af", "current": "#2a78d6", "tt03": "#eb6834", "tt0": "#1baf7a"}
LAB = {"start": "start layout", "current": "current utility", "tt03": "two-tier λ = 0.3", "tt0": "two-tier λ = 0"}

tt03 = json.load(open(os.path.join(D, "det_count_2p5deg", "detector_count_scan.json")))
cur = json.load(open(os.path.join(D, "det_count_2p5deg", "current_utility_scored_2p5deg.json")))
p0 = os.path.join(D, "det_count_2p5deg_lam0", "detector_count_scan.json")
tt0 = json.load(open(p0)) if os.path.exists(p0) else {}

runs = {}   # (start, variant) -> (start metrics, optimized metrics, layout path)
for s in ("grid", "center"):
    st = tt03[f"n20_{s}_two_tier"]["start"]
    cur_path = (os.path.join(D, "det_count_layout_n20_grid_current.pt") if s == "grid"
                else os.path.join(D, "det_count_center", "det_count_layout_n20_center_current.pt"))
    runs[(s, "current")] = (st, cur[f"{s}_current"], cur_path)
    runs[(s, "tt03")] = (st, tt03[f"n20_{s}_two_tier"]["optimized"],
                         os.path.join(D, "det_count_2p5deg", f"det_count_layout_n20_{s}_two_tier.pt"))
    if f"n20_{s}_two_tier" in tt0:
        runs[(s, "tt0")] = (st, tt0[f"n20_{s}_two_tier"]["optimized"],
                            os.path.join(D, "det_count_2p5deg_lam0", f"det_count_layout_n20_{s}_two_tier.pt"))

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)
starts = {}
for scheme in ("grid", "center"):
    torch.manual_seed(0)
    e0, n0 = sample_initial_layout_ne(mountain, n_units=20, scheme=scheme)
    starts[scheme] = project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())

keys = [k for k in [("grid", "current"), ("grid", "tt03"), ("grid", "tt0"),
                    ("center", "current"), ("center", "tt03"), ("center", "tt0")] if k in runs]
fig = plt.figure(figsize=(17, 8.2))
gs = fig.add_gridspec(2, len(keys), height_ratios=[1.25, 1], hspace=0.28, wspace=0.05)
for i, k in enumerate(keys):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(runs[k][2], map_location="cpu")
    e0, n0 = starts[k[0]]
    a.scatter(mesh[:, 0], mesh[:, 1], s=3, color="#e8e6df", lw=0)
    a.scatter(e0.numpy(), n0.numpy(), s=30, facecolor="none", edgecolor=COL["start"], lw=1.0)
    a.scatter(L["x"].numpy(), L["y"].numpy(), s=34, color=COL[k[1]], lw=0)
    o = runs[k][1]
    a.set_title(f"{k[0]} start · {LAB[k[1]]}\ngood {o['frac_good_t']:.1%} · angle {o['ang_med']:.1f}°",
                fontsize=10, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

metrics = [("frac_good_t", "showers reconstructed well (2.5°, 0.2)", 100, "%"),
           ("ang_med", "median angle error [deg]", 1, "°"),
           ("U_kernel", "current utility on the kernel", 1, "")]
sub = gs[1, :].subgridspec(1, len(metrics), wspace=0.3)
w = 0.2
for m, (key, title, scale, unit) in enumerate(metrics):
    a = fig.add_subplot(sub[0, m])
    for gi, g in enumerate(("grid", "center")):
        present = ["start"] + [v for v in ("current", "tt03", "tt0") if (g, v) in runs]
        for vi, v in enumerate(present):
            val = (runs[(g, "current")][0] if v == "start" else runs[(g, v)][1])[key] * scale
            xx = gi + (vi - (len(present) - 1) / 2) * w
            a.bar(xx, val, width=w - 0.03, color=COL[v], label=LAB[v] if gi == 0 else None)
            a.text(xx, val, f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    a.set_xticks([0, 1]); a.set_xticklabels(["grid start", "center start"])
    a.set_title(title, fontsize=10, color=INK)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if m == 0:
        handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("20 detectors (run 15), 'good' = angle < 2.5° and |Δlog10 E| < 0.2: two-tier optimized at 2.5°, "
             "current-utility layouts re-scored", fontsize=12, color=INK)
out = os.path.join(D, "det20_summary_2p5deg.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

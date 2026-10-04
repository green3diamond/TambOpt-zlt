"""Detector-count scan summary: layouts (one row per detector count) and held-out kernel
metrics. Collects every finished run: grid start (current, two-tier lambda 0.3,
two-tier lambda 0) and center start (current, two-tier lambda 0.3)."""
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
SOURCES = [  # (folder, start scheme, utility key in JSON, variant)
    ("", "grid", "current", "current"), ("", "grid", "two_tier", "tt03"),
    ("det_count_lam0", "grid", "two_tier", "tt0"),
    ("det_count_center", "center", "current", "current"), ("det_count_center", "center", "two_tier", "tt03"),
]

runs = {}   # (n, start, variant) -> (start metrics, optimized metrics, layout path)
for folder, start, util, var in SOURCES:
    p = os.path.join(D, folder, "detector_count_scan.json")
    if not os.path.exists(p):
        continue
    res = json.load(open(p))
    for k, r in res.items():
        if r["start_scheme"] == start and r["utility"] == util:
            runs[(r["n"], start, var)] = (r["start"], r["optimized"],
                                          os.path.join(D, folder, f"det_count_layout_{k}.pt"))
counts = sorted({k[0] for k in runs})
cols = [(s, v) for s, v in [("grid", "current"), ("grid", "tt03"), ("grid", "tt0"),
                             ("center", "current"), ("center", "tt03")] if any((n, s, v) in runs for n in counts)]

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)


def start_layout(n, scheme):
    torch.manual_seed(0)
    e0, n0 = sample_initial_layout_ne(mountain, n_units=n, scheme=scheme)
    return project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())


fig = plt.figure(figsize=(3.4 * len(cols), 6.2 * len(counts) + 5))
gs = fig.add_gridspec(len(counts) + 1, len(cols), height_ratios=[1.25] * len(counts) + [1], hspace=0.3, wspace=0.1)
for i, n in enumerate(counts):
    for j, (s, v) in enumerate(cols):
        a = fig.add_subplot(gs[i, j])
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if (n, s, v) not in runs:
            a.set_title(f"{n} det · {s} start · {LAB[v]}\n(not run)", fontsize=9, color=GREY)
            continue
        st, o, path = runs[(n, s, v)]
        L = torch.load(path, map_location="cpu")
        e0, n0 = start_layout(n, s)
        a.scatter(mesh[:, 0], mesh[:, 1], s=3, color="#e8e6df", lw=0)
        a.scatter(e0.numpy(), n0.numpy(), s=22, facecolor="none", edgecolor=COL["start"], lw=0.9)
        a.scatter(L["x"].numpy(), L["y"].numpy(), s=26, color=COL[v], lw=0)
        a.set_title(f"{n} det · {s} start · {LAB[v]}\ngood {o['frac_good_5']:.1%} · angle {o['ang_med']:.1f}°",
                    fontsize=9, color=INK)
        a.set_aspect("equal")

metrics = [("frac_good_5", "showers reconstructed well (5°, 0.2)", 100, "%"),
           ("ang_med", "median angle error [deg]", 1, "°"),
           ("U_kernel", "current utility on the kernel", 1, "")]
groups = [(n, s) for n in counts for s in ("grid", "center") if any((n, s, v) in runs for v in COL)]
sub = gs[len(counts), :].subgridspec(1, len(metrics), wspace=0.25)
w = 0.2
for m, (key, title, scale, unit) in enumerate(metrics):
    a = fig.add_subplot(sub[0, m])
    for gi, (n, s) in enumerate(groups):
        present = ["start"] + [v for v in ("current", "tt03", "tt0") if (n, s, v) in runs]
        first = next(v for v in present[1:])
        for vi, v in enumerate(present):
            val = (runs[(n, s, first)][0] if v == "start" else runs[(n, s, v)][1])[key] * scale
            xx = gi + (vi - (len(present) - 1) / 2) * w
            a.bar(xx, val, width=w - 0.03, color=COL[v], label=LAB[v])
            a.text(xx, val, f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=7, color=GREY)
    a.set_xticks(range(len(groups))); a.set_xticklabels([f"{n} det\n{s} start" for n, s in groups], fontsize=9)
    a.set_title(title, fontsize=10, color=INK)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
handles, labels = [], []
for h, l in zip(*a.get_legend_handles_labels()):
    if l not in labels:
        handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Detector-count scan (run 15): 3,000 Adam steps through the surrogate, judged on held-out kernel "
             "showers", fontsize=12, color=INK, y=0.995)
out = os.path.join(D, "detector_count_scan_summary.png")
fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
print(f"[write] {out}  ({len(runs)} runs)")

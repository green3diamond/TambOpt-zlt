"""Core test figure: 100-detector layouts (top) and held-out kernel scores (bottom):
share of showers good at 5 deg and at 2.5 deg (|dlog10 E| < 0.2), and median angle error."""
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
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
res = json.load(open(os.path.join(D, "core_test_scores.json")))
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if l.strip())
names = [n for n in paths if n in res]
KIND = {"grid": "#b9b7af"}
for n in names:
    KIND.setdefault(n, "#eb6834" if n.startswith("core_r") else "#eda100" if n.startswith("core") else "#2a78d6")
LABEL = {"grid": "grid", "core_r100": "core r=100 m", "core_r200": "core r=200 m", "core_r300": "core r=300 m",
         "core_r500": "core r=500 m", "core300_60_spread40": "60 in 300 m core\n+ 40 spread",
         "core300_40_spread60": "40 in 300 m core\n+ 60 spread", "core300_60_rim40": "60 in 300 m core\n+ 40 on rim",
         "core200_50_spread50": "50 in 200 m core\n+ 50 spread", "opt_stage4_grid": "optimized\n(Stage 4, grid)",
         "opt_stage4_center": "optimized\n(Stage 4, center)", "opt_kernel_grid": "optimized\n(kernel, grid)",
         "opt_kernel_center": "optimized\n(kernel, center)"}

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)

fig = plt.figure(figsize=(1.45 * len(names) + 1, 10))
gs = fig.add_gridspec(4, len(names), height_ratios=[1.3, 1, 1, 1], hspace=0.45, wspace=0.08)
for i, n in enumerate(names):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(paths[n], map_location="cpu")
    a.scatter(mesh[:, 0], mesh[:, 1], s=1.5, color="#e8e6df", lw=0)
    a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=5, color=KIND[n], lw=0)
    a.set_title(LABEL.get(n, n), fontsize=8, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

rows = [("frac_good_5", "showers good at 5°, |Δlog10 E| < 0.2", 100, "%"),
        ("frac_good_t", "showers good at 2.5°, |Δlog10 E| < 0.2", 100, "%"),
        ("ang_med", "median angle error [deg]", 1, "°")]
for r, (key, title, scale, unit) in enumerate(rows):
    a = fig.add_subplot(gs[r + 1, :])
    vals = [res[n][key] * scale for n in names]
    x = np.arange(len(names))
    a.bar(x, vals, color=[KIND[n] for n in names], width=0.7)
    for xx, v in zip(x, vals):
        a.text(xx, v, f"{v:.1f}{unit}", ha="center", va="bottom", fontsize=8, color=GREY)
    ref = res["grid"][key] * scale
    a.axhline(ref, color="#b9b7af", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels([LABEL.get(n, n).replace("\n", " ") for n in names], fontsize=7.5,
                                       rotation=25, ha="right")
    a.set_title(title + "   (dashed: grid)", fontsize=10, color=INK, loc="left")
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    if r < 2:
        a.set_xticklabels([])
fig.suptitle("Core test, 100 detectors (run 15): hand-made cores and hybrids vs grid and optimized layouts, "
             "held-out kernel showers", fontsize=12, color=INK)
out = os.path.join(D, "core_test.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

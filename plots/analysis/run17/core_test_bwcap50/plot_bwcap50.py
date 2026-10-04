"""Core-test layouts (grid, 8 hand-made, 6 kernel-optimized) scored on the kernel with the widths
uncapped (../core_test/core_test_scores.json) and capped at 50 m (core_test_scores_cap50.json):
current utility, share of showers good at 5° and 2.5°, median angle error. Same 1,536 held-out showers, same kernel-trained recon."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-bwcap50 (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(os.path.dirname(D), "core_test")
INK, GREY = "#1f2328", "#52514e"
u0 = json.load(open(os.path.join(CT, "core_test_scores.json")))
u1 = json.load(open(os.path.join(D, "core_test_scores_cap50.json")))
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(CT, "layouts.txt")) if l.strip())
names = [n for n in paths if n in u0 and n in u1]
KIND = {n: "#b9b7af" if n == "grid" else "#eb6834" if n.startswith("core_r") else "#eda100" if n.startswith("core")
        else "#1baf7a" for n in names}
LABEL = {"grid": "grid", "core_r100": "core r=100 m", "core_r200": "core r=200 m", "core_r300": "core r=300 m",
         "core_r500": "core r=500 m", "core300_60_spread40": "60 in 300 m core\n+ 40 spread",
         "core300_40_spread60": "40 in 300 m core\n+ 60 spread", "core300_60_rim40": "60 in 300 m core\n+ 40 on rim",
         "core200_50_spread50": "50 in 200 m core\n+ 50 spread"}
for tag, txt in (("current", "current"), ("twotier_l0", "two-tier λ=0"), ("twotier_l0p3", "two-tier λ=0.3")):
    for start in ("grid", "center"):
        LABEL[f"kern_{tag}_{start}"] = f"kernel-optimized,\n{txt} ({start})"

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)

ROWS = [("U_kernel", "current utility on the kernel", 1, ""),
        ("frac_good_5", "showers good at 5°, |Δlog10 E| < 0.2", 100, "%"),
        ("frac_good_t", "showers good at 2.5°, |Δlog10 E| < 0.2", 100, "%"),
        ("ang_med", "median angle error [deg]", 1, "°")]
fig = plt.figure(figsize=(1.45 * len(names) + 1, 15))
gs = fig.add_gridspec(1 + len(ROWS), len(names), height_ratios=[1.3] + [1.15] * len(ROWS), hspace=0.42, wspace=0.08)
for i, n in enumerate(names):
    a = fig.add_subplot(gs[0, i])
    L = torch.load(paths[n], map_location="cpu")
    a.scatter(mesh[:, 0], mesh[:, 1], s=1.5, color="#e8e6df", lw=0)
    a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=5, color=KIND[n], lw=0)
    a.set_title(LABEL.get(n, n), fontsize=8, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)

x = np.arange(len(names)); w = 0.38
for r, (key, title, scale, unit) in enumerate(ROWS):
    a = fig.add_subplot(gs[r + 1, :])
    v0 = [u0[n][key] * scale for n in names]
    v1 = [u1[n][key] * scale for n in names]
    a.bar(x - w / 2, v0, width=w, color=[KIND[n] for n in names], alpha=0.35,
          label="widths uncapped (dotted: grid)")
    a.bar(x + w / 2, v1, width=w, color=[KIND[n] for n in names], label="widths capped at 50 m (dashed: grid)")
    for xx, a0, a1 in zip(x, v0, v1):
        a.text(xx - w / 2, a0, f"{a0:.1f}", ha="center", va="bottom", fontsize=6.5, color=GREY)
        a.text(xx + w / 2, a1, f"{a1:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    a.axhline(u0["grid"][key] * scale, color="#b9b7af", ls=":", lw=1)
    a.axhline(u1["grid"][key] * scale, color="#b9b7af", ls="--", lw=1)
    a.set_ylim(0, max(v0 + v1) * 1.18)
    a.set_xticks(x)
    a.set_xticklabels([LABEL.get(n, n).replace("\n", " ") for n in names] if r == len(ROWS) - 1 else [],
                      fontsize=7.5, rotation=25, ha="right")
    a.set_title(title + "   (pale: uncapped · solid: capped at 50 m)", fontsize=10, color=INK, loc="left")
    a.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("Core test, 100 detectors (run 17, 45° plane): kernel widths uncapped vs capped at 50 m · "
             "1,536 held-out showers, recon trained on (uncapped) kernel responses", fontsize=12, color=INK)
out = os.path.join(D, "core_test_bwcap50.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

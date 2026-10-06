"""Run 18, 'good' = angle only: core-test layouts (bars), detector-count grids and growing disks (curves)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY, BLUE, ORANGE, GREEN, GOLD, SAND = "#1f2328", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b9b7af"
s5 = json.load(open(os.path.join(D, "scores_angleonly_5deg.json")))
s25 = json.load(open(os.path.join(D, "scores_angleonly_2.5deg.json")))
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if l.strip())
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
span = float(max(np.ptp(mesh[:, 0]), np.ptp(mesh[:, 1])))
g5, g25, med = (lambda k: 100 * s5[k]["frac_good_t"]), (lambda k: 100 * s25[k]["frac_good_t"]), (lambda k: s5[k]["ang_med"])

fig = plt.figure(figsize=(20, 15))
gsp = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.18)

# 1. core-test layouts
names = [n for n in paths if not n.startswith(("grid_n", "disk_r"))]
col = {n: SAND if n == "grid" else ORANGE if n.startswith("core_r") else GOLD if n.startswith("core")
       else GREEN for n in names}
a = fig.add_subplot(gsp[0, :])
x = np.arange(len(names))
a.bar(x - 0.2, [g5(n) for n in names], 0.38, color=[col[n] for n in names], label="within 5°")
a.bar(x + 0.2, [g25(n) for n in names], 0.38, color=[col[n] for n in names], alpha=0.5, label="within 2.5°")
for xx, n in zip(x, names):
    a.text(xx - 0.2, g5(n), f"{g5(n):.0f}", ha="center", va="bottom", fontsize=8, color=GREY)
    a.text(xx + 0.2, g25(n), f"{g25(n):.0f}", ha="center", va="bottom", fontsize=8, color=GREY)
    a.text(xx, -9, f"{med(n):.1f}°", ha="center", fontsize=8, color=INK)
a.axhline(g5("grid"), color=SAND, ls="--", lw=1)
a.set_xticks(x)
a.set_xticklabels([n.replace("kern_", "opt: ").replace("_", " ") for n in names], rotation=30, ha="right", fontsize=8.5)
a.set_ylim(-12, 105)
a.set_ylabel("showers within the angle [%]")
a.set_title("Core-test layouts, 'good' = angle only (solid: 5°, faded: 2.5°; number under each bar: median angle)",
            loc="left", fontsize=11.5, color=INK)

# 2. detector-count grids
ns = [4, 5, 10, 20, 25, 50, 100]
keys = [f"grid_n{n}" if n != 100 else "grid" for n in ns]
a = fig.add_subplot(gsp[1, 0])
a.plot(ns, [g5(k) for k in keys], "o-", color=BLUE, lw=2, label="within 5°")
a.plot(ns, [g25(k) for k in keys], "o-", color=ORANGE, lw=2, label="within 2.5°")
a2 = a.twinx()
a2.plot(ns, [med(k) for k in keys], "s:", color=GREY, lw=1.5)
a2.set_ylabel("median angle [deg] (dotted)", color=GREY); a2.set_yscale("log")
a.set(xscale="log", xlabel="detectors in the array (grid over the plane)", ylabel="showers within the angle [%]",
      title="Number of detectors (grid)")
a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns]); a.legend(fontsize=9, frameon=False)

# 3. growing disks
rs = sorted(int(n[6:]) for n in paths if n.startswith("disk_r"))
dk = [f"disk_r{r}" for r in rs]
a = fig.add_subplot(gsp[1, 1])
a.plot(rs, [g5(k) for k in dk], "o-", color=BLUE, lw=2, label="within 5°")
a.plot(rs, [g25(k) for k in dk], "o-", color=ORANGE, lw=2, label="within 2.5°")
a.axhline(g5("grid"), color=BLUE, ls="--", lw=1, label="grid, 5°")
a.axhline(g25("grid"), color=ORANGE, ls="--", lw=1, label="grid, 2.5°")
for v, lab in ((span / 2, "half the plane"), (span, "plane size"), (2 * span, "2 × plane")):
    a.axvline(v, color=GREY, ls=":", lw=1); a.text(v, 102, lab, fontsize=8, color=GREY, ha="center")
a2 = a.twinx()
a2.plot(rs, [med(k) for k in dk], "s:", color=GREY, lw=1.5)
a2.set_ylabel("median angle [deg] (dotted)", color=GREY); a2.set_yscale("log")
a.set(xscale="log", xlabel="disk radius [m] (100 detectors, sunflower, at the plane centre)",
      ylabel="showers within the angle [%]", title="Circular arrays of growing radius", ylim=(0, 108))
a.legend(fontsize=9, frameon=False, loc="center left")

# 4. disk layouts
sub = gsp[2, :].subgridspec(1, len(rs), wspace=0.05)
for i, (r, k) in enumerate(zip(rs, dk)):
    a = fig.add_subplot(sub[i])
    L = torch.load(paths[k], map_location="cpu")
    a.scatter(mesh[:, 0], mesh[:, 1], s=0.5, color="#e8e6df", lw=0)
    a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=6, color=BLUE, lw=0)
    a.set_title(f"r = {r / 1000:g} km\n{g5(k):.0f}% · {med(k):.1f}°", fontsize=8.5, color=INK)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
fig.suptitle("Run 18 (6 km plane facing South, shower-frame kernel): showers reconstructed within 5° / 2.5°, "
             "angle only (no energy condition) · 1,536 held-out showers\nDisks larger than the plane: detectors "
             "outside snap to the nearest edge face", fontsize=13, color=INK)
out = os.path.join(D, "angleonly_scores.png")
fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

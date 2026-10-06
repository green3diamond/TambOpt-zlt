"""Core test with 20 / 50 / 100 detectors (run 18), 'good' = angle only (5 deg solid, 2.5 deg faded)."""
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
A = os.path.join(os.path.dirname(D), "angleonly")
INK, GREY, SAND, ORANGE, GOLD, BLUE = "#1f2328", "#52514e", "#b9b7af", "#eb6834", "#eda100", "#2a78d6"
s5 = {**json.load(open(os.path.join(A, "scores_angleonly_5deg.json"))), **json.load(open(os.path.join(D, "scores_5deg.json")))}
s25 = {**json.load(open(os.path.join(A, "scores_angleonly_2.5deg.json"))), **json.load(open(os.path.join(D, "scores_2.5deg.json")))}
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if l.strip())
paths.update(dict(l.strip().split("=", 1) for l in open(os.path.join(A, "layouts.txt")) if l.strip()))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
BASE = ["grid", "core_r100", "core_r200", "core_r300", "core_r500", "core300_60_spread40", "core300_40_spread60",
        "core300_60_rim40", "core200_50_spread50"]
LAB = {"grid": "grid", "core_r100": "core 100 m", "core_r200": "core 200 m", "core_r300": "core 300 m",
       "core_r500": "core 500 m", "core300_60_spread40": "60% in 300 m\n+ 40% spread",
       "core300_40_spread60": "40% in 300 m\n+ 60% spread", "core300_60_rim40": "60% in 300 m\n+ 40% rim",
       "core200_50_spread50": "50% in 200 m\n+ 50% spread"}
COL = {b: SAND if b == "grid" else ORANGE if b.startswith("core_r") else GOLD for b in BASE}
key = lambda b, N: b if N == 100 else f"{b}_n{N}"                     # noqa: E731
dkey = lambda r, N: f"disk_r{r}" if N == 100 else f"disk_r{r}_n{N}"   # noqa: E731
g5 = lambda k: 100 * s5[k]["frac_good_t"]                           # noqa: E731
g25 = lambda k: 100 * s25[k]["frac_good_t"]                         # noqa: E731

NS = (20, 50, 100)
fig = plt.figure(figsize=(21, 17))
gsp = fig.add_gridspec(4, 3, height_ratios=[0.8, 1.1, 0.8, 1.1], hspace=0.55, wspace=0.16)
for c, N in enumerate(NS):
    sub = gsp[0, c].subgridspec(1, len(BASE), wspace=0.04)
    for i, b in enumerate(BASE):
        a = fig.add_subplot(sub[i]); L = torch.load(paths[key(b, N)], map_location="cpu")
        a.scatter(mesh[:, 0], mesh[:, 1], s=0.3, color="#e8e6df", lw=0)
        a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=3, color=COL[b], lw=0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    a = fig.add_subplot(gsp[1, c]); x = np.arange(len(BASE))
    v5 = [g5(key(b, N)) for b in BASE]; v25 = [g25(key(b, N)) for b in BASE]
    a.bar(x - 0.2, v5, 0.38, color=[COL[b] for b in BASE]); a.bar(x + 0.2, v25, 0.38, color=[COL[b] for b in BASE], alpha=0.45)
    for xx, v, md in zip(x, v5, [s5[key(b, N)]["ang_med"] for b in BASE]):
        a.text(xx - 0.2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=GREY)
        a.text(xx, -9, f"{md:.1f}°", ha="center", fontsize=7.5, color=INK)
    a.axhline(v5[0], color=SAND, ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels([LAB[b] for b in BASE], rotation=40, ha="right", fontsize=8)
    a.set_ylim(-12, 100); a.set_title(f"{N} detectors: grid, cores, hybrids", loc="left", fontsize=11.5, color=INK)
    if c == 0:
        a.set_ylabel("showers within the angle [%]\n(solid 5°, faded 2.5°; median angle below)")
    rs = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
    sub = gsp[2, c].subgridspec(1, 7, wspace=0.04)
    for i, r in enumerate((300, 1000, 2000, 3000, 4500, 6000, 12000)):
        a2 = fig.add_subplot(sub[i]); L = torch.load(paths[dkey(r, N)], map_location="cpu")
        a2.scatter(mesh[:, 0], mesh[:, 1], s=0.3, color="#e8e6df", lw=0)
        a2.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=3, color=BLUE, lw=0)
        a2.set_title(f"{r / 1000:g} km", fontsize=8); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
        for sp in a2.spines.values():
            sp.set_visible(False)
    a = fig.add_subplot(gsp[3, c])
    a.plot(rs, [g5(dkey(r, N)) for r in rs], "o-", color=BLUE, lw=2, label="disk, within 5°")
    a.plot(rs, [g25(dkey(r, N)) for r in rs], "o-", color=ORANGE, lw=2, label="disk, within 2.5°")
    a.axhline(v5[0], color=BLUE, ls="--", lw=1, label="grid, 5°"); a.axhline(v25[0], color=ORANGE, ls="--", lw=1, label="grid, 2.5°")
    best = max(rs, key=lambda r: g5(dkey(r, N)))
    a.set(xscale="log", ylim=(0, 100), xlabel="disk radius [m]", title=f"{N} detectors: disks of growing radius "
          f"(best {best / 1000:g} km: {g5(dkey(best, N)):.0f}%)")
    a.grid(alpha=0.25, lw=0.5)
    if c == 0:
        a.set_ylabel("showers within the angle [%]"); a.legend(fontsize=8, frameon=False)
fig.suptitle("Run 18 core test with 20, 50 and 100 detectors · 'good' = angle only · 1,536 held-out showers, shower-frame "
             "kernel + run 18 recon\nHybrids keep the 100-detector proportions; disks larger than the plane snap to its edges",
             fontsize=13, color=INK)
out = os.path.join(D, "core_test_ndet.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"[write] {out}")

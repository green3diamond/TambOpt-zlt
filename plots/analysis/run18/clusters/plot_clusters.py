"""Clustered layouts vs grid (run 18): share within 1 / 2.5 / 5 deg (angle only) vs cluster radius."""
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
S = {t: json.load(open(os.path.join(D, f"scores_pad_{t}deg.json"))) for t in ("1", "2.5", "5")}
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if l.strip())
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
SPLITS = {20: ((4, 5), (5, 4), (10, 2)), 50: ((10, 5), (25, 2)), 100: ((20, 5), (25, 4))}
RADII = (50, 100, 200, 300, 500)
COLS = ["#2a78d6", "#eb6834", "#1baf7a"]
fig = plt.figure(figsize=(20, 16))
gsp = fig.add_gridspec(4, 3, height_ratios=[0.9, 1, 1, 1], hspace=0.45, wspace=0.2)
for c, N in enumerate((20, 50, 100)):
    sub = gsp[0, c].subgridspec(1, len(SPLITS[N]) + 1, wspace=0.05)
    shows = [f"grid_n{N}"] + [f"clu{K}x{M}_r200_n{N}" for K, M in SPLITS[N]]
    for i, k in enumerate(shows):
        a = fig.add_subplot(sub[i]); L = torch.load(paths[k], map_location="cpu")
        a.scatter(mesh[:, 0], mesh[:, 1], s=0.3, color="#e8e6df", lw=0)
        a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=5, color="#2a78d6", lw=0)
        a.set_title("grid" if i == 0 else k.split("_")[0].replace("clu", "") + " (r 200 m)", fontsize=9)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for row, t in enumerate(("1", "2.5", "5")):
        a = fig.add_subplot(gsp[row + 1, c])
        for j, (K, M) in enumerate(SPLITS[N]):
            a.plot(RADII, [100 * S[t][f"clu{K}x{M}_r{r}_n{N}"]["frac_good_t"] for r in RADII], "o-", color=COLS[j],
                   lw=2, label=f"{K} clusters × {M}")
        a.axhline(100 * S[t][f"grid_n{N}"]["frac_good_t"], color="#52514e", ls="--", lw=1.5, label="grid")
        a.set(xscale="log", xlabel="cluster radius [m]", ylabel=f"within {t}° [%]",
              title=f"{N} detectors · within {t}°")
        a.set_xticks(RADII); a.set_xticklabels([str(r) for r in RADII]); a.set_ylim(bottom=0)
        a.grid(alpha=0.25, lw=0.5)
        if row == 0:
            a.legend(fontsize=8, frameon=False)
fig.suptitle("Run 18: clustered grids (clusters on an even grid, sunflower disk per cluster) vs plain grid · angle only · "
             "1,536 held-out showers · N < 100: empty recon slots set to 0", fontsize=13)
out = os.path.join(D, "clusters.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("[write]", out)

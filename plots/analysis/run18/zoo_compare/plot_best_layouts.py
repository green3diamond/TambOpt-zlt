"""Detector positions of the best layout per family and detector count (the layouts behind compare_best.png).

One figure per angle (1 and 5 deg): rows = 20 / 50 / 100 detectors, columns = families. Each cell shows the zoo
recon's best layout of that family (blue) on the plane; when the old recon's best is a different layout it is
drawn too (grey rings). Titles give the layout names and the shares within the angle for both recons.
Writes best_layouts_1deg.png and best_layouts_5deg.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(Z)
FAMS = ["grid", "disk", "cluster", "core", "hybrid", "optimized"]
INK, GREY, BLUE = "#1f2328", "#8a8f98", "#2a78d6"

paths = {}
for f in (f"{D}/core_test_ndet/layouts.txt", f"{D}/clusters/layouts.txt", f"{Z}/layouts_angleonly.txt"):
    for line in open(f):
        if line.strip():
            k, v = line.strip().split("=", 1)
            paths.setdefault(k, v)

m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
S = json.load(open(f"{Z}/compare_summary.json"))


def xy(name):
    L = torch.load(paths[name], map_location="cpu")
    return np.asarray(L["x"], dtype=float) / 1e3, np.asarray(L["y"], dtype=float) / 1e3


for th in ("1", "5"):
    fig, axs = plt.subplots(3, len(FAMS), figsize=(4.2 * len(FAMS), 13.2))
    for i, n in enumerate((20, 50, 100)):
        for j, f in enumerate(FAMS):
            ax = axs[i, j]
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            c = S.get(f"{f}|{n}|{th}")
            if not c:
                ax.text(0.5, 0.5, f"{f}\nnot tested at {n}", ha="center", va="center", color=GREY,
                        transform=ax.transAxes)
                continue
            ax.scatter(mesh[:, 0] / 1e3, mesh[:, 1] / 1e3, s=1, color="#ecebe6", lw=0)
            zn, on = c["zoo"]["layout"], c["old"]["layout"]
            if on != zn:
                ax.scatter(*xy(on), s=34, facecolors="none", edgecolors=GREY, lw=1.0)
            ax.scatter(*xy(zn), s=14, color=BLUE, lw=0)
            t = f"{f.upper()} · {zn}\nzoo {100 * c['zoo']['frac']:.1f}%"
            t += (f"  ·  old {100 * c['old']['frac']:.1f}%" if on == zn
                  else f"\nold best (grey): {on} {100 * c['old']['frac']:.1f}%")
            ax.set_title(t, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{n} detectors", fontsize=12)
                ax.yaxis.set_visible(True); ax.set_yticks([])
    fig.suptitle(f"Best layout of each family within {th}° (zoo recon's best in blue; old recon's best in grey "
                 f"when different) · 6 × 6 km plane", fontsize=14, color=INK, y=1.0)
    fig.tight_layout()
    out = f"{Z}/best_layouts_{th}deg.png"
    fig.savefig(out, dpi=110, facecolor="white", bbox_inches="tight")
    print("[write]", out)

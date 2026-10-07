"""Catalogue of the 112 distinct test layouts behind lit_geometry.png / lit_resolution*.png, with their scores.

Each panel: the detector positions on the 6 km plane, and (relative-time recon, the same 1,536 held-out showers,
from lit_geometry_pairs.npz) the share within 1 and 5 deg, the median angle error, and per shower the median
number of lit detectors (>= 1 count), their median nearest-neighbour spacing and their baseline (largest distance).
Ranked best to worst by the share within 1 deg (frame colour = family). Writes layout_catalog_p<k>.png
(45 panels per page, shaped like a PDF slide).
"""
import os
import re
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
LISTS = (f"{D}/core_test_ndet/layouts.txt", f"{D}/clusters/layouts.txt", f"{Z}/layouts_angleonly.txt",
         f"{Z}/layouts_disks_n10.txt")
FAMS = ("grid", "disk", "cluster", "core", "hybrid", "optimized")
COL = {"grid": "#2a78d6", "disk": "#1baf7a", "cluster": "#eb6834", "core": "#c23b6b", "hybrid": "#8a5cd1",
       "optimized": "#52514e"}

P = np.load(f"{Z}/lit_geometry_pairs.npz")
names = list(P["names"])
# Same order and de-duplication as lit_geometry.py: first listed path for each distinct set of positions.
paths, seen, seen_xy = {}, set(), set()
for f in LISTS:
    for line in open(f):
        if not line.strip():
            continue
        name, path = line.strip().split("=", 1)
        if path in seen:
            continue
        seen.add(path)
        L = torch.load(path, map_location="cpu")
        e, n = np.asarray(L["x"], float), np.asarray(L["y"], float)
        key = tuple(sorted(zip(np.round(e, 1), np.round(n, 1))))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        paths[name] = (e, n)
assert list(paths) == names, "layout order differs from lit_geometry_pairs.npz"


def family(x):
    return ("optimized" if x.startswith("kern_") else "grid" if x.startswith("grid") else "disk" if x.startswith("disk")
            else "cluster" if x.startswith("clu") else "core" if re.match(r"core_r\d+", x) else "hybrid")


def ndet(x):
    m = re.search(r"_n(\d+)$", x)
    return int(m.group(1)) if m else 100


def param(x):
    m = re.search(r"_r(\d+)", x)
    return int(m.group(1)) if m else 0


lit, ang, lay, nn, base = P["lit"], P["ang"], P["lay"], P["nn"], P["base"]
f1 = {x: float(np.mean(ang[lay == i] < 1)) for i, x in enumerate(names)}
f5 = {x: float(np.mean(ang[lay == i] < 5)) for i, x in enumerate(names)}
# Ranked best to worst by the share of showers within 1 deg (ties: by share within 5 deg).
order = sorted(names, key=lambda x: (-f1[x], -f5[x], x))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU) / 1e3
ROWS, COLS = 5, 9
PER = ROWS * COLS
for page in range(0, len(order), PER):
    fig, axs = plt.subplots(ROWS, COLS, figsize=(33.5, 18))      # the slide's shape (3020 x 1620 px)
    for ax in axs.flat:
        ax.axis("off")
    for rank, (ax, nm) in enumerate(zip(axs.flat, order[page:page + PER]), start=page + 1):
        k = lay == names.index(nm)
        a, l = ang[k], lit[k]
        e, n = paths[nm]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(mesh[:, 0], mesh[:, 1], s=0.6, color="#ecebe6", lw=0)
        ax.scatter(e / 1e3, n / 1e3, s=9 if ndet(nm) <= 50 else 6, color=COL[family(nm)], lw=0)
        ax.set_xlim(mesh[:, 0].min() - 0.2, mesh[:, 0].max() + 0.2); ax.set_ylim(mesh[:, 1].min() - 0.2, mesh[:, 1].max() + 0.2)
        ax.set_aspect("equal")
        ax.set_title(f"#{rank}  {nm}  ·  {ndet(nm)} det.\n"
                     f"1°: {100 * np.mean(a < 1):.1f}%   5°: {100 * np.mean(a < 5):.1f}%   median {np.median(a):.2f}°\n"
                     f"lit {np.median(l):.0f}   NN {np.nanmedian(nn[k]):.0f} m   baseline {np.nanmedian(base[k]):.0f} m",
                     fontsize=10, color="#1f2328")
        for sp in ax.spines.values():
            sp.set_color(COL[family(nm)]); sp.set_linewidth(1.2)
    fig.suptitle(f"Test layouts ranked by share of showers within 1°: #{page + 1}–{min(page + PER, len(order))} of {len(order)} · "
                 "relative-time recon, 1,536 held-out showers · frame: grid blue, disk green, cluster orange, core red, "
                 "hybrid purple, optimized grey\nlit = median detectors with ≥ 1 count per shower · NN spacing / baseline = "
                 "medians over showers with ≥ 2 lit", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.3)
    out = f"{Z}/layout_catalog_p{page // PER + 1}.png"
    fig.savefig(out, dpi=80, facecolor="white"); plt.close(fig)
    print("[write]", out)

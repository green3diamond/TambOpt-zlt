"""Real layouts of the run 19 counting tests on the run 19 plane mesh (East, North, km):
    layouts_annuli.png   annuli r_in = 0 / 2 / 3 / 4 km (R = 6 km, 100 det.) and the grid, whole plane
    layouts_density.png  centred disks R = 0.5 / 0.75 / 1 / 1.5 km (100 det.) and the grid, whole plane
    layouts_region.png   Malata-like region: rows R = 500 / 750 / 1000 m; even disk (99) and triangles r = 10 / 25 / 100 m,
                         zoomed to the disk
Writes into <RUN_LOCATION>/diagnostics/counting/.

    python plots/analysis/run19/counting/plot_layouts.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "counting")
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3
centre = mesh.mean(0)


def paths(f):
    return dict(l.strip().split("=", 1) for l in open(os.path.join(D, f)) if l.strip())


def xy(p):
    L = torch.load(p, map_location="cpu")
    return np.asarray(L["x"], float) / 1e3, np.asarray(L["y"], float) / 1e3


def panel(ax, p, title, lim=None, s=10, col="#2a78d6"):
    ax.scatter(mesh[:, 0], mesh[:, 1], s=0.3, color="#e3e1da", lw=0, rasterized=True)
    e, n = xy(p)
    ax.scatter(e, n, s=s, color=col, lw=0)
    if lim is not None:
        ax.set_xlim(centre[0] - lim, centre[0] + lim); ax.set_ylim(centre[1] - lim, centre[1] + lim)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11); ax.tick_params(labelsize=8)
    ax.set_xlabel("East [km]", fontsize=9)


A, Dn, Rg = paths("annuli_layouts.txt"), paths("density_layouts.txt"), paths("region_layouts.txt")

fig, axs = plt.subplots(1, 5, figsize=(22, 5))
for ax, (k, t) in zip(axs, [(f"annulus_rin{r}_R6000_n100", f"annulus r_in = {r / 1e3:g} km") for r in (0, 2000, 3000, 4000)]
                      + [("grid_n100", "grid")]):
    panel(ax, A[k], t)
axs[0].set_ylabel("North [km]")
fig.suptitle("Hollow-centre test: 100 detectors, annulus r_in – 6 km (real layouts on the run 19 plane)")
fig.tight_layout(); fig.savefig(os.path.join(D, "layouts_annuli.png"), dpi=110, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(22, 5))
for ax, (k, t) in zip(axs, [(f"disk_r{r}_n100", f"disk R = {r / 1e3:g} km") for r in (500, 750, 1000, 1500)]
                      + [("grid_n100", "grid")]):
    panel(ax, Dn[k], t)
axs[0].set_ylabel("North [km]")
fig.suptitle("Density test: 100 detectors in centred disks (real layouts on the run 19 plane)")
fig.tight_layout(); fig.savefig(os.path.join(D, "layouts_density.png"), dpi=110, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(3, 4, figsize=(18, 13.5))
for i, R in enumerate((500, 750, 1000)):
    lim = R / 1e3 + 0.25
    panel(axs[i, 0], Rg[f"disk_r{R}_n99"], f"R = {R} m, even disk (99)", lim, s=14)
    for j, r in enumerate((10, 25, 100)):
        panel(axs[i, j + 1], Rg[f"tri33disk_R{R}_r{r}_n99"], f"R = {R} m, 33 triangles r = {r} m", lim, s=14, col="#c0392b")
    axs[i, 0].set_ylabel("North [km]")
fig.suptitle("Malata-like region: even disks vs triangle clusters (real layouts, zoomed to the disk)")
fig.tight_layout(); fig.savefig(os.path.join(D, "layouts_region.png"), dpi=110, facecolor="white"); plt.close(fig)
print("[write]", D, "layouts_annuli.png layouts_density.png layouts_region.png")

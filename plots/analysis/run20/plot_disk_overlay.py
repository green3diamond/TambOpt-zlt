"""Run 20 (Malata): the grid and centred disk layouts of several radii on the detector region, to choose the cable
reference L_ref. The first panel is grid_n99 (make_triplets.py grid()). Each panel: the 266 faces (East-North, coloured by Up), the disk circle, its 99 sunflower points (as make_triplets.py
disk()) after project_to_mountain_ne; points snapped to the mesh are drawn in red. The title gives the share snapped
and the 3-D minimum-spanning-tree cable length of the snapped layout (Up interpolated on the mesh vertices).
Writes <run 20>/diagnostics/disk_overlay.png.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_disk_overlay.py [--radii R ...]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, project_to_mountain_ne, sample_initial_layout_ne  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--radii", type=int, nargs="+", default=[150, 400, 750, 1000, 1500])
ap.add_argument("--n", type=int, default=99)
args = ap.parse_args()
OUT = os.path.join(C.RUN_LOCATION, "diagnostics", "disk_overlay.png")

m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)   # (F, 3, 3) ENU
vx = tri.reshape(-1, 3)
up = LinearNDInterpolator(vx[:, :2], vx[:, 2])
up_near = lambda p: vx[cdist(p, vx[:, :2]).argmin(1), 2]                                           # outside the hull


def disk(n, R):
    i = np.arange(n) + 0.5
    r, th = R * np.sqrt(i / n), i * np.pi * (3 - np.sqrt(5))
    return centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)


def mst_km(p):
    return minimum_spanning_tree(cdist(p, p)).sum() / 1e3


def grid(n):
    torch.manual_seed(0)
    return tuple(np.asarray(t, float) for t in sample_initial_layout_ne(m, n_units=n, scheme="grid"))


panels = [(None, grid(args.n))] + [(R, disk(args.n, R)) for R in args.radii]
fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.9), constrained_layout=True)
for ax, (R, (e0, n0)) in zip(axs, panels):
    e, n = (t.numpy().astype(float) for t in project_to_mountain_ne(m, torch.tensor(e0).float(), torch.tensor(n0).float()))
    moved = np.hypot(e - e0, n - n0) > 1.0
    en = np.c_[e, n]
    z = up(en); z = np.where(np.isfinite(z), z, up_near(en))
    L = mst_km(np.c_[en, z])
    pc = PolyCollection(tri[:, :, :2], array=tri[:, :, 2].mean(1), cmap="Greys", edgecolors="white", linewidths=0.3)
    ax.add_collection(pc)
    if R is not None:
        ax.add_patch(plt.Circle(centre, R, fill=False, color="#2a6fb0", lw=1.2, ls="--"))
    ax.scatter(e0[moved], n0[moved], s=6, c="#e8a3a3", lw=0, label="before snapping")
    ax.scatter(e[~moved], n[~moved], s=9, c="#2a6fb0", lw=0, label="on the region")
    ax.scatter(e[moved], n[moved], s=9, c="#c0392b", lw=0, label="snapped to the mesh")
    h = max(R or 0, np.abs(tri[..., :2] - centre).max()) * 1.08
    ax.set_xlim(centre[0] - h, centre[0] + h); ax.set_ylim(centre[1] - h, centre[1] + h); ax.set_aspect("equal")
    ax.set_title(f"{'Grid' if R is None else f'Disk R = {R} m'}: {100 * moved.mean():.0f}% snapped\nMST cable {L:.1f} km (3-D)", fontsize=10)
    ax.set_xlabel("East [m]"); ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(f"{'grid' if R is None else f'R={R} m'}: {moved.sum()}/{args.n} snapped, MST {L:.2f} km")
axs[0].set_ylabel("North [m]")
axs[0].legend(fontsize=7, loc="lower left", frameon=False)
cb = fig.colorbar(pc, ax=axs, shrink=0.7, pad=0.01); cb.set_label("Up [m]"); cb.outline.set_visible(False)
fig.suptitle(f"Run 20, Malata detector region: {args.n}-detector grid and centred disks (as make_triplets.py)", fontsize=12)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, facecolor="white")
print("[write]", OUT)

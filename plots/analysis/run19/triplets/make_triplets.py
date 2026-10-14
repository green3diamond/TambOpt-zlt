"""Run 19 triplet test layouts: writes <RUN_LOCATION>/diagnostics/triplets/layouts/*.pt and layouts.txt.

    tri<K>_r<r>_n<3K>   K cluster centres on an even grid, 3 detectors per cluster on an equilateral triangle of
                        circumradius r (side r*sqrt(3)), each triangle at a random rotation; K = 33 / 16 / 7 / 3, r = 10-500 m
    grid_n<N>, disk_r<R>_n<N>   the grid and centred disks at the same N = 99 / 48 / 21 / 9 (the main test set has 100 / 50 / 20 / 10)
Detectors off the plane snap to its nearest face, as everywhere in the pipeline.
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

T = os.path.join(C.RUN_LOCATION, "diagnostics", "triplets")
os.makedirs(os.path.join(T, "layouts"), exist_ok=True)
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
DISKS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
KS = (33, 16, 7, 3)
TRI_R = (10, 25, 50, 100, 200, 500)
lines = []


def snap(e, n):
    e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float(), n.float()


def save(name, en):
    p = os.path.join(T, "layouts", f"{name}.pt")
    torch.save(dict(x=en[0], y=en[1]), p)
    lines.append(f"{name}={p}")


def grid(n):
    torch.manual_seed(0)
    return sample_initial_layout_ne(m, n_units=n, scheme="grid")


def disk(n, R):
    i = np.arange(n) + 0.5
    r, th = R * np.sqrt(i / n), i * np.pi * (3 - np.sqrt(5))
    return centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)


for K in KS:
    N = 3 * K
    ce, cn = (np.asarray(t, float) for t in grid(K))
    assert ce.size == K, (K, ce.size)
    th = np.random.default_rng(K).uniform(0, 2 * np.pi, K)[:, None] + 2 * np.pi * np.arange(3) / 3
    for r in TRI_R:
        save(f"tri{K}_r{r}_n{N}", snap((ce[:, None] + r * np.cos(th)).ravel(), (cn[:, None] + r * np.sin(th)).ravel()))
    save(f"grid_n{N}", snap(*grid(N)))
    for R in DISKS:
        save(f"disk_r{R}_n{N}", snap(*disk(N, R)))
open(os.path.join(T, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts -> {T}/layouts.txt")

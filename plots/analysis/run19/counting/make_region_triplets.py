"""Run 19 counting test, Malata-like density: triplet clusters inside a disk at the plane centre.
Writes <RUN_LOCATION>/diagnostics/counting/layouts/*.pt and region_layouts.txt:

    tri33disk_R<R>_r<r>_n99   33 cluster centres on a sunflower disk of radius R (as the disk_r<R> layouts), 3 detectors
                              per cluster on an equilateral triangle of circumradius r, each at a random rotation (the
                              rotations of make_triplets.py, rng(33)); R = 500 / 750 / 1000 m, r = 10 / 25 / 50 / 100 m
    disk_r<R>_n99             the 99-detector disks of the triplet test (triplets/layouts.txt), same R, for comparison
Detectors off the plane snap to its nearest face, as everywhere in the pipeline.
--radii / --tri-r replace R and r (run 20, Malata: --radii 150 --tri-r 25; each R needs disk_r<R>_n99 in
triplets/layouts.txt, i.e. make_triplets.py --disks including it).

    python plots/analysis/run19/counting/make_region_triplets.py [--radii R ...] [--tri-r r ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, project_to_mountain_ne  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "counting")
os.makedirs(os.path.join(D, "layouts"), exist_ok=True)
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
ap = argparse.ArgumentParser()
ap.add_argument("--radii", type=int, nargs="+", default=[500, 750, 1000], help="cluster-disk radii R [m]")
ap.add_argument("--tri-r", type=int, nargs="+", default=[10, 25, 50, 100], help="triangle circumradii r [m]")
args = ap.parse_args()
K, RS, TRI_R = 33, tuple(args.radii), tuple(args.tri_r)
lines = []


def snap(e, n):
    e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float(), n.float()


def disk(n, R):                                                              # as make_triplets.py / tests/make_layouts.py
    i = np.arange(n) + 0.5
    r, th = R * np.sqrt(i / n), i * np.pi * (3 - np.sqrt(5))
    return centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)


th = np.random.default_rng(K).uniform(0, 2 * np.pi, K)[:, None] + 2 * np.pi * np.arange(3) / 3
for R in RS:
    ce, cn = disk(K, R)
    for r in TRI_R:
        name = f"tri{K}disk_R{R}_r{r}_n{3 * K}"
        e, n = snap((ce[:, None] + r * np.cos(th)).ravel(), (cn[:, None] + r * np.sin(th)).ravel())
        p = os.path.join(D, "layouts", f"{name}.pt")
        torch.save(dict(x=e, y=n), p)
        lines.append(f"{name}={p}")
tri = dict(l.strip().split("=", 1) for l in open(os.path.join(C.RUN_LOCATION, "diagnostics", "triplets", "layouts.txt")) if l.strip())
lines += [f"disk_r{R}_n99={tri[f'disk_r{R}_n99']}" for R in RS]
open(os.path.join(D, "region_layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts -> {D}/region_layouts.txt")

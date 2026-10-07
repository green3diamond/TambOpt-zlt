"""10-detector sunflower disks (radius 100 m - 12 km) at the plane centre, built exactly like the 20 / 50-detector
disks in ../core_test_ndet/make_layouts.py (detectors off the plane snap to its nearest edge).
Writes layouts_n10/disk_r<R>_n10.pt and layouts_disks_n10.txt."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, project_to_mountain_ne  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
DISKS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
N = 10
os.makedirs(os.path.join(Z, "layouts_n10"), exist_ok=True)
lines = []
for R in DISKS:
    i = np.arange(N) + 0.5
    r, th = R * np.sqrt(i / N), i * np.pi * (3 - np.sqrt(5))
    e, n = project_to_mountain_ne(m, torch.as_tensor(centre[0] + r * np.cos(th)).float(),
                                  torch.as_tensor(centre[1] + r * np.sin(th)).float())
    p = os.path.join(Z, "layouts_n10", f"disk_r{R}_n{N}.pt")
    torch.save(dict(x=e.float(), y=n.float()), p)
    lines.append(f"disk_r{R}_n{N}={p}")
open(os.path.join(Z, "layouts_disks_n10.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts")

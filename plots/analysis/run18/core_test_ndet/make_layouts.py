"""Run 18 core test with fewer detectors (N = 20, 50): grid, compact sunflower cores (r 100-500 m) at the
plane centre, core + spread / rim hybrids in the same proportions as the 100-detector core test, and
disks of growing radius (100 m - 12 km; detectors off the plane snap to its nearest edge face).
Outer detectors of the hybrids come from the 100-detector grid outside the core (evenly thinned, or the
farthest for 'rim'), as in ../core_test/make_core_layouts.py. Writes layouts/<name>.pt and layouts.txt."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")   # branch run18-showerframe (run 18 constants)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
CORES = (100, 200, 300, 500)
DISKS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
HYB = (("core300_60_spread40", 0.6, 300, "spread"), ("core300_40_spread60", 0.4, 300, "spread"),
       ("core300_60_rim40", 0.6, 300, "rim"), ("core200_50_spread50", 0.5, 200, "spread"))


def snap(e, n):
    e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float(), n.float()


def disk(n, R):
    i = np.arange(n) + 0.5
    r, th = R * np.sqrt(i / n), i * np.pi * (3 - np.sqrt(5))
    return centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)


torch.manual_seed(0)
g100 = np.stack([t.numpy() for t in snap(*sample_initial_layout_ne(m, n_units=100, scheme="grid"))], 1)
dist = np.hypot(*(g100 - centre).T)
lines = []
for N in (20, 50):
    def save(name, en):
        p = os.path.join(D, "layouts", f"{name}_n{N}.pt")
        torch.save(dict(x=en[0], y=en[1]), p)
        lines.append(f"{name}_n{N}={p}")
    torch.manual_seed(0)
    save("grid", snap(*sample_initial_layout_ne(m, n_units=N, scheme="grid")))
    for R in CORES:
        save(f"core_r{R}", snap(*disk(N, R)))
    for name, frac, rc, outer in HYB:
        nc = int(round(frac * N)); k = N - nc
        ce, cn = disk(nc, rc)
        out = g100[dist > rc]
        pick = (out[np.round(np.linspace(0, len(out) - 1, k)).astype(int)] if outer == "spread"
                else out[np.argsort(-np.hypot(*(out - centre).T))[:k]])
        save(name, snap(np.concatenate([ce, pick[:, 0]]), np.concatenate([cn, pick[:, 1]])))
    for R in DISKS:
        save(f"disk_r{R}", snap(*disk(N, R)))
open(os.path.join(D, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts")

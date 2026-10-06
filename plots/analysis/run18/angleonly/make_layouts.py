"""Run 18 angle-only rescoring: collect every layout from the earlier plots (core test: grid, compact
cores, hybrids, kernel-optimized; detector-count grids of 4-100) and add 100-detector sunflower disks
at the plane centre with radius 100 m - 12 km (twice the 6 km plane; detectors off the plane snap to
its nearest edge face, as the pipeline does). Writes layouts/<name>.pt and layouts.txt."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")   # branch run18-showerframe (run 18 constants)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(os.path.dirname(D))
RADII = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
COUNTS = (4, 5, 10, 20, 25, 50)
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
lines = [l.strip() for l in open(os.path.join(RUN, "diagnostics/core_test/layouts.txt")) if l.strip()]


def save(name, e, n):
    p = os.path.join(D, "layouts", f"{name}.pt")
    torch.save(dict(x=e.float(), y=n.float()), p)
    lines.append(f"{name}={p}")


for nd in COUNTS:
    torch.manual_seed(0)
    e, n = sample_initial_layout_ne(m, n_units=nd, scheme="grid")
    save(f"grid_n{nd}", *project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float()))
i = np.arange(C.N_DETECTORS) + 0.5
for R in RADII:
    r, th = R * np.sqrt(i / C.N_DETECTORS), i * np.pi * (3 - np.sqrt(5))
    e, n = project_to_mountain_ne(m, torch.as_tensor(centre[0] + r * np.cos(th)).float(),
                                  torch.as_tensor(centre[1] + r * np.sin(th)).float())
    save(f"disk_r{R}", e, n)
open(os.path.join(D, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts; plane centre E {centre[0]:.0f} N {centre[1]:.0f}")

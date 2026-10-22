"""Run 19 hollow-centre test: 100 detectors evenly filling an annulus r_in <= r <= 6 km at the plane centre (sunflower
with area-uniform radii, as the disk_r<R> layouts, which are r_in = 0), r_in = 0 / 2 / 3 / 4 km. Writes
<RUN_LOCATION>/diagnostics/counting/layouts/annulus_rin<r_in>_R6000_n100.pt and annuli_layouts.txt (+ grid_n100).

    python plots/analysis/run19/counting/make_annuli.py
"""
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
N, R = 100, 6000
lines = []
for r_in in (0, 2000, 3000, 4000):
    i = np.arange(N) + 0.5
    r, th = np.sqrt(r_in ** 2 + (R ** 2 - r_in ** 2) * i / N), i * np.pi * (3 - np.sqrt(5))
    e, n = project_to_mountain_ne(m, torch.as_tensor(centre[0] + r * np.cos(th)).float(),
                                  torch.as_tensor(centre[1] + r * np.sin(th)).float())
    name = f"annulus_rin{r_in}_R{R}_n{N}"
    p = os.path.join(D, "layouts", f"{name}.pt")
    torch.save(dict(x=e.float(), y=n.float()), p)
    lines.append(f"{name}={p}")
tests = dict(l.strip().split("=", 1) for l in open(os.path.join(C.RUN_LOCATION, "diagnostics", "tests", "layouts.txt")) if l.strip())
lines.append(f"grid_n100={tests['grid_n100']}")
open(os.path.join(D, "annuli_layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts -> {D}/annuli_layouts.txt")

"""Run 19 hollow-centre test: 100 detectors evenly filling an annulus r_in <= r <= 6 km at the plane centre (sunflower
with area-uniform radii, as the disk_r<R> layouts, which are r_in = 0), r_in = 0 / 2 / 3 / 4 km. Writes
<RUN_LOCATION>/diagnostics/counting/layouts/annulus_rin<r_in>_R6000_n100.pt and annuli_layouts.txt (+ grid_n100).
--R / --rin / --n / --list for other annuli (e.g. donuts at the scale of the best disk, no grid appended).

    python plots/analysis/run19/counting/make_annuli.py
    python plots/analysis/run19/counting/make_annuli.py --R 1500 --rin 250 500 750 1000 --n 99 --list donuts_R1500.txt
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
ap.add_argument("--R", type=float, default=6000)
ap.add_argument("--rin", type=float, nargs="+", default=[0, 2000, 3000, 4000])
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--list", default="annuli_layouts.txt")
args = ap.parse_args()
N, R = args.n, int(args.R)
lines = []
for r_in in (int(r) for r in args.rin):
    i = np.arange(N) + 0.5
    r, th = np.sqrt(r_in ** 2 + (R ** 2 - r_in ** 2) * i / N), i * np.pi * (3 - np.sqrt(5))
    e, n = project_to_mountain_ne(m, torch.as_tensor(centre[0] + r * np.cos(th)).float(),
                                  torch.as_tensor(centre[1] + r * np.sin(th)).float())
    name = f"annulus_rin{r_in}_R{R}_n{N}"
    p = os.path.join(D, "layouts", f"{name}.pt")
    torch.save(dict(x=e.float(), y=n.float()), p)
    lines.append(f"{name}={p}")
if args.list == "annuli_layouts.txt":
    tests = dict(l.strip().split("=", 1) for l in open(os.path.join(C.RUN_LOCATION, "diagnostics", "tests", "layouts.txt")) if l.strip())
    lines.append(f"grid_n100={tests['grid_n100']}")
open(os.path.join(D, args.list), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts -> {D}/{args.list}")

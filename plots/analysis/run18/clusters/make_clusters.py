"""Run 18: clustered layouts. K cluster centres on an even grid over the plane, M detectors per cluster
on a small sunflower disk of radius r around each centre (K x M = N). N = 20 (4x5, 5x4, 10x2), 50 (10x5,
25x2) and 100 (20x5, 25x4); r = 50, 100, 200, 300, 500 m. Detectors snap to the plane like everywhere
else. Writes layouts/<name>.pt and layouts.txt (plus the plain grids for reference)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
SPLITS = {20: ((4, 5), (5, 4), (10, 2)), 50: ((10, 5), (25, 2)), 100: ((20, 5), (25, 4))}
RADII = (50, 100, 200, 300, 500)
lines = []


def snap(e, n):
    e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float(), n.float()


def save(name, en):
    p = os.path.join(D, "layouts", f"{name}.pt")
    torch.save(dict(x=en[0], y=en[1]), p)
    lines.append(f"{name}={p}")


for N, splits in SPLITS.items():
    torch.manual_seed(0)
    save(f"grid_n{N}", snap(*sample_initial_layout_ne(m, n_units=N, scheme="grid")))
    for K, M in splits:
        torch.manual_seed(0)
        ce, cn = (np.asarray(t, float) for t in sample_initial_layout_ne(m, n_units=K, scheme="grid"))
        i = np.arange(M) + 0.5
        for r in RADII:
            rr, th = (r * np.sqrt(i / M) if M > 1 else np.zeros(1)), i * np.pi * (3 - np.sqrt(5))
            e = (ce[:, None] + rr * np.cos(th)).ravel(); n = (cn[:, None] + rr * np.sin(th)).ravel()
            save(f"clu{K}x{M}_r{r}_n{N}", snap(e, n))
open(os.path.join(D, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(len(lines), "layouts")

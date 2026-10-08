"""Run 19 test layouts (the run 18 families, on the run's plane): writes <RUN_LOCATION>/diagnostics/tests/layouts/*.pt
and layouts.txt.

    grid_n<N>                 even grid over the plane, N = 4, 5, 10, 20, 25, 50, 100
    disk_r<R>_n<N>            sunflower disk at the plane centre, N = 10 / 20 / 50 / 100, R = 0.1-12 km
    core_r<R>_n<N>            the compact disks again under the core name, R = 100-500 m, N = 20 / 50 / 100
    core<R>_<a>_<how><b>_n<N> hybrids: a% of the detectors in a core of radius R, the rest from the 100-grid outside it
                              (evenly thinned = spread, farthest = rim), N = 20 / 50 / 100
    clu<K>x<M>_r<r>_n<N>      K cluster centres on an even grid, M detectors on a small disk of radius r around each
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

T = os.path.join(C.RUN_LOCATION, "diagnostics", "tests")
os.makedirs(os.path.join(T, "layouts"), exist_ok=True)
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
centre = np.asarray(m.centroids_ENU)[:, :2].mean(0)
DISKS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000)
CORES = (100, 200, 300, 500)
HYB = (("core300_60_spread40", 0.6, 300, "spread"), ("core300_40_spread60", 0.4, 300, "spread"),
       ("core300_60_rim40", 0.6, 300, "rim"), ("core200_50_spread50", 0.5, 200, "spread"))
SPLITS = {20: ((4, 5), (5, 4), (10, 2)), 50: ((10, 5), (25, 2)), 100: ((20, 5), (25, 4))}
CLU_R = (50, 100, 200, 300, 500, 1000)
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


for N in (4, 5, 10, 20, 25, 50, 100):
    save(f"grid_n{N}", snap(*grid(N)))
for N in (10, 20, 50, 100):
    for R in DISKS:
        save(f"disk_r{R}_n{N}", snap(*disk(N, R)))
g100 = np.stack([t.numpy() for t in snap(*grid(100))], 1)
dist = np.hypot(*(g100 - centre).T)
for N in (20, 50, 100):
    for R in CORES:
        save(f"core_r{R}_n{N}", snap(*disk(N, R)))
    for name, frac, rc, outer in HYB:
        nc = int(round(frac * N)); k = N - nc
        ce, cn = disk(nc, rc)
        out = g100[dist > rc]
        pick = (out[np.round(np.linspace(0, len(out) - 1, k)).astype(int)] if outer == "spread"
                else out[np.argsort(-np.hypot(*(out - centre).T))[:k]])
        save(f"{name}_n{N}", snap(np.concatenate([ce, pick[:, 0]]), np.concatenate([cn, pick[:, 1]])))
    for K, M in SPLITS[N]:
        ce, cn = (np.asarray(t, float) for t in grid(K))
        i = np.arange(M) + 0.5
        for r in CLU_R:
            rr, th = r * np.sqrt(i / M), i * np.pi * (3 - np.sqrt(5))
            save(f"clu{K}x{M}_r{r}_n{N}", snap((ce[:, None] + rr * np.cos(th)).ravel(), (cn[:, None] + rr * np.sin(th)).ravel()))
open(os.path.join(T, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} layouts -> {T}/layouts.txt; plane centre E {centre[0]:.0f} N {centre[1]:.0f}, "
      f"E {m.east_lo:.0f}..{m.east_hi:.0f}, N {m.n_min:.0f}..{m.n_max:.0f}")

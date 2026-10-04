"""Hand-made 100-detector layouts for the core test: compact cores of several radii at the mesh
centre, core + outer hybrids, plus the grid and run 17's kernel-optimized layouts (kernel_opt_run17.sh) for comparison.
(The surrogate-optimized Stage 4 layouts and the first core test are in archive/stage4_surrogate_2026-10-01.)
Writes one <name>.pt ({x: East, y: North}) per layout into this folder and layouts.txt with
name=path lines for score_layouts.py."""
import os
import sys

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-plane (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(os.path.dirname(D))
N = 100
mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)[:, :2]
centre = mesh.mean(0)


def disk(n, radius):
    """Evenly spread points in a disk (sunflower pattern) around the mesh centre."""
    i = np.arange(n) + 0.5
    r = radius * np.sqrt(i / n)
    th = i * np.pi * (3 - np.sqrt(5))
    return centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)


def snap(e, n):
    e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float(), n.float()


torch.manual_seed(0)
ge, gn = sample_initial_layout_ne(mountain, n_units=N, scheme="grid")
ge, gn = snap(ge, gn)
grid = np.stack([ge.numpy(), gn.numpy()], 1)
dist = np.hypot(*(grid - centre).T)

layouts = {"grid": (ge, gn)}
for R in (100, 200, 300, 500):
    layouts[f"core_r{R}"] = snap(*disk(N, R))


def hybrid(n_core, r_core, outer):
    ce, cn = disk(n_core, r_core)
    k = N - n_core
    out = grid[dist > r_core]
    if outer == "spread":          # evenly thinned grid points outside the core
        pick = out[np.round(np.linspace(0, len(out) - 1, k)).astype(int)]
    else:                          # "rim": the grid points farthest from the centre
        pick = out[np.argsort(-np.hypot(*(out - centre).T))[:k]]
    return snap(np.concatenate([ce, pick[:, 0]]), np.concatenate([cn, pick[:, 1]]))


layouts["core300_60_spread40"] = hybrid(60, 300, "spread")
layouts["core300_40_spread60"] = hybrid(40, 300, "spread")
layouts["core300_60_rim40"] = hybrid(60, 300, "rim")
layouts["core200_50_spread50"] = hybrid(50, 200, "spread")

lines = []
for name, (e, n) in layouts.items():
    p = os.path.join(D, f"{name}.pt")
    torch.save(dict(x=e, y=n), p)
    lines.append(f"{name}={p}")
existing = {}
for tag in ("current", "twotier_l0", "twotier_l0p3"):          # kernel-in-the-loop (kernel_opt_run17.sh)
    for start in ("grid", "center"):
        existing[f"kern_{tag}_{start}"] = f"test_v6_run_04_optimize_lbfgs_kernel_{tag}_lr10_{start}/layout_best.pt"
for name, rel in existing.items():
    if os.path.exists(os.path.join(RUN, rel)):
        lines.append(f"{name}={os.path.join(RUN, rel)}")
    else:
        print(f"[skip] {name}: {rel} not there yet")
open(os.path.join(D, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print(f"centre E={centre[0]:.0f} N={centre[1]:.0f}; wrote {len(layouts)} layouts + {len(existing)} existing")

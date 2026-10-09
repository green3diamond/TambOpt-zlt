"""Layouts for the 1 deg angle-only optimisations (two-tier, lambda 0, 5k steps, relative-time recon):
the grid / centre starts (init_layout of 04_optimize_lbfgs_kernel.py, deterministic) and each run's
layout_best.pt, for 100 and 10 detectors. Writes layouts/*.pt and layouts.txt (name=path)."""
import os
import sys

import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, project_to_mountain_ne, sample_initial_layout_ne  # noqa: E402

O = os.path.dirname(os.path.abspath(__file__))
R18 = os.path.dirname(os.path.dirname(os.path.dirname(O)))
OPT = f"{R18}/test_v6_run_04_optimize_lbfgs_kernel_twotier_l0_1deg_angleonly_lr10_{{}}_reltime{{}}"
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
lines = []
for n in (100, 10):
    for init in ("grid", "center"):
        e, nn = sample_initial_layout_ne(m, n_units=n, scheme=init)
        e, nn = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(nn).float())
        p = f"{O}/layouts/init_{init}_n{n}.pt"
        torch.save(dict(x=e.float(), y=nn.float()), p)
        best = OPT.format(init, "" if n == 100 else f"_n{n}") + "/layout_best.pt"
        assert os.path.exists(best), best
        lines += [f"init_{init}_n{n}={p}", f"opt_{init}_n{n}={best}"]
open(f"{O}/layouts.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

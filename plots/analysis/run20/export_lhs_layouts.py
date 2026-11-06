"""Run 20 LHS repeats: the 5 Latin-hypercube starts (Stage 4 lhs_on_mesh, seeds 1-5) and the final layouts of the 60
surrogate runs started from them (5 deg utility: 4 energy bands + 4 construction penalties, sur5_lhs<s>_<name>;
penalty only: 4 penalties, pen_lhs<s>_<name>), plus the 100-detector grid, as the validation list
<run 20>/diagnostics/lhs/layouts.txt (layout_final.pt next to each run's opt_state.pt).

    source slurm/run20_guard.sh; python plots/analysis/run20/export_lhs_layouts.py
"""
import importlib.util
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402

spec = importlib.util.spec_from_file_location("s4", os.path.join(REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
S4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S4)
PRE = os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_")
OUT = os.path.join(C.RUN_LOCATION, "diagnostics", "lhs")
SEEDS = (1, 2, 3, 4, 5)
SUR = ("e500_567", "e567_633", "e633_700", "eall", "slope", "access", "pads", "contour")
PEN = ("slope", "access", "pads", "contour")
os.makedirs(OUT, exist_ok=True)
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
lines = []
e, n = S4.init_layout(m, "grid", "cpu", 100)
torch.save(dict(x=e.cpu(), y=n.cpu()), os.path.join(OUT, "grid_n100.pt"))
lines.append("grid_n100=" + os.path.join(OUT, "grid_n100.pt"))
for s in SEEDS:
    e, n = S4.init_layout(m, f"lhs:{s}", "cpu", 100)
    f = os.path.join(OUT, f"lhs{s}_start.pt")
    torch.save(dict(x=e.cpu(), y=n.cpu()), f)
    lines.append(f"lhs{s}_start={f}")
    for tag, names in (("sur5", SUR), ("pen", PEN)):
        for k in names:
            d = PRE + f"{tag}_lhs{s}_{k}"
            st = torch.load(os.path.join(d, "opt_state.pt"), map_location="cpu", weights_only=False)
            f = os.path.join(d, "layout_final.pt")
            torch.save(dict(x=st["xy"]["x"].detach().float().cpu(), y=st["xy"]["y"].detach().float().cpu(),
                            epoch=st["epoch"]), f)
            lines.append(f"{tag}_lhs{s}_{k}={f}")
            print(f"[write] {f} (step {st['epoch']})")
open(os.path.join(OUT, "layouts.txt"), "w").write("\n".join(lines) + "\n")
print("[write]", os.path.join(OUT, "layouts.txt"), len(lines), "layouts")

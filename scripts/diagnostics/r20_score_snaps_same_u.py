"""Run 20: score the layout snapshots of one sur5snap_<name> run (slurm/run20_snap_runs.sh) with exactly the objective
that run optimised -- Stage 4's utility (TAMBOPT_UTILITY / TRIG_* env as in the run: 5 deg reward x the smooth Poisson
trigger probability on expected counts, x30), minus its construction penalty, on its own scoring set (the last
--n-score pool showers, the --e-range band) -- once with the Stage 2 surrogate (should reproduce the run's own
scoring-set U) and once with the kernel (same formula, the kernel as the detector response).
Writes {step: {"surrogate": U, "kernel": U}} to <run 20>/diagnostics/snaps/same_u_<name>.json.

    TAMBOPT_... (as the run) python scripts/diagnostics/r20_score_snaps_same_u.py --name eall --n-score 3072
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("s4", os.path.join(REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
S4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S4)
import modules.constants as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--n-score", type=int, default=1024)
ap.add_argument("--e-range", type=float, nargs=2, default=None)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--seed", type=int, default=S4.SEED)
args = ap.parse_args()
dev = S4.DEVICE

st = torch.load(os.path.join(C.RUN_LOCATION, f"test_v6_run_04_optimize_lbfgs_kernel_sur5snap_{args.name}",
                             "opt_state.pt"), map_location="cpu", weights_only=False)
snaps = st["snaps"]
mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
recon = S4.load_recon(C.RECON_FOLDER + "_deepsets", dev)
pen = S4.make_penalties(mountain, surface)                  # TAMBOPT_CONSTRUCT as in the run (None: no penalty)
sur = dict(track_seen=False, counting="none", surrogate=S4.load_dual_surrogate(C.FNN_FOLDER, dev, n_det=100),
           p_min=0.5, window_ns=0.0, gate="none", t_fallback_us=27.7)
pool = S4.MainPool(mountain)
lo_hi = args.e_range


def band(batch):
    if lo_hi is None:
        return batch
    clouds, bws, prim, frames = batch
    le = prim[:, 3] * (S4.LOG_E_MAX - S4.LOG_E_MIN) + S4.LOG_E_MIN
    k = (le >= lo_hi[0]) & (le < lo_hi[1])
    return ([c[k] for c in clouds], [w[k] if w is not None else None for w in bws], prim[k],
            frames[k] if frames is not None else None)


a = pool.n - args.n_score
sets = {"kernel": band(pool.load(a, pool.n)), "surrogate": band(pool.load_prim(a, pool.n))}
print(f"[score set] events {a}..{pool.n}, {sets['kernel'][2].shape[0]} in band", flush=True)


def gen():
    g = torch.Generator(device=dev)
    g.manual_seed(int(args.seed * 1000003 + 999_999_937))       # Stage 4's scoring draws
    return g


res = {}
for step, (x, y) in sorted(snaps.items()):
    xy = S4.LearnableXY(x.float().clone(), y.float().clone(), device=str(dev)).to(dev)
    row = {}
    for resp_name, resp in (("surrogate", sur), ("kernel", None)):
        tot = S4.pool_utility(xy, sets[resp_name], surface, recon, args.chunk, backward=False, resp=resp, gen=gen())
        if pen is not None:
            tot = S4.add_cable(pen, xy, tot, False)
        row[resp_name] = float(tot["U"])
    res[int(step)] = row
    log_u = next((s["U_score"] for s in st["score_log"] if s["epoch"] == step), float("nan"))
    print(f"  step {step:4d}: surrogate {row['surrogate']:+.3f} (run's own {log_u:+.3f})  kernel {row['kernel']:+.3f}",
          flush=True)

out = os.path.join(C.RUN_LOCATION, "diagnostics", "snaps", f"same_u_{args.name}.json")
json.dump(res, open(out, "w"), indent=1)
print("[write]", out)

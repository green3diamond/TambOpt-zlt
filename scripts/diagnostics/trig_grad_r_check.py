"""How much does the detached axis distance change the swgo_trigger gradient? (audit: `axis_distance` runs under
no_grad, background.py:137, so mu = E x particles_per_E(s, r) gets no gradient through r.)

Per layout and batch: the Stage 4 utility gradient (pool_utility, as the _m5k runs: mean time model without noise)
once as now (r detached) and once with r attached (a differentiable axis_distance patched into Stage 4). Reports
  cosine(now, attached), |attached - now| / |attached|, U (must be identical), and for scale the cosine between the
  attached gradients of different batches (batch noise).
Writes <run>/diagnostics/trig_grad_r_check.json.

    TAMBOPT_UTILITY=swgo_trigger ... python scripts/diagnostics/trig_grad_r_check.py [--k 4] [--batch 5120]
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
from modules.data.dataset_builder import detectors_in_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--batch", type=int, default=5120)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--opt", default=os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_a2_"
                                              "lr10_grid_b2048_m5k", "layout_best.pt"))
ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics", "trig_grad_r_check.json"))
args = ap.parse_args()
dev = S4.DEVICE
assert S4.UTILITY_MODE == "swgo_trigger", "set TAMBOPT_UTILITY=swgo_trigger"

mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
recon = S4.load_recon(C.RECON_FOLDER + "_deepsets", dev)
pool = S4.MainPool(mountain)
n_sweep = pool.n - 1024
starts = np.random.default_rng(7).choice(n_sweep // args.batch, args.k, replace=False) * args.batch
batches = [pool.load(int(a), int(a) + args.batch) for a in starts]
print(f"[pool] {pool.n} events; {args.k} batches of {args.batch}", flush=True)

detached = S4.axis_distance


def attached(e_det, n_det, surface, frames):
    return detectors_in_frames(e_det, n_det, surface, frames)[0].float()


LAYOUTS = {"grid": S4.init_layout(mountain, "grid", dev, C.N_DETECTORS)}
if os.path.exists(args.opt):
    o = torch.load(args.opt, map_location="cpu")
    LAYOUTS["opt swgotrig grid m5k"] = (o["x"].float().to(dev), o["y"].float().to(dev))


def grad(e0, n0, b, fn):
    S4.axis_distance = fn
    xy = S4.LearnableXY(e0.clone(), n0.clone(), device=str(dev)).to(dev)
    tot = S4.pool_utility(xy, b, surface, recon, args.chunk, backward=True, resp=None, gen=None)
    S4.axis_distance = detached
    return torch.cat([xy.x.grad, xy.y.grad]).detach().cpu().double(), float(tot["U"])


cos = torch.nn.functional.cosine_similarity
res = {}
for name, (e0, n0) in LAYOUTS.items():
    rows, GA = [], []
    for i, b in enumerate(batches):
        gd, ud = grad(e0, n0, b, detached)
        ga, ua = grad(e0, n0, b, attached)
        GA.append(ga)
        rows.append(dict(cos=float(cos(gd, ga, dim=0)), rel_diff=float((ga - gd).norm() / ga.norm()),
                         norm_ratio=float(gd.norm() / ga.norm()), U_detached=ud, U_attached=ua))
        print(f"[{name}] batch {i}: cos(now, attached) {rows[-1]['cos']:.4f}  |diff|/|attached| "
              f"{rows[-1]['rel_diff']:.3f}  |now|/|attached| {rows[-1]['norm_ratio']:.3f}  U {ud:.4f} / {ua:.4f}", flush=True)
    G = torch.stack(GA); Gn = G / G.norm(dim=1, keepdim=True)
    bn = float((Gn @ Gn.T)[~torch.eye(len(G), dtype=bool)].mean())
    res[name] = dict(batches=rows, cos_mean=float(np.mean([r["cos"] for r in rows])),
                     rel_diff_mean=float(np.mean([r["rel_diff"] for r in rows])), batch_cos_attached=bn)
    print(f"[{name}] mean cos(now, attached) {res[name]['cos_mean']:.4f}; mean |diff|/|attached| "
          f"{res[name]['rel_diff_mean']:.3f}; batch-to-batch cos (attached) {bn:.3f}", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(config=vars(args), results=res), open(args.out, "w"), indent=1)
print("[write]", args.out)

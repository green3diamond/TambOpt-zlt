"""Locate the non-finite values behind the noise-aware Stage 4 NaNs: for a saved layout (nan_guard_layout.pt /
nan_state.pt) and the step's own batch and noise draws, check every stage of utility_of_xy per sub-batch.

    python diag_nan.py <layout.pt> <window_ns> <lambda>
"""
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
path, W, lam = sys.argv[1], float(sys.argv[2]), sys.argv[3]
R = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k"
os.environ.update(TAMBOPT_UTILITY="two_tier", TAMBOPT_TWO_TIER_LAMBDA=lam,
                  TAMBOPT_TWO_TIER_CALIB=R + "/two_tier_calibration_kernel_5deg.json")
_s = importlib.util.spec_from_file_location("s4", os.path.join(REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
S4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(S4)
from modules.optimize.objective import primary_to_physical_labels, _sph  # noqa: E402
from modules.optimize.utility import angle_between_deg  # noqa: E402

dev = S4.DEVICE
L = torch.load(path, map_location="cpu")
epoch = int(L["epoch"])
x, y = L["x"].float().to(dev), L["y"].float().to(dev)
print(f"layout {path}: epoch {epoch}, positions finite {bool(torch.isfinite(x).all() and torch.isfinite(y).all())}")
mountain = S4.load_tr_mountain(S4.GEOMETRY_PATH_RESOLVED, S4.GEOMETRY_GROUP, S4.DET_KEY, east_entry=S4.EAST_ENTRY,
                               layer_east_dx=S4.LAYER_EAST_DX, n_planes=S4.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
recon = S4.load_recon(R + "/test_v6_run_03_recentered_deepsets", dev)
pool = S4.MainPool(mountain)
B, n_sweep = 4096, pool.n - 1024
a0 = int(np.random.default_rng([S4.SEED, epoch]).integers(0, n_sweep - B + 1))
clouds, bws, prim, frames = pool.load(a0, a0 + B)
g = torch.Generator(device=dev); g.manual_seed(int(S4.SEED * 1000003 + epoch))
xy = torch.stack([x, y], -1)[None]
tot = dict(k=0, kfin=0, nfin=0, rec=0, Einf=0, anan=0, enan=0, einf=0)
for lo in range(0, prim.shape[0], 32):
    hi = min(lo + 32, prim.shape[0])
    base = S4.BatchKernel([c[lo:hi].to(dev) for c in clouds], [w[lo:hi].to(dev) if w is not None else None for w in bws],
                          surface, differentiable=False, frames=None if frames is None else frames[lo:hi].to(dev))
    kern = S4.NoisyGated(base, W, "knn23", g, 27.7, straight_through=False)
    with torch.no_grad():
        p = prim[lo:hi].to(dev)
        out0 = base(p, xy.expand(hi - lo, -1, -1))
        out = kern(p, xy.expand(hi - lo, -1, -1))
        feats = torch.cat([xy.expand(hi - lo, -1, -1), out], -1)
        pred = recon(feats)
        E_p, th_p, ph_p = primary_to_physical_labels(pred)
        E_t, th_t, ph_t = primary_to_physical_labels(p)
        a = angle_between_deg(_sph(th_p, ph_p), _sph(th_t, ph_t))
        e = (torch.log10(E_p.clamp(min=1.0)) - torch.log10(E_t)).abs()
    tot["k"] += hi - lo
    tot["kfin"] += int((~torch.isfinite(out0)).any(-1).any(-1).sum())
    tot["nfin"] += int((~torch.isfinite(out)).any(-1).any(-1).sum())
    tot["rec"] += int((~torch.isfinite(pred)).any(-1).sum())
    tot["Einf"] += int((~torch.isfinite(E_p)).sum())
    tot["anan"] += int((~torch.isfinite(a)).sum())
    tot["enan"] += int(torch.isnan(e).sum()); tot["einf"] += int(torch.isinf(e).sum())
    if (~torch.isfinite(pred)).any() or (~torch.isfinite(E_p)).any():
        i = int(((~torch.isfinite(pred)).any(-1) | ~torch.isfinite(E_p)).nonzero()[0])
        print(f"  shower {lo + i}: pred {pred[i].tolist()}  E_pred {float(E_p[i]):.3g}  "
              f"max input E {float(torch.expm1(out[i, :, 0]).max()):.3g}  lit {int((out[i, :, 0] > 0).sum())}  "
              f"timed {int((out[i, :, 1] > 0).sum())}  log10E_true {float(torch.log10(E_t[i])):.2f}")
print("showers", tot["k"], "| kernel non-finite", tot["kfin"], "| after noise/gate", tot["nfin"], "| recon output",
      tot["rec"], "| E_pred inf", tot["Einf"], "| angle non-finite", tot["anan"], "| e NaN", tot["enan"], "inf", tot["einf"])

"""Gradient check of the layout utility with the analytical recon and the selected time model (TAMBOPT_TIME_MODEL),
Malata, uniform grid, CORSIKA 8 pool showers.

U is made deterministic in the layout by resetting the time draws (timing.reseed) before every evaluation.
  1. per detector (the most-hit ones), East and North: autograd vs central finite difference at several steps h,
     with the hard switches between the + and - evaluations counted: timed detectors, reconstructible showers
     (soft hit sum >= 2.5) and, for the edge model, lag-CDF grid bins;
  2. directional: moving every detector by delta along the normalised autograd gradient, and along its sign
     (Adam-like), the change of U vs the first-order prediction.

    python grad_check.py <out.json> [n_showers] [n_detectors]
Run under slurm/run20_guard.sh with TAMBOPT_UTILITY=swgo_trigger (TRIG_THETA=1, WIDTH=0.2, QREF=5).
"""
import importlib.util, json, os, sys, time
import numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING
from modules.optimize.objective import utility_of_xy
from modules.optimize import analytic_reco as AR
from modules.optimize.analytic_diff import AnalyticRecon

OUT = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 64; ND = int(sys.argv[3]) if len(sys.argv) > 3 else 5
dev = S4.DEVICE
print("time model", TIMING.MODEL, "device", dev, flush=True)
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, B, 4200)
clouds, _, prim, frames = pool.load(0, pool.n)
cl = [c.to(dev) for c in clouds]; p_ = prim.to(dev); f_ = frames.to(dev)
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = g["x"].float().to(dev), g["y"].float().to(dev)
n_plane = AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev)
recon = AnalyticRecon(surface, n_plane, kappa=10.0)


def U_of(e, n, grad=False):
    """U and the hard-switch state (timed, reconstructible, edge grid bins) of one evaluation"""
    TIMING.reseed(123)
    k = S4.BatchKernel(cl, [None] * len(cl), surface, differentiable=True, frames=f_)
    recon.bind(k, p_)
    with (torch.enable_grad() if grad else torch.no_grad()):
        U, _, _ = utility_of_xy(e, n, p_, k, recon)
    st = dict(mu=k.mu.detach(), bins=TIMING.LAST.get("k"), timed=k.last_out[..., 1] > 0)
    return U, st


# record the response's output for the switch counts
_call = S4.BatchKernel.__call__
def _wrapped(self, pb, xy):
    o = _call(self, pb, xy); self.last_out = o.detach(); return o
S4.BatchKernel.__call__ = _wrapped


def switches(a, b):
    ta, tb = a["timed"], b["timed"]
    ra = ((torch.sigmoid(10 * (a["mu"] - 1)) * ta).sum(-1) >= 2.5); rb = ((torch.sigmoid(10 * (b["mu"] - 1)) * tb).sum(-1) >= 2.5)
    d = dict(timed=int((ta != tb).sum()), recon=int((ra != rb).sum()))
    if a["bins"] is not None:
        d["bins"] = int((a["bins"] != b["bins"]).sum())
    return d


def state(e, n):
    U, st = U_of(e, n)
    return float(U), st


t0 = time.time()
e = e0.clone().requires_grad_(True); n = n0.clone().requires_grad_(True)
U, st0 = U_of(e, n, grad=True); U.backward()
gE, gN = e.grad.detach().clone(), n.grad.detach().clone()
U0 = float(U)
print(f"U = {U0:.4f}, |grad| = {float(torch.sqrt((gE**2 + gN**2).sum())):.4g}, {time.time() - t0:.0f}s", flush=True)
res = dict(model=TIMING.MODEL, n_showers=B, U=U0, per_detector=[], directional=[])

hits = (st0["mu"] >= 1).sum(0)
dets = hits.argsort(descending=True)[:ND].tolist()
for j in dets:
    for ax, nm, gj in ((0, "E", float(gE[j])), (1, "N", float(gN[j]))):
        row = dict(det=j, axis=nm, auto=gj, fd={}, sw={})
        for h in (0.1, 0.5, 2.0):
            d = torch.zeros_like(e0); d[j] = h
            up = state(e0 + d, n0) if ax == 0 else state(e0, n0 + d)
            dn = state(e0 - d, n0) if ax == 0 else state(e0, n0 - d)
            row["fd"][h] = (up[0] - dn[0]) / (2 * h); row["sw"][h] = switches(up[1], dn[1])
        res["per_detector"].append(row)
        print(f"det {j:3d} {nm}: auto {gj:+.5f}  fd " + "  ".join(f"h={h}:{v:+.5f} {row['sw'][h]}" for h, v in row["fd"].items()),
              flush=True)

gn = torch.sqrt(gE ** 2 + gN ** 2).clamp(min=1e-12)
for kind in ("grad", "sign"):
    if kind == "grad":
        dE, dN = gE / gn.max(), gN / gn.max()        # largest move = delta
    else:
        dE, dN = torch.sign(gE), torch.sign(gN)      # every detector moves delta per axis (Adam-like)
    pred1 = float((gE * dE + gN * dN).sum())
    for delta in (0.5, 1.0, 3.0):
        Ud = state(e0 + delta * dE, n0 + delta * dN)[0]
        res["directional"].append(dict(kind=kind, delta_m=delta, dU=Ud - U0, pred=delta * pred1))
        print(f"directional {kind} delta {delta} m: dU {Ud - U0:+.5f}  first-order {delta * pred1:+.5f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT, f"{time.time() - t0:.0f}s")

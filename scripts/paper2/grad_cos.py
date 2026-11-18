"""Is the layout gradient a shared direction or batch noise? Analytical recon, selected time model
(TAMBOPT_TIME_MODEL), Malata, uniform grid, CORSIKA 8 pool showers as in the optimization runs (C8Pool seed 4200).

Full-batch gradients of U (pool_utility, sub-batches of --chunk, as one Adam step of the run):
  gA  : showers [0, B),  time draws seed 1
  gA2 : showers [0, B),  time draws seed 2      -> cos(gA, gA2): effect of the time noise alone
  gB  : showers [B, 2B), time draws seed 1      -> cos(gA, gB):  effect of the shower sample
Reported: cosine of the 200-vectors, sign agreement of the coordinates (what Adam follows), per-detector 2D cosine
(median over the detectors), and the same for the half-batches [0, B/2) vs [B/2, B) to see the trend with N.

    python grad_cos.py <out.json> [B] [chunk]
"""
import importlib.util, json, os, sys, time
import numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING
from modules.optimize import analytic_reco as AR
from modules.optimize.analytic_diff import AnalyticRecon

OUT = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 5120; CH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = S4.DEVICE
print("time model", TIMING.MODEL, "B", B, flush=True)
t0 = time.time()
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, 2 * B, 4200)
print(f"pool {pool.n} showers, {time.time() - t0:.0f}s", flush=True)
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = g["x"].float(), g["y"].float()
n_plane = AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev)
recon = AnalyticRecon(surface, n_plane, kappa=10.0)


def grad(a, b, seed):
    TIMING.reseed(seed)
    xy = S4.LearnableXY(e0.clone(), n0.clone(), device=str(dev)).to(dev)
    batch = pool.load(a, b)
    tot = S4.pool_utility(xy, batch, surface, recon, CH, backward=True)
    gr = torch.cat([p.grad.detach().flatten() for p in xy.parameters()]).cpu()
    print(f"  [{a}, {b}) seed {seed}: U {tot['U']:.4f}, |g| {float(gr.norm()):.4g}, {time.time() - t0:.0f}s", flush=True)
    return gr, tot["U"]


def compare(x, y):
    nd = len(x) // 2
    cos = float(torch.dot(x, y) / (x.norm() * y.norm()))
    sign = float((torch.sign(x) == torch.sign(y)).float().mean())
    X, Y = x.view(2, nd).T, y.view(2, nd).T                      # (nd, 2): per-detector (E, N)
    ok = (X.norm(dim=1) > 0) & (Y.norm(dim=1) > 0)
    pd = torch.nn.functional.cosine_similarity(X[ok], Y[ok], dim=1)
    return dict(cos=cos, sign_agree=sign, det_cos_median=float(pd.median()), det_cos_mean=float(pd.mean()),
                n_det=int(ok.sum()))


gA, uA = grad(0, B, 1); gA2, uA2 = grad(0, B, 2); gB, uB = grad(B, 2 * B, 1)
gH1, _ = grad(0, B // 2, 1); gH2, _ = grad(B // 2, B, 1)
res = dict(model=TIMING.MODEL, B=B, U=dict(A=uA, A_seed2=uA2, B=uB),
           time_noise=compare(gA, gA2), showers=compare(gA, gB), half_batches=compare(gH1, gH2))
for k in ("time_noise", "showers", "half_batches"):
    print(k, res[k], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT, f"{time.time() - t0:.0f}s")

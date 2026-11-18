"""Gradient agreement of the paper-2 Malata optimizations (no analytical recon): kernel response (default time model),
DeepSets recon, utility from the environment (TAMBOPT_UTILITY etc., as the runs), uniform grid. Two disjoint batches
per size B; cosine of the 200-vectors, sign agreement, per-detector 2D cosine (as scripts/paper2/grad_cos.py).

  --pool fm : the run-20 flow-matching pool (MainPool, the runs' showers), recon = the run's DeepSets
  --pool c8 : CORSIKA 8 training showers (C8Pool seed 4200), recon = the C8 fine-tuned DeepSets (tmp_rings/c8ft/ft20)

    python grad_cos_ds.py <out.json> --pool fm|c8 [--bs 512,5120] [--chunk 64] [--recon DIR]
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np, torch

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--pool", choices=("fm", "c8"), required=True)
ap.add_argument("--bs", default="512,5120"); ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--recon", default=None)
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C

BS = [int(b) for b in args.bs.split(",")]
dev = S4.DEVICE
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
N = 2 * max(BS)
pool = S4.C8Pool(m, N, 4200) if args.pool == "c8" else S4.MainPool(m)
rdir = args.recon or ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/c8ft/ft20" if args.pool == "c8"
                      else C.RECON_FOLDER + "_deepsets")
recon = S4.load_recon(rdir, dev)
log(f"pool {args.pool} ({pool.n} showers), recon {rdir}, utility {S4.UTILITY_MODE}, B {BS}")
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = torch.as_tensor(g["x"]).float(), torch.as_tensor(g["y"]).float()


def grad(a, b):
    xy = S4.LearnableXY(e0.clone(), n0.clone(), device=str(dev)).to(dev)
    tot = S4.pool_utility(xy, pool.load(a, b), surface, recon, args.chunk, backward=True)
    gr = torch.cat([p.grad.detach().flatten() for p in xy.parameters()]).cpu()
    log(f"  [{a}, {b}): U {tot['U']:.4f}, |g| {float(gr.norm()):.4g}, finite {bool(torch.isfinite(gr).all())}")
    return gr, tot["U"]


def compare(x, y):
    nd = len(x) // 2
    X, Y = x.view(2, nd).T, y.view(2, nd).T
    ok = (X.norm(dim=1) > 0) & (Y.norm(dim=1) > 0)
    pd = torch.nn.functional.cosine_similarity(X[ok], Y[ok], dim=1)
    return dict(cos=float(torch.dot(x, y) / (x.norm() * y.norm())),
                sign_agree=float((torch.sign(x) == torch.sign(y)).float().mean()),
                det_cos_median=float(pd.median()) if ok.any() else None, n_det=int(ok.sum()))


res = dict(pool=args.pool, recon=rdir, chunk=args.chunk,
           utility={k: v for k, v in os.environ.items() if k.startswith("TAMBOPT_") and ("TRIG" in k or "UTIL" in k
                                                                                        or "TWO_TIER" in k or "TIME" in k)},
           by_B={})
for B in BS:
    gA, uA = grad(0, B); gB, uB = grad(B, 2 * B)
    res["by_B"][str(B)] = dict(U=[uA, uB], g_norm=[float(gA.norm()), float(gB.norm())], **compare(gA, gB))
    log(f"B {B}: {res['by_B'][str(B)]}")
    json.dump(res, open(args.out, "w"), indent=1)
log("done")

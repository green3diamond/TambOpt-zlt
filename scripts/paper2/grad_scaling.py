"""Does a bigger corpus make the batch gradients agree? Gradient of the utility at the uniform grid on consecutive
disjoint chunks of C showers (setup of grad_cos_k.py: pool, recon, utility and time model from the environment),
saved per chunk so that a batch of m chunks is the size-weighted mean of their gradients. Resumable: chunks already
in <out.pt> are skipped.

    python grad_scaling.py <out.pt> --pool fm|c8 [--recon analytic|DIR] [--chunk-showers 5000] [--n-chunks 0=all]
    python grad_scaling.py --analyse <out.pt> ...      # cos(B) table and the fit cos = B / (B + B*)
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np, torch

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")


def batch_grads(G, n, m):
    """consecutive groups of m chunks -> size-weighted mean gradients"""
    k = len(G) // m
    return [(G[i * m:(i + 1) * m] * n[i * m:(i + 1) * m, None]).sum(0) / n[i * m:(i + 1) * m].sum() for i in range(k)], \
        [float(n[i * m:(i + 1) * m].sum()) for i in range(k)]


def analyse(paths):
    from scipy.optimize import curve_fit
    out = {}
    for p in paths:
        d = torch.load(p, weights_only=False)
        G = torch.stack(d["g"]).double(); n = torch.tensor(d["n"], dtype=torch.float64)
        rows = []
        for m in [1, 2, 4, 6, 10]:                   # B up to 50k (5,000-shower chunks)
            gs, ns = batch_grads(G, n, m)
            if len(gs) < 2:
                break
            gs = torch.stack(gs)
            c = torch.nn.functional.cosine_similarity(gs[:, None], gs[None], dim=-1)
            iu = torch.triu_indices(len(gs), len(gs), 1)
            cs = c[iu[0], iu[1]].numpy()
            rest = [float(torch.nn.functional.cosine_similarity(gs[i], torch.cat([gs[:i], gs[i + 1:]]).mean(0), dim=0))
                    for i in range(len(gs))]
            rows.append(dict(B=float(np.mean(ns)), K=len(gs), cos_mean=float(cs.mean()),
                             cos_sd=float(cs.std(ddof=1)) if len(cs) > 1 else float("nan"),
                             cos_to_rest=float(np.mean(rest)), g_norm_median=float(gs.norm(dim=1).median())))
        B = np.array([r["B"] for r in rows]); cm = np.array([r["cos_mean"] for r in rows])
        try:
            (bstar,), cov = curve_fit(lambda b, s: b / (b + s), B, cm, p0=[1e4], bounds=(1.0, 1e9))
            fit = dict(B_star=float(bstar), B_star_err=float(np.sqrt(cov[0, 0])),
                       B_for_cos_half=float(bstar), B_for_cos_0p8=float(4 * bstar))
        except Exception as e:  # noqa: BLE001
            fit = dict(error=str(e))
        # per-chunk norm spread: heavy tails mean outlier showers, not Gaussian scatter
        gn = G.norm(dim=1).numpy()
        out[os.path.basename(p)] = dict(meta=d["meta"], rows=rows, fit=fit, n_chunks=len(G),
                                        chunk_g_norm_quantiles=np.quantile(gn, [0, .25, .5, .75, 1]).round(4).tolist())
        print(f"\n## {os.path.basename(p)}  ({d['meta']['recon']}, {len(G)} chunks of {d['meta']['chunk_showers']})")
        print("| B | K | cos (pairs) | cos to mean of others | median |g| |\n|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['B']:.0f} | {r['K']} | {r['cos_mean']:.3f} ± {r['cos_sd']:.3f} | {r['cos_to_rest']:.3f} | "
                  f"{r['g_norm_median']:.3g} |")
        print("fit:", fit, " chunk |g| quantiles:", out[os.path.basename(p)]["chunk_g_norm_quantiles"])
    return out


if len(sys.argv) > 1 and sys.argv[1] == "--analyse":
    res = analyse(sys.argv[2:])
    json.dump(res, open(os.path.splitext(sys.argv[2])[0] + "_analysis.json" if len(sys.argv) == 3 else
                        os.path.join(os.path.dirname(sys.argv[2]), "analysis.json"), "w"), indent=1)
    sys.exit()

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--pool", choices=("fm", "c8"), required=True)
ap.add_argument("--recon", default=None, help="DeepSets folder, or 'analytic'")
ap.add_argument("--chunk-showers", type=int, default=5000); ap.add_argument("--n-chunks", type=int, default=0)
ap.add_argument("--chunk", type=int, default=64, help="GPU sub-batch (memory only)")
ap.add_argument("--c8-n", type=int, default=51_200, help="C8Pool size (held in memory: 51,200 needs ~120 GB)")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING

dev = S4.DEVICE
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, args.c8_n, 4200) if args.pool == "c8" else S4.MainPool(m)
rdir = args.recon or ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/c8ft/ft20" if args.pool == "c8"
                      else C.RECON_FOLDER + "_deepsets")
if rdir == "analytic":
    from modules.optimize import analytic_reco as AR
    from modules.optimize.analytic_diff import AnalyticRecon
    recon = AnalyticRecon(surface, AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev),
                          kappa=10.0)
    rdir = f"analytic ({AR.MODEL})"
else:
    recon = S4.load_recon(rdir, dev)
CS = args.chunk_showers
K = pool.n // CS if not args.n_chunks else min(args.n_chunks, pool.n // CS)
meta = dict(pool=args.pool, pool_n=int(pool.n), recon=rdir, chunk_showers=CS, utility=S4.UTILITY_MODE,
            env={k: v for k, v in os.environ.items() if k.startswith("TAMBOPT_")})
log(f"pool {args.pool} ({pool.n} showers), recon {rdir}, utility {S4.UTILITY_MODE}, {K} chunks of {CS}")
d = torch.load(args.out, weights_only=False) if os.path.exists(args.out) else dict(meta=meta, g=[], n=[], U=[], logE=[])
g0 = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = torch.as_tensor(g0["x"]).float(), torch.as_tensor(g0["y"]).float()
for i in range(len(d["g"]), K):
    a, b = i * CS, (i + 1) * CS
    TIMING.reseed(1 + i)                                  # one time-noise draw per chunk
    xy = S4.LearnableXY(e0.clone(), n0.clone(), device=str(dev)).to(dev)
    batch = pool.load(a, b)
    tot = S4.pool_utility(xy, batch, surface, recon, args.chunk, backward=True)
    gr = torch.cat([p.grad.detach().flatten() for p in xy.parameters()]).cpu()
    d["g"].append(gr); d["n"].append(int(batch[2].shape[0])); d["U"].append(float(tot["U"]))
    d["logE"].append(float(batch[2][:, 3].float().mean()) if batch[2].shape[1] > 3 else float("nan"))
    torch.save(d, args.out)
    log(f"  chunk {i} [{a}, {b}): n {d['n'][-1]}  U {tot['U']:.4f}  |g| {float(gr.norm()):.4g}  "
        f"finite {bool(torch.isfinite(gr).all())}")
log("done")

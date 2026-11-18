"""Is the layout gradient a shared direction once the response is the smooth expected response of Jay's symbolic model
(modules/optimize/expected_response.py) and the batches are large? Same test as scripts/paper2/grad_cos.py (kernel:
cosine 0.07 between two 5,120-shower batches), with the expected response in place of the kernel.

Primaries: the run-20 training corpus (decay vertex, direction, energy, EM/hadronic class), events drawn without
replacement (seed 4200); per batch size B, showers [0, B) and [B, 2B) of that draw. Utility as the runs:
TAMBOPT_UTILITY=swgo_trigger, TAMBOPT_TRIG_THETA=1, TAMBOPT_TRIG_WIDTH=0.2, TAMBOPT_TRIG_QREF=5 (set by the job),
analytical recon (AnalyticRecon, kappa 10), Malata uniform grid. Full-batch gradients by sub-batches of --chunk.
Reported per B: U of each batch, |g|, cosine of the 200-vectors, sign agreement, per-detector 2D cosine.

    python grad_cos_expected.py <out.json> [B list, default 5120,51200] [chunk 256] [table workers 16]
"""
import importlib.util, json, os, sys, time
from multiprocessing import Pool
import numpy as np, torch

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
from modules.data.dataset_builder import _load_corpus_metadata, shower_frames
from modules.optimize import analytic_reco as AR
from modules.optimize import expected_response as XR
from modules.optimize.analytic_diff import AnalyticRecon

OUT = sys.argv[1]
BS = [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "5120,51200").split(",")]
CH = int(sys.argv[3]) if len(sys.argv) > 3 else 256
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
TABLE_CHUNK = 1000
dev = S4.DEVICE
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
log(f"B {BS}, chunk {CH}, utility {os.environ.get('TAMBOPT_UTILITY')}")

m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
meta = _load_corpus_metadata(m, C.DUAL_SHOWER_CACHE_PATH, 0)
n_ev = meta.k_sp                                                    # events = the first species block
N = 2 * max(BS)
if N > n_ev:
    raise ValueError(f"need {N} events, the corpus has {n_ev}")
sel = np.sort(np.random.default_rng(4200).choice(n_ev, N, replace=False))
dirs = meta.dirs[sel].double(); pos = meta.positions[sel].double(); prim = meta.primaries[sel].float()
log10_e = (prim[:, 3] * (C.LOG_E_MAX - C.LOG_E_MIN) + C.LOG_E_MIN).numpy().astype(np.float64)
cls = prim[:, 4].round().long().numpy()
frames = shower_frames(pos, dirs)
log(f"{N} of {n_ev} events; classes {np.bincount(cls).tolist()}; log10 E {log10_e.min():.2f}-{log10_e.max():.2f}")

PART1, LAWS = XR.load_model()


def _tables(a):
    b = min(a + TABLE_CHUNK, N)
    return a, XR.tables(PART1, LAWS, log10_e[a:b], dirs[a:b].numpy(), cls[a:b], seed=a)


logd = np.empty((N, len(C.SPECIES_NAMES), C.N_PLANES, len(XR.ring_geometry()[0])), np.float32)
tconst = np.empty((N, len(C.SPECIES_NAMES), 3), np.float32)
with Pool(WORKERS) as pool:
    for a, (ld, tc) in pool.imap_unordered(_tables, range(0, N, TABLE_CHUNK)):
        logd[a:a + len(ld)], tconst[a:a + len(tc)] = ld, tc
log(f"tables {logd.shape} ({logd.nbytes / 1e9:.1f} GB)")
logd_t, tconst_t = torch.as_tensor(logd), torch.as_tensor(tconst)

g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = torch.as_tensor(g["x"]).float(), torch.as_tensor(g["y"]).float()
n_plane = AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev)
recon = AnalyticRecon(surface, n_plane, kappa=10.0)


def grad(a, b):
    """Full-batch gradient of U over events [a, b) at the uniform grid -> (200-vector, U, mean hits)."""
    xy = S4.LearnableXY(e0.clone(), n0.clone(), device=str(dev)).to(dev)
    U_tot, hits = 0.0, 0.0
    for lo in range(a, b, CH):
        hi = min(lo + CH, b)
        resp = XR.ExpectedResponse(logd_t[lo:hi].to(dev), tconst_t[lo:hi].to(dev), frames[lo:hi].float().to(dev),
                                   surface)
        x, y = xy()
        p = prim[lo:hi].to(dev)
        recon.bind(resp, p)
        U, _, _ = S4.utility_of_xy(x, y, p, resp, recon)
        share = (hi - lo) / (b - a)
        (-share * U).backward()
        U_tot += share * float(U)
        hits += share * float((resp.mu >= 1).sum(1).float().mean())
    gr = torch.cat([q.grad.detach().flatten() for q in xy.parameters()]).cpu()
    log(f"  [{a}, {b}): U {U_tot:.4f}, hits/shower {hits:.1f}, |g| {float(gr.norm()):.4g}, finite {bool(torch.isfinite(gr).all())}")
    return gr, U_tot, hits


def compare(x, y):
    nd = len(x) // 2
    X, Y = x.view(2, nd).T, y.view(2, nd).T
    ok = (X.norm(dim=1) > 0) & (Y.norm(dim=1) > 0)
    pd = torch.nn.functional.cosine_similarity(X[ok], Y[ok], dim=1)
    return dict(cos=float(torch.dot(x, y) / (x.norm() * y.norm())),
                sign_agree=float((torch.sign(x) == torch.sign(y)).float().mean()),
                det_cos_median=float(pd.median()), det_cos_mean=float(pd.mean()), n_det=int(ok.sum()))


res = dict(response="expected (symbolic model)", model=XR.MODEL, chunk=CH, n_events=N,
           utility={k: os.environ.get(k) for k in ("TAMBOPT_UTILITY", "TAMBOPT_TRIG_THETA", "TAMBOPT_TRIG_WIDTH",
                                                    "TAMBOPT_TRIG_QREF")}, by_B={})
for B in BS:
    gA, uA, hA = grad(0, B)
    gB, uB, hB = grad(B, 2 * B)
    res["by_B"][str(B)] = dict(U=[uA, uB], hits_per_shower=[hA, hB], g_norm=[float(gA.norm()), float(gB.norm())],
                               **compare(gA, gB))
    log(f"B {B}: {res['by_B'][str(B)]}")
    json.dump(res, open(OUT, "w"), indent=1)
log("done")

"""Calibration and checks of the detector time model (modules/optimize/timing.py, TAMBOPT_TIME_MODEL=lead), Malata.

1. Raw CORSIKA 8 (training files combined_<species>.h5): each 10 m pixel's lag behind its plane's front,
   t - (l + 1) 500 m / c, vs distance r from the shower axis (per layer, the energy-weighted centroid); per r bin the
   energy-weighted mean and spread of log lag and of lag; Linsley fit of the lag spread, sigma(r) = s0 (1 + r/rt)^b.
2. Kernel (the c8 pool's placement, uniform grid): per timed detector r, the moments m, s, the model T - t_plane,
   and the default T - t_plane; binned like 1, so the kernel moments can be compared with the raw pixels.
3. Gradient check (analytical recon, swgo_trigger 1 deg utility, 64 showers): autograd vs central finite
   difference of U in East and North for 3 detectors, the time draws reset before every evaluation.

    python fit_delays.py <out_dir> [n_raw_showers] [n_pool]
Run under slurm/run20_guard.sh with TAMBOPT_UTILITY=swgo_trigger, TAMBOPT_TIME_MODEL=mean (or lead).
"""
import importlib.util, json, os, sys
import numpy as np, torch, h5py
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING
from modules.data.dataset_builder import compute_labels_batch
from modules.optimize.background import axis_distance
from modules.optimize.objective import utility_of_xy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

assert TIMING.enabled(), "set TAMBOPT_TIME_MODEL=mean or lead"
OUT = sys.argv[1]; NRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 400; NPOOL = int(sys.argv[3]) if len(sys.argv) > 3 else 200
os.makedirs(OUT, exist_ok=True)
EDGES = np.array([0, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800])
RC = 0.5 * (EDGES[1:] + EDGES[:-1])
CNS = 0.299792458                                                   # m / ns
res = {"edges_m": EDGES.tolist()}


def binned(r, x, w):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (r >= lo) & (r < hi)
        if w[k].sum() <= 0:
            out.append([np.nan, np.nan, 0]); continue
        mu = np.average(x[k], weights=w[k]); sd = np.sqrt(np.average((x[k] - mu) ** 2, weights=w[k]))
        out.append([float(mu), float(sd), int(k.sum())])
    return np.array(out)


# ── 1. raw CORSIKA 8 pixels ───────────────────────────────────────────────────
raw = {}
for sp in C.SPECIES_NAMES:
    with h5py.File(f"{S4.C8_DIR}/{S4.C8_FILES[sp]}", "r") as f:
        ds = f["showers"]; P = int(f["shape"][1])
        idx = np.random.default_rng(0).choice(len(ds), NRAW, replace=False)
        rr, ll, lag, ww = [], [], [], []
        for i in sorted(idx.tolist()):
            a = np.asarray(ds[i], dtype=np.float64).reshape(-1, 5); a = a[a[:, 3] > 0]
            for L in np.unique(a[:, 2]):
                b = a[a[:, 2] == L]
                cx, cy = np.average(b[:, 0], weights=b[:, 3]), np.average(b[:, 1], weights=b[:, 3])
                rr.append(np.hypot(b[:, 0] - cx, b[:, 1] - cy))
                lag.append(np.clip(b[:, 4] * 1e9 - (L + 1) * 500.0 / CNS, 0.1, None)); ww.append(b[:, 3])
    r, lg, w = map(np.concatenate, (rr, lag, ww))
    bl, bg = binned(r, np.log(lg), w), binned(r, lg, np.ones_like(lg))     # lag: pixel mean (kernel weight only)
    ok = (bg[:, 2] > 20) & np.isfinite(bg[:, 1])
    try:
        p, _ = curve_fit(lambda x, s0, rt, b: s0 * (1 + x / rt) ** b, RC[ok], bg[ok, 1], p0=(2.0, 30.0, 1.5),
                         bounds=([1e-3, 1.0, 0.1], [1e3, 1e3, 4.0]), maxfev=20000)
    except Exception as e:                                            # keep the binned values
        p = [np.nan] * 3; print(sp, "Linsley fit failed:", e)
    raw[sp] = dict(loglag=bl.tolist(), lag=bg.tolist(), linsley=dict(s0_ns=p[0], rt_m=p[1], b=p[2]))
    print(f"[raw] {sp}: Linsley s0 {p[0]:.2f} ns, rt {p[1]:.1f} m, b {p[2]:.2f}", flush=True)
res["raw"] = raw

# ── 2. kernel on the c8 pool, uniform grid ───────────────────────────────────
dev = S4.DEVICE
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, NPOOL, 4200)
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = g["x"].float().to(dev), g["y"].float().to(dev)
clouds, bws, prim, frames = pool.load(0, pool.n)
fr = frames.to(dev)
with torch.no_grad():
    kern = S4.BatchKernel([c.to(dev) for c in clouds], bws, surface, differentiable=False, frames=fr)
    xy = torch.stack([e0, n0], -1)[None].expand(pool.n, -1, -1)
    out = kern(prim.to(dev), xy)
    mus, moms, Tmin = [], [], []
    for c in clouds:
        mom = {}
        E, T = compute_labels_batch(c.to(dev), e0, n0, surface, frames=fr, moments=mom)
        moms.append(mom); Tmin.append(T)
    r = axis_distance(e0, n0, surface, fr).cpu().numpy()
    mu = kern.mu.cpu().numpy()
    Tlead = torch.expm1(out[..., 1].clamp(min=0)).cpu().numpy() * 1e3          # ns (T_LOG_SCALE: us)
    tpl = moms[0]["t_plane"].cpu().numpy() * 1e9
    Told = torch.stack([torch.where(t > 0, t, torch.full_like(t, np.inf)) for t in Tmin]).amin(0).cpu().numpy() * 1e9
timed = Tlead > 0
mk = timed & (mu >= 1)
kres = {}
for nm, x in [("model_lag", Tlead - tpl), ("old_lag", Told - tpl)]:
    kres[nm] = binned(r[mk], x[mk], np.ones(mk.sum())).tolist()
for si, sp in enumerate(C.SPECIES_NAMES):
    mm, ss, ww = (moms[si][k].cpu().numpy() for k in ("m", "s", "w"))
    k2 = mk & (ww > 0)
    kres[f"m_{sp}"] = binned(r[k2], mm[k2], ww[k2]).tolist()
    kres[f"s_{sp}"] = binned(r[k2], ss[k2], ww[k2]).tolist()
    gm, gv = (moms[si][k].cpu().numpy() for k in ("mean", "var"))
    kres[f"mean_{sp}"] = binned(r[k2], gm[k2], np.ones(k2.sum())).tolist()
    kres[f"sd_{sp}"] = binned(r[k2], np.sqrt(gv[k2]), np.ones(k2.sum())).tolist()
res["kernel"] = kres
print("[kernel]", TIMING.MODEL, "timed detectors", int(mk.sum()), "old lag p50/p90 [ns]",
      np.percentile((Told - tpl)[mk], [50, 90]).round(2).tolist(), "model lag p50/p90",
      np.percentile((Tlead - tpl)[mk], [50, 90]).round(2).tolist(), flush=True)

# ── 3. gradient check (analytical recon) ─────────────────────────────────────
from modules.optimize import analytic_reco as AR
from modules.optimize.analytic_diff import AnalyticRecon
n_plane = AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev)
recon = AnalyticRecon(surface, n_plane, kappa=10.0)
B = 64
cl = [c[:B].to(dev) for c in clouds]; p_ = prim[:B].to(dev); f_ = fr[:B]


def U_of(e, n):
    TIMING.reseed(123)
    k = S4.BatchKernel(cl, [None] * len(cl), surface, differentiable=True, frames=f_)
    recon.bind(k, p_)
    U, _, _ = utility_of_xy(e, n, p_, k, recon)
    return U


e = e0.clone().requires_grad_(True); n = n0.clone().requires_grad_(True)
U = U_of(e, n); U.backward()
hit = torch.as_tensor(mu[:B] >= 1).sum(0).argsort(descending=True)[:3].tolist()
gc = []
h = 0.5
for j in hit:
    row = {"det": j, "auto_E": float(e.grad[j]), "auto_N": float(n.grad[j])}
    for ax, nm in ((0, "E"), (1, "N")):
        d = torch.zeros_like(e0); d[j] = h
        with torch.no_grad():
            up = U_of(e0 + d, n0) if ax == 0 else U_of(e0, n0 + d)
            dn = U_of(e0 - d, n0) if ax == 0 else U_of(e0, n0 - d)
        row[f"fd_{nm}"] = float((up - dn) / (2 * h))
    gc.append(row); print("[grad]", row, flush=True)
res["grad_check"] = dict(U=float(U), h_m=h, rows=gc)

json.dump(res, open(f"{OUT}/fit_delays.json", "w"), indent=1, default=float)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for sp in C.SPECIES_NAMES:
    a = np.array(raw[sp]["lag"]); l = np.array(raw[sp]["loglag"])
    ax[0].plot(RC, a[:, 0], "o-", ms=3, label=f"raw {sp} mean")
    ax[1].plot(RC, a[:, 1], "o-", ms=3, label=f"raw {sp}")
    km = np.array(kres[f"mean_{sp}"]); ks = np.array(kres[f"sd_{sp}"])
    ax[0].plot(RC, km[:, 0], "x--", ms=4, label=f"kernel group mean {sp}")
    ax[2].plot(RC, a[:, 1], "o-", ms=3, label=f"raw {sp}")
    ax[2].plot(RC, ks[:, 0], "x--", ms=4, label=f"kernel group sd {sp}")
    L = raw[sp]["linsley"]
    if np.isfinite(L["s0_ns"]):
        xx = np.linspace(1, 800, 200); ax[1].plot(xx, L["s0_ns"] * (1 + xx / L["rt_m"]) ** L["b"], "k:", lw=0.6)
for nm, st in (("model_lag", "s-"), ("old_lag", "d-")):
    b = np.array(kres[nm]); ax[0].plot(RC, b[:, 0], st, ms=3, label=f"kernel T-t_plane ({nm.split('_')[0]})")
ax[0].set(xscale="log", yscale="symlog", xlabel="r [m]", ylabel="lag [ns]", title="mean lag")
ax[1].set(xscale="log", yscale="log", xlabel="r [m]", ylabel="spread [ns]", title="lag spread (raw) + Linsley fits")
ax[2].set(xscale="log", yscale="log", xlabel="r [m]", ylabel="lag spread [ns]", title="raw pixel vs kernel group spread")
for a in ax: a.legend(fontsize=6)
fig.savefig(f"{OUT}/fit_delays.png", dpi=120)
print("wrote", OUT)

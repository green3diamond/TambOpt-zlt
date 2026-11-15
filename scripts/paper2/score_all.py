"""Score layouts on the run's held-out showers (noise-free kernel, expected counts, default recon) with Stage 4's own
response (BatchKernel) and utility code (modules/optimize/objective.utility_of_xy), one kernel pass per layout.

Per layout: share of ALL showers within 1 / 2.5 / 5 deg, median angular error, |dlog10 E| < 0.2 share, detected share
(>= 3 detectors with E > 1), expected Poisson trigger rate (2 m^2, k = 3), and the utilities
  orig (TAMBOPT_UTILITY=current), twotier_l0p3, twotier_l0 (5 deg), threetier, trig5_poisson, trig5_step,
  trig1_poisson (1 deg / 0.2 / 5); bounded utilities divided by TWO_TIER_SCALE (so in [0, 1]).

    python score_all.py <out.json> <n_events> name=layout.pt ...
Run under the run's guard (run 20: slurm/run20_guard.sh).
"""
import importlib.util, json, os, sys
os.environ["TAMBOPT_UTILITY"] = "swgo_trigger"            # BatchKernel then also computes the expected counts mu
os.environ.setdefault("TAMBOPT_TWO_TIER_CALIB", "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
                      "detector_optimization_v6/19_plane45S_12km_showerframe_400k/two_tier_calibration_kernel_5deg.json")
import numpy as np, torch
R = "/n/home05/zdimitrov/tambo/TambOpt"; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.optimize.objective as O
import modules.constants as C
from modules.data.dataset_builder import _load_corpus_metadata
from modules.optimize.counting import trigger_prob

out_path, n_ev, pairs = sys.argv[1], int(sys.argv[2]), [a.split("=", 1) for a in sys.argv[3:]]
dev = S4.DEVICE
mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
recon = S4.load_recon(os.environ.get("RECON_DIR", C.RECON_FOLDER + "_deepsets"), dev)


class HeldOut(S4.MainPool):
    def __init__(self):
        import showerdata
        self.path = C.HELDOUT_SHOWER_CACHE_PATH
        self.meta = _load_corpus_metadata(mountain, self.path, 0)
        self.n, self.bw = self.meta.k_sp, None

    def clouds(self, s, a, b, with_frames=False):
        import showerdata
        m = self.meta
        pts = showerdata.load(self.path, start=s * m.per_sp + a, stop=s * m.per_sp + b).points
        c = torch.as_tensor(pts, dtype=torch.float32)
        fr = S4.place_clouds_shower(c, m.positions[a:b], m.dirs[a:b], event_ids=torch.arange(a, b), species=s)
        return (c, fr) if with_frames else c


def corsika_events(max_deg=3.0):
    """CORSIKA 8 test showers (Hamza's combined_<species>_test.h5), each placed at the decay vertex of the run's tau
    with the closest direction (kept within max_deg); own direction and energy. Returns clouds, bws, prim, frames."""
    import h5py
    H = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3"
    FILES = {"electron": "combined_electrons_test.h5", "muon": "combined_muons_test.h5",
             "photon": "combined_photons_test.h5"}
    with h5py.File(f"{H}/{FILES['electron']}", "r") as f:
        dirs = torch.as_tensor(f["directions"][:], dtype=torch.float32); en = torch.as_tensor(f["energies"][:, 0])
        pdg = torch.as_tensor(f["pdg"][:].astype(np.int64))
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    with h5py.File(C.TAU_WHOLESKY_PATH, "r") as f:
        tpos = np.asarray(f["position"][...]).T; tdir = np.asarray(f["direction"][...]).T; te = np.asarray(f["energy"][...])
    k = (te >= 10 ** C.LOG_E_MIN) & (te <= 10 ** C.LOG_E_MAX)
    tpos = torch.as_tensor(tpos[k], dtype=torch.float32)
    tdir = torch.as_tensor(tdir[k], dtype=torch.float32).to(dev); tdir = tdir / tdir.norm(dim=1, keepdim=True)
    d = dirs.to(dev); bc = torch.full((len(d),), -2.0, device=dev); bi = torch.zeros(len(d), dtype=torch.long, device=dev)
    for lo in range(0, tdir.shape[0], 200_000):
        c, j = (d @ tdir[lo:lo + 200_000].T).max(1)
        u = c > bc; bc[u] = c[u]; bi[u] = j[u] + lo
    ang = torch.rad2deg(torch.arccos(bc.clamp(-1, 1))).cpu()
    keep = torch.nonzero(ang < max_deg).squeeze(1)
    keep = keep[:n_ev]
    pos = tpos[bi.cpu()[keep]]
    print(f"CORSIKA: {len(keep)} of {len(d)} test showers within {max_deg} deg of a tau direction", flush=True)
    centre = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32).mean(0)
    from modules.surrogates import encode_primary
    prim = encode_primary(dirs[keep], en[keep].float(), pdg[keep], pos, centre)
    clouds, frames = [], None
    for si, sp in enumerate(C.SPECIES_NAMES):
        with h5py.File(f"{H}/{FILES[sp]}", "r") as f:
            P = int(f["shape"][1])
            arr = np.zeros((len(keep), P, 5), np.float32)
            for r, i in enumerate(keep.tolist()):
                a = np.asarray(f["showers"][i], dtype=np.float32).reshape(-1, 5); arr[r, :len(a)] = a
        c = torch.as_tensor(arr)
        fr = S4.place_clouds_shower(c, pos, dirs[keep], event_ids=keep, species=si)
        clouds.append(c); frames = fr
    return clouds, [None] * len(clouds), prim, frames


if os.environ.get("SRC") == "c8":
    clouds, bws, prim, frames = corsika_events()
else:
    pool = HeldOut()
    n_ev = min(n_ev, pool.n)
    clouds, bws, prim, frames = pool.load(0, n_ev)
N = prim.shape[0]
print(f"{N} held-out showers", flush=True)
CH = 32


class Cached:
    def __init__(self, out, mu, gate_mask=None):
        self.out, self.mu, self.gate_mask = out, mu, gate_mask

    def __call__(self, p, xy):
        return self.out


MODES = {
    "orig": dict(UTILITY_MODE="current"),
    "twotier_l0p3": dict(UTILITY_MODE="two_tier", TWO_TIER_LAMBDA=0.3),
    "twotier_l0": dict(UTILITY_MODE="two_tier", TWO_TIER_LAMBDA=0.0),
    "threetier": dict(UTILITY_MODE="three_tier"),
    "trig5_poisson": dict(UTILITY_MODE="swgo_trigger", TWO_TIER_LAMBDA=0.3, TRIG_MODE="poisson", TRIG_THETA=5.0,
                          TRIG_WIDTH=1.0, TRIG_QREF=20.0),
    "trig5_step": dict(UTILITY_MODE="swgo_trigger", TWO_TIER_LAMBDA=0.3, TRIG_MODE="step", TRIG_THETA=5.0,
                       TRIG_WIDTH=1.0, TRIG_QREF=20.0),
    "trig1_poisson": dict(UTILITY_MODE="swgo_trigger", TWO_TIER_LAMBDA=0.3, TRIG_MODE="poisson", TRIG_THETA=1.0,
                          TRIG_WIDTH=0.2, TRIG_QREF=5.0),
}

res = json.load(open(out_path)) if os.path.exists(out_path) else {}
# BG_NS > 0: cosmic background in one readout window (Stage 4's NoisyGated, t_fallback 27.7 us) and the kNN 2-of-3 gate;
# the trigger utilities then count only read-out detectors (TRIG_GATED). Background seeded per layout (same draws for
# every layout).
BG_NS = float(os.environ.get("BG_NS", "0"))
# RECO=v1|v2|v3: angular metrics from the analytical reconstruction (modules/optimize/analytic_reco.py, port of
# TamboDirReco.jl) instead of the DeepSets recon; hits = detectors with >= 1 expected particle and a time; a shower
# with < 3 hits or a failed fit counts as 180 deg. Utilities and the energy column are not computed.
RECO = os.environ.get("RECO", "")
ANA_HIT = os.environ.get("ANA_HIT", "mu")              # mu: particle counts (as TamboDirReco); E: kernel energy
if RECO:
    from modules.optimize import analytic_reco as AR
    _tri = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32)
    N_PLANE = AR.plane_normal(_tri).to(dev)
    print(f"analytical recon {RECO}; detector-plane normal {N_PLANE.tolist()}", flush=True)
    assert BG_NS == 0, "analytical recon: background not supported"
# BG_GATE=none: background without the kNN gate (every detector read out); default knn23
BG_GATE = os.environ.get("BG_GATE", "knn23")
print(f"background {BG_NS:g} ns" + (f", gate {BG_GATE}" if BG_NS > 0 else ""), flush=True)
# REF_LAYOUT=<layout .pt>: same-shower check; the triggered selection (*_cnt metrics) keeps only showers triggered by
# both the scored layout and this reference layout (no background)
REF = os.environ.get("REF_LAYOUT", "")
if REF:
    _R = torch.load(REF, map_location="cpu", weights_only=False)
    xr = torch.as_tensor(_R["x"]).float().to(dev); yr = torch.as_tensor(_R["y"]).float().to(dev)
    print(f"triggered selection AND reference layout {REF}", flush=True)
for name, path in pairs:
    if name in res:
        continue
    L = torch.load(path, map_location="cpu", weights_only=False)
    x = torch.as_tensor(L["x"]).float().to(dev); y = torch.as_tensor(L["y"]).float().to(dev)
    acc = {k: 0.0 for k in MODES}
    a_all, e_all, det_all, trig_all, cnt_all = [], [], [], [], []
    gen = torch.Generator(device=dev); gen.manual_seed(20261107)
    with torch.no_grad():
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            kern = S4.BatchKernel([c[lo:hi].to(dev) for c in clouds], [None] * len(clouds), surface,
                                  differentiable=False, frames=frames[lo:hi].to(dev))
            p = prim[lo:hi].to(dev)
            xy = torch.stack([x, y], -1)[None].expand(hi - lo, -1, -1)
            ref_ok = None
            if REF:
                o_r = kern(p, torch.stack([xr, yr], -1)[None].expand(hi - lo, -1, -1))
                ref_ok = ((kern.mu >= 1) & (o_r[..., 1] > 0)).sum(1).ge(3)
            if BG_NS > 0:
                kern = S4.NoisyGated(kern, BG_NS, BG_GATE, gen, 27.7, straight_through=False)
            out = kern(p, xy); cache = Cached(out, kern.mu, getattr(kern, "gate_mask", None))
            _gm = getattr(kern, "gate_mask", None)
            _mu = kern.mu * _gm if _gm is not None else kern.mu
            _c = ((_mu >= 1) & (out[..., 1] > 0)).sum(1).ge(3)                     # >= 3 detectors with >= 1 particle
            cnt_all.append((_c & ref_ok if ref_ok is not None else _c).cpu())
            if RECO:
                mu = kern.mu; t_us = torch.expm1(out[..., 1].clamp(min=0)); Edet = torch.expm1(out[..., 0].clamp(min=0))
                if ANA_HIT == "mu":                       # realistic: >= 1 expected particle (2 m^2), count = mu
                    hit, cnt = (mu >= 1) & (t_us > 0), mu
                else:                                     # as the DeepSets recon sees it: E > 1 with a time, count = E
                    hit, cnt = (Edet > 1) & (t_us > 0), Edet
                X = torch.stack([x, y, surface(y, x)], -1)[None].expand(hi - lo, -1, -1)
                if lo == 0 and os.environ.get("DUMP"):
                    torch.save(dict(X=X.cpu(), t=t_us.cpu(), n=cnt.cpu(), hit=hit.cpu(), p=p.cpu(), npl=N_PLANE.cpu()),
                               os.environ["DUMP"])
                rr = AR.RUNS[RECO](X, t_us, cnt, hit, N_PLANE)
                dtrue = torch.nn.functional.normalize(p[:, :3], dim=-1)
                if lo == 0:
                    print(f"  [check] true dir into plane {float(((dtrue * N_PLANE).sum(-1) <= 0).float().mean()):.2f}  "
                          f"hits/shower {float(hit.sum(1).float().mean()):.1f}  valid {float(rr['valid'].float().mean()):.2f}  "
                          f">=3 hits {float((hit.sum(1) >= 3).float().mean()):.2f}  finite {float(torch.isfinite(rr['dir']).all(-1).float().mean()):.2f}",
                          flush=True)
                a = torch.rad2deg(torch.arccos((rr["dir"] * dtrue).sum(-1).clamp(-1, 1)))
                a = torch.where(rr["valid"], a, torch.full_like(a, 180.0))
                a_all.append(a.cpu()); e_all.append(torch.full_like(a, float("nan")).cpu())
                det_all.append(((torch.expm1(out[..., 0]) > 1).sum(1) >= 3).cpu())
                trig_all.append(trigger_prob(mu, 3).cpu())
                continue
            Ed, Ep, th, ph = O._predict_primary(x, y, p, cache, recon)
            Et, tht, pht = O.primary_to_physical_labels(p)
            a = O.angle_between_deg(O._sph(th, ph), O._sph(tht, pht))
            a_all.append(a.cpu()); e_all.append((torch.log10(Ep) - torch.log10(Et)).abs().cpu())
            det_all.append(((torch.expm1(out[..., 0]) > 1).sum(1) >= 3).cpu())
            gm = cache.gate_mask
            trig_all.append(trigger_prob(kern.mu * gm if gm is not None else kern.mu, 3).cpu())
            for k, cfg in MODES.items():
                for kk, vv in cfg.items():
                    setattr(O, kk, vv)
                O.TRIG_GATED = BG_NS > 0 and BG_GATE == "knn23"
                U, _, _ = O.utility_of_xy(x, y, p, cache, recon)
                scale = 1.0 if k == "orig" else O.TWO_TIER_SCALE
                acc[k] += float(U) / scale * (hi - lo) / N
    a = torch.cat(a_all).numpy(); e = torch.cat(e_all).numpy()
    r = dict(path=path, n=int(N), med=float(np.median(a)), e02=float(np.mean(e < 0.2)),
             det=float(torch.cat(det_all).float().mean()), trig=float(torch.cat(trig_all).mean()))
    for t in (1, 2.5, 5):
        r[f"w{t:g}"] = float(np.mean(a < t))
    cm = torch.cat(cnt_all).numpy().astype(bool)            # count-detected: >= 3 detectors with mu >= 1
    ac = a[cm]
    r.update(n_cnt=int(cm.sum()), med_cnt=float(np.median(ac)) if len(ac) else float("nan"),
             e02_cnt=float(np.mean(e[cm] < 0.2)) if len(ac) else float("nan"),
             **{f"w{t:g}_cnt": float(np.mean(ac < t)) if len(ac) else float("nan") for t in (1, 2.5, 5)})
    dm = torch.cat(det_all).numpy().astype(bool)            # detected: >= 3 detectors with E > 1
    ad = a[dm]
    r.update(n_det=int(dm.sum()), med_det=float(np.median(ad)) if len(ad) else float("nan"),
             **{f"w{t:g}_det": float(np.mean(ad < t)) if len(ad) else float("nan") for t in (1, 2.5, 5)})
    r.update({k: float("nan") for k in acc} if RECO else acc)
    res[name] = r
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"[{name}] " + " ".join(f"{k}={v:.4g}" for k, v in r.items() if k != "path"), flush=True)
print("done", flush=True)

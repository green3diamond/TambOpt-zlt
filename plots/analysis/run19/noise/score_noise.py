"""Run 19: score the test layouts and the optimised designs with cosmic-ray background added to the detectors.

Background (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004): 1,826 muons / m^2 / s; detectors of 2 m^2, so per detector and
shower lambda = 2 * 1.826e-6 * W[ns] in a window W centred on the true front arrival at the detector. Per window:
    E' = E + N_bg,  N_bg ~ Poisson(lambda)  (one particle per background muon)
    T' = min(T, earliest background time),  background times ~ U(t_front - W/2, t_front + W/2)
    t_front,i = t_ref + (r_i - r_ref) . n / c   (r = (E, N, Up); t_ref, r_ref = the earliest timed detector)
Trigger: >= 3 detectors with E' > 1. The recon is unchanged (run 19, trained without noise).
Writes noise_scores.json: {layout: {W: metrics}}; W = 0 is the no-noise baseline through the same code path.

    python score_noise.py [--windows 0 128 1000 10000]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
HERE = os.path.dirname(os.path.abspath(__file__))
R = C.RUN_LOCATION
T_DIR = os.path.join(R, "diagnostics", "tests")
OPT = os.path.join(R, "diagnostics", "opt", "opt_scores.json")
RATE_PER_M2_NS = 1.826e-6
AREA_M2 = 2.0
C_M_PER_US = 299.792458
MIN_HIT = 3

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000], help="integration windows [ns]")
ap.add_argument("--n-events", type=int, default=5000)
args = ap.parse_args()

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=C.RECON_FOLDER + "_deepsets", with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
n_true = DCS.sph(th_t, ph_t)                                                  # (B, 3) ENU, propagation direction

layouts = {}
for line in open(os.path.join(T_DIR, "layouts.txt")):
    if line.strip():
        k, p = line.strip().split("=", 1); layouts[k] = p
for k, r in json.load(open(OPT)).items():
    layouts["opt:" + k] = r["path"]
print(f"{len(layouts)} layouts, windows {args.windows} ns", flush=True)

out = {}
T_FALLBACK = None
for li, (name, path) in enumerate(layouts.items()):
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    nd = e.numel()
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    P = torch.stack([e, n, surface(n, e)], -1)                                # (nd, 3) ENU m
    with torch.no_grad():
        lab = kern(prim, xy)
        E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
        if T_FALLBACK is None:
            T_FALLBACK = float(t[timed].median()) if timed.any() else 10.0
        big = torch.finfo(t.dtype).max
        k_ref = torch.where(timed, t, torch.full_like(t, big)).argmin(1)
        any_t = timed.any(1)
        r_ref = torch.where(any_t[:, None], P[k_ref], P.mean(0).expand(B, 3))
        t_ref = torch.where(any_t, t.gather(1, k_ref[:, None]).squeeze(1), torch.full_like(t[:, 0], T_FALLBACK))
        t_front = t_ref[:, None] + ((P[None] - r_ref[:, None]) * n_true[:, None]).sum(-1) / C_M_PER_US   # (B, nd) us
        res = {}
        for W in args.windows:
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W))
            lam = AREA_M2 * RATE_PER_M2_NS * W
            Nb = torch.poisson(torch.full_like(E, lam), generator=g) if lam > 0 else torch.zeros_like(E)
            u = torch.rand(E.shape, generator=g, device=dev)
            frac = 1.0 - u.pow(1.0 / Nb.clamp(min=1.0))                          # min of N_bg uniforms ~ Beta(1, N_bg)
            t_bg = (t_front - W / 2000.0 + (W / 1000.0) * frac).clamp(min=1e-6)
            hasbg = Nb > 0
            Tn = torch.where(hasbg, torch.where(timed, torch.minimum(t, t_bg), t_bg), t)
            En = E + Nb
            feats = torch.cat([xy, torch.log1p(En)[..., None], torch.log1p(Tn)[..., None]], -1)
            E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
            a, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
            trig = (En > 1).sum(1) >= MIN_HIT
            a, trig = a.cpu().numpy(), trig.cpu().numpy()
            r = dict(n_det=nd, lam=lam, nbg_mean=float(Nb.mean()), bg_only=float((hasbg & ~timed).sum(1).float().mean()),
                     t_changed=float((hasbg & timed & (t_bg < t)).float().mean()), triggered=float(trig.mean()),
                     ang_med=float(np.median(a)), ang_med_trig=float(np.median(a[trig])) if trig.any() else float("nan"))
            for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
                r[f"notrig_{kk}"] = float(np.mean(a < th))
                r[f"all_{kk}"] = float(np.mean((a < th) & trig))
                r[f"trig_{kk}"] = float(np.mean(a[trig] < th)) if trig.any() else float("nan")
            res[f"{W:g}"] = r
    out[name] = res
    s = "  ".join(f"W={w}: {100*v['all_1']:.1f}/{100*v['all_5']:.1f} trig {100*v['triggered']:.0f}%" for w, v in res.items())
    print(f"[{name:30s}] n={nd:3d}  {s}", flush=True)
json.dump(dict(rate_per_m2_ns=RATE_PER_M2_NS, area_m2=AREA_M2, min_hit=MIN_HIT, n_events=B, t_fallback_us=T_FALLBACK,
               windows_ns=args.windows, scores=out), open(os.path.join(HERE, "noise_scores.json"), "w"), indent=1)
print("[write]", os.path.join(HERE, "noise_scores.json"))

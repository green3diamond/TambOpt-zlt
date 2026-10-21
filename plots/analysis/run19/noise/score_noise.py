"""Run 19: score the test layouts and the optimised designs with cosmic-ray background added to the detectors.

Background (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004): 1,826 muons / m^2 / s; detectors of 2 m^2, so per detector and
shower lambda = 2 * 1.826e-6 * W[ns] in one readout window W, shared by all detectors and centred on the shower's
earliest timed detector t_ref. Per window:
    E' = E + N_bg * E_PER_BG_MUON,  N_bg ~ Poisson(lambda)  (a muon in kernel units, ~899; modules/optimize/background.py)
    T' = min(T, earliest background time),  background times ~ U(t_ref - W/2, t_ref + W/2)
Trigger: >= 3 detectors with E' > 1. Recon: the clean run 19 recon, or --recon-dir (e.g. the noise fine-tune).
Writes noise_scores.json: {layout: {W: metrics}}; W = 0 is the no-noise baseline through the same code path.

    python score_noise.py [--windows 0 128 1000 10000] [--layouts other_layouts.txt --no-opt --out other.json]
                          [--recon-dir <RECON_FOLDER>_deepsets_noise]
The background model lives in modules/optimize/background.py (shared with the noise fine-tune).
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
from modules.optimize.background import RATE_PER_M2_NS, AREA_M2, E_PER_BG_MUON, BG_UNITS, window_centres, add_background  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
HERE = os.path.dirname(os.path.abspath(__file__))
R = C.RUN_LOCATION
T_DIR = os.path.join(R, "diagnostics", "tests")
OPT = os.path.join(R, "diagnostics", "opt", "opt_scores.json")
MIN_HIT = 3

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000], help="integration windows [ns]")
ap.add_argument("--n-events", type=int, default=5000)
ap.add_argument("--layouts", default=os.path.join(T_DIR, "layouts.txt"), help="name=path.pt lines")
ap.add_argument("--no-opt", action="store_true", help="skip the optimised designs in opt_scores.json")
ap.add_argument("--out", default=os.path.join(HERE, "noise_scores.json"))
ap.add_argument("--recon-dir", default=C.RECON_FOLDER + "_deepsets")
args = ap.parse_args()

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
n_true = DCS.sph(th_t, ph_t)                                                  # (B, 3) ENU, propagation direction

layouts = {}
for line in open(args.layouts):
    if line.strip():
        k, p = line.strip().split("=", 1); layouts[k] = p
for k, r in ({} if args.no_opt else json.load(open(OPT))).items():
    layouts["opt:" + k] = r["path"]
print(f"{len(layouts)} layouts, windows {args.windows} ns", flush=True)

out = {}
T_FALLBACK = None
for li, (name, path) in enumerate(layouts.items()):
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    nd = e.numel()
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    with torch.no_grad():
        lab = kern(prim, xy)
        E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
        if T_FALLBACK is None:
            T_FALLBACK = float(t[timed].median()) if timed.any() else 10.0
        t_front = window_centres(t, timed, T_FALLBACK)                          # (B, nd) us
        res = {}
        for W in args.windows:
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W))
            lam = AREA_M2 * RATE_PER_M2_NS * W
            En, Tn, Nb, hasbg = add_background(E, t, timed, t_front, W, g)
            feats = torch.cat([xy, torch.log1p(En)[..., None], torch.log1p(Tn)[..., None]], -1)
            E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
            a, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
            trig = (En > 1).sum(1) >= MIN_HIT
            a, trig = a.cpu().numpy(), trig.cpu().numpy()
            r = dict(n_det=nd, lam=lam, nbg_mean=float(Nb.mean()), bg_only=float((hasbg & ~timed).sum(1).float().mean()),
                     t_changed=float((hasbg & timed & (Tn < t)).float().mean()), triggered=float(trig.mean()),
                     ang_med=float(np.median(a)), ang_med_trig=float(np.median(a[trig])) if trig.any() else float("nan"))
            for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
                r[f"notrig_{kk}"] = float(np.mean(a < th))
                r[f"all_{kk}"] = float(np.mean((a < th) & trig))
                r[f"trig_{kk}"] = float(np.mean(a[trig] < th)) if trig.any() else float("nan")
            res[f"{W:g}"] = r
    out[name] = res
    s = "  ".join(f"W={w}: {100*v['all_1']:.1f}/{100*v['all_5']:.1f} trig {100*v['triggered']:.0f}%" for w, v in res.items())
    print(f"[{name:30s}] n={nd:3d}  {s}", flush=True)
json.dump(dict(recon_dir=args.recon_dir, rate_per_m2_ns=RATE_PER_M2_NS, area_m2=AREA_M2, bg_units=BG_UNITS, e_per_bg_muon=E_PER_BG_MUON, min_hit=MIN_HIT, n_events=B, t_fallback_us=T_FALLBACK,
               windows_ns=args.windows, scores=out), open(args.out, "w"), indent=1)
print("[write]", args.out)

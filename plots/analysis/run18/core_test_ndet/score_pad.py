"""Score N-detector layouts with the run's recon, padding the 100 recon slots with ZEROS instead of
repeating the N detectors: the N real detectors carry their kernel signals and positions, the remaining
100 - N slots are (x, y, log1p E, log1p T) = (0, 0, 0, 0). The recon has no mask, so a zero slot reads as
a dark detector at ENU (0, 0) (the middle of the plane's bottom edge in run 18).
Writes scores_pad_<theta>deg.json with score_layouts.py's keys (frac_good_t = angle only at --theta-good;
frac_good_5 = 5 deg and |dlog10 E| < 0.2; ang_med).

    python score_pad.py --layouts layouts.txt [--layouts ../angleonly/layouts.txt]
"""
import argparse
import importlib.util
import json
import os
import sys

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
GS = DCS.GS
D = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--layouts", action="append", required=True)
ap.add_argument("--n-events", type=int, default=1536)
ap.add_argument("--thetas", type=float, nargs="+", default=[5.0, 2.5])
args = ap.parse_args()
dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)

layouts = {}
for f in args.layouts:
    layouts.update(dict(l.strip().split("=", 1) for l in open(f) if l.strip()))
outs = {th: {} for th in args.thetas}
for name, path in layouts.items():
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    nd = e.numel()
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    with torch.no_grad():
        lab = kern(prim, xy)                                                   # (B, nd, 2) log1p E, T
        feats = torch.zeros(B, C.N_DETECTORS, 4, device=dev)
        feats[:, :nd, :2] = xy
        feats[:, :nd, 2:] = lab
        E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
        a, de = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
    good5 = float(((a < 5) & (de < 0.2)).float().mean())
    for th, out in outs.items():
        out[name] = dict(frac_good_t=float((a < th).float().mean()), frac_good_5=good5,
                         ang_med=float(a.median()), dE_med=float(de.median()), n_det=nd, padding="zeros")
    print(f"[{name:28s}] n={nd:3d}  " + "  ".join(f"within {th:g}° {outs[th][name]['frac_good_t']:.3f}" for th in outs)
          + f"  good5(+E) {good5:.3f}  angle {float(a.median()):.2f}", flush=True)
for th, out in outs.items():
    json.dump(out, open(os.path.join(D, f"scores_pad_{th:g}deg.json"), "w"), indent=1)
print("[write]", ", ".join(f"scores_pad_{th:g}deg.json" for th in outs))

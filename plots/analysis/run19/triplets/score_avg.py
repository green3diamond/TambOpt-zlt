"""Run 19 triplet test, averaged stations: each triangle of 3 detectors is read out as ONE station.

Per triplet layout tri<K>_r<r>_n<3K> (detectors stored cluster by cluster) and window W (background as in
modules/optimize/background.py):
    station position = centroid of its 3 detectors
    station E        = mean of the 3 detectors' E' (signal + background)
    station T        = mean of T' over the station's timed detectors (signal or background); untimed if none
The recon (run 19, unchanged) gets the K stations. Trigger: >= 3 DETECTORS with E' > 1 (the same hardware as the
separate-detector test). For reference, 'centres': one detector at each station centroid (K detectors, no averaging).
Writes triplet_avg_scores.json: {layout: {"avg" | "centres": {W: metrics}}}.

    python score_avg.py [--windows 0 128 1000 10000] [--recon-dir ...]
"""
import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from modules.optimize.background import window_centres, add_background  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
T_DIR = os.path.join(C.RUN_LOCATION, "diagnostics", "triplets")
MIN_HIT = 3

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000])
ap.add_argument("--n-events", type=int, default=5000)
ap.add_argument("--recon-dir", default=C.RECON_FOLDER + "_deepsets")
ap.add_argument("--out", default=os.path.join(T_DIR, "triplet_avg_scores.json"))
args = ap.parse_args()

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
n_true = DCS.sph(th_t, ph_t)
T_FALLBACK = None


def metrics(a, trig, extra):
    r = dict(extra, triggered=float(trig.mean()), ang_med=float(np.median(a)),
             ang_med_trig=float(np.median(a[trig])) if trig.any() else float("nan"))
    for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
        r[f"notrig_{kk}"] = float(np.mean(a < th))
        r[f"all_{kk}"] = float(np.mean((a < th) & trig))
        r[f"trig_{kk}"] = float(np.mean(a[trig] < th)) if trig.any() else float("nan")
    return r


def recon_err(xy, lE, lT):
    E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, lE[..., None], lT[..., None]], -1)))
    return DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))[0].cpu().numpy()


def labels(e, n):
    """Kernel E [particles], t [us], timed for detectors at (e, n)."""
    global T_FALLBACK
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    lab = kern(prim, xy)
    E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
    if T_FALLBACK is None:
        T_FALLBACK = float(t[timed].median()) if timed.any() else 10.0
    return xy, E, t, timed


layouts = {}
for line in open(os.path.join(T_DIR, "layouts.txt")):
    k, p = line.strip().split("=", 1)
    if re.fullmatch(r"tri\d+_r\d+_n\d+", k):
        layouts[k] = p
print(f"{len(layouts)} triplet layouts, windows {args.windows} ns, recon {args.recon_dir}", flush=True)

out = {}
for li, (name, path) in enumerate(layouts.items()):
    K = int(re.match(r"tri(\d+)_", name).group(1))
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    res = {"avg": {}, "centres": {}}
    with torch.no_grad():
        xy, E, t, timed = labels(e, n)
        t_front = window_centres(t, timed, T_FALLBACK)
        ce, cn = e.view(K, 3).mean(1), n.view(K, 3).mean(1)
        xy_c, E_c, t_c, timed_c = labels(ce, cn)
        tf_c = window_centres(t_c, timed_c, T_FALLBACK)
        for W in args.windows:
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W))
            En, Tn, Nb, hasbg = add_background(E, t, timed, t_front, W, g)
            tm = (timed | hasbg).view(B, K, 3).float()
            E_st = En.view(B, K, 3).mean(-1)
            T_st = (Tn.view(B, K, 3) * tm).sum(-1) / tm.sum(-1).clamp(min=1.0)    # 0 (untimed) if no timed detector
            trig = ((En > 1).sum(1) >= MIN_HIT).cpu().numpy()
            a = recon_err(xy_c, torch.log1p(E_st), torch.log1p(T_st))
            res["avg"][f"{W:g}"] = metrics(a, trig, dict(n_det=3 * K, n_st=K, nbg_mean=float(Nb.mean()),
                                                          st_timed=float((tm.sum(-1) > 0).float().sum(1).mean())))
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W) + 7)
            En, Tn, Nb, _ = add_background(E_c, t_c, timed_c, tf_c, W, g)
            trig = ((En > 1).sum(1) >= MIN_HIT).cpu().numpy()
            a = recon_err(xy_c, torch.log1p(En), torch.log1p(Tn))
            res["centres"][f"{W:g}"] = metrics(a, trig, dict(n_det=K))
    out[name] = res
    s = "  ".join(f"W={w}: avg {100*res['avg'][w]['notrig_1']:.1f}/{100*res['avg'][w]['all_1']:.1f}  "
                  f"ctr {100*res['centres'][w]['notrig_1']:.1f}/{100*res['centres'][w]['all_1']:.1f}" for w in res["avg"])
    print(f"[{name:16s}] (1 deg no trig / trig) {s}", flush=True)
json.dump(dict(recon_dir=args.recon_dir, windows_ns=args.windows, min_hit=MIN_HIT, n_events=B, scores=out),
          open(args.out, "w"), indent=1)
print("[write]", args.out)

"""Score the run 19 layout optimisations (finished runs only) and their grid / centre starts on the 5,000 held-out
test showers with the run's recon: share within 1 / 2.5 / 5 deg over all showers and over detected showers only
(>= 3 detectors with > 1 particle), the detected share and the median error. Writes opt_scores.json."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
O = os.path.dirname(os.path.abspath(__file__))
R = C.RUN_LOCATION
RUNS = (("current", "current_lr10", "", 20000), ("twotier_l0", "twotier_l0_lr10", "", 20000),
        ("twotier_l0p3", "twotier_l0p3_lr10", "", 20000), ("threetier", "threetier_090301_lr10", "", 20000),
        ("twotier_l0p3_b2048", "twotier_l0p3_lr10", "_b2048_lrd1", 5000))           # (key, folder, tail, steps)
MIN_HIT = 3

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
layouts = {}
for N in (100, 50, 20):
    SN = "" if N == 100 else f"_n{N}"
    for scheme in ("grid", "center"):
        e, n = sample_initial_layout_ne(m, n_units=N, scheme=scheme)
        e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
        p = f"{O}/layouts/start_{scheme}{SN}.pt"; torch.save(dict(x=e.float(), y=n.float()), p); layouts[f"start_{scheme}{SN}"] = p
    for key, suf, tail, steps in RUNS:
        for scheme in ("grid", "center"):
            d = f"{R}/test_v6_run_04_optimize_lbfgs_kernel_{suf}_{scheme}{SN}{tail}"
            log = f"{d}/optimize_log.json"
            if os.path.exists(f"{d}/layout_best.pt") and os.path.exists(log):
                L = json.load(open(log)); h = L if isinstance(L, list) else next(
                    v for v in L.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "U" in v[0])
                if len(h) >= steps:
                    layouts[f"{key}_{scheme}{SN}"] = f"{d}/layout_best.pt"
print("layouts:", list(layouts))

surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=C.RECON_FOLDER + "_deepsets", with_fnn=False)
clouds, B, _, prim = load_events(5000, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
out = {}
for name, path in layouts.items():
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    with torch.no_grad():
        lab = kern(prim, xy)
        E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, lab], -1)))
        a, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
        hit = (torch.expm1(lab[..., 0]) > 1).sum(1)
    a, hit = a.cpu().numpy(), hit.cpu().numpy()
    det = hit >= MIN_HIT
    r = dict(path=path, n_det=int(e.numel()), n_events=B, detected=float(det.mean()), hit_med=float(np.median(hit)), ang_med=float(np.median(a)),
             ang_med_det=float(np.median(a[det])) if det.any() else float("nan"))
    for th, k in ((1, "1"), (2.5, "2.5"), (5, "5")):
        r[f"all_{k}"] = float(np.mean(a < th))
        r[f"det_{k}"] = float(np.mean(a[det] < th)) if det.any() else float("nan")
    out[name] = r
    print(f"[{name:22s}] all 1/2.5/5: {100*r['all_1']:5.1f} {100*r['all_2.5']:5.1f} {100*r['all_5']:5.1f}  "
          f"detected {100*r['detected']:5.1f}%  det-only: {100*r['det_1']:5.1f} {100*r['det_2.5']:5.1f} {100*r['det_5']:5.1f}  "
          f"median {r['ang_med']:.2f} (det {r['ang_med_det']:.2f})", flush=True)
json.dump(out, open(f"{O}/opt_scores.json", "w"), indent=1)
print("[write]", f"{O}/opt_scores.json")

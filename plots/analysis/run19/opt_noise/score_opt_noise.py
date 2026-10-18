"""Score the noise-aware Stage 4 layouts (finished runs: layout_best.pt present) and references on the 5,000 held-out
test showers with the KERNEL and the clean run 19 recon, in five readouts:
    clean          no background, no gate
    1us / 10us     background in one readout window (1 / 10 us) around the earliest hit, no gate
    1us_knn / 10us_knn   the same with the exact kNN 2-of-3 gate (E' > 0.001, detector + its 2 nearest neighbours)
References: the grid and centre starts, grid_n100 and disk_r6000_n100 (layout tests). Same noise draws for every
layout (seeded per readout). Writes opt_noise_scores.json: {layout: {readout: metrics}} and opt_noise_errors.npz (per-shower angle errors,
key "<layout>|<readout>").

    python score_opt_noise.py
"""
import glob
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
from modules.optimize.background import window_centres, add_background, knn_pairs, knn_gate  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
R = C.RUN_LOCATION
OUT = os.path.join(R, "diagnostics", "opt_noise")
READOUTS = (("clean", 0, False), ("1us", 1000, False), ("1us_knn", 1000, True), ("10us", 10000, False), ("10us_knn", 10000, True))

layouts = {}
for d in sorted(glob.glob(os.path.join(R, "test_v6_run_04_optimize_lbfgs_kernel_twotier_*_b4096_bg*_knn23*"))):
    if os.path.exists(os.path.join(d, "layout_best.pt")):
        layouts[d.split("_lbfgs_kernel_")[1]] = os.path.join(d, "layout_best.pt")
for k in ("start_grid", "start_center"):
    layouts[k] = os.path.join(R, "diagnostics", "opt", "layouts", f"{k}.pt")
for line in open(os.path.join(R, "diagnostics", "tests", "layouts.txt")):
    k, p = line.strip().split("=", 1)
    if k in ("grid_n100", "disk_r6000_n100"):
        layouts[k] = p
print(f"{len(layouts)} layouts:", list(layouts), flush=True)

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=C.RECON_FOLDER + "_deepsets", with_fnn=False)
clouds, B, _, prim = load_events(5000, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
T_FALLBACK = 27.7

out, errs = {}, {}
for name, path in layouts.items():
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    nn = knn_pairs(e, n)
    res = {}
    with torch.no_grad():
        lab = kern(prim, xy)
        E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
        t_c = window_centres(t, timed, T_FALLBACK)
        for ri, (rd, W, gate) in enumerate(READOUTS):
            g = torch.Generator(device=dev); g.manual_seed(1000 + ri)
            En, Tn, Nb, hasbg = add_background(E, t, timed, t_c, W, g)
            keep = knn_gate(En, nn)[0] if gate else torch.ones_like(En)
            Ek, Tk = En * keep, Tn * keep
            E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, torch.log1p(Ek)[..., None],
                                                                         torch.log1p(Tk)[..., None]], -1)))
            a = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))[0].cpu().numpy()
            r = dict(n_det=int(e.numel()), ang_med=float(np.median(a)), read=float(keep.mean()),
                     ghosts=float(((hasbg & ~timed) & (keep > 0)).float().sum(1).mean()))
            for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
                r[f"w{kk}"] = float(np.mean(a < th))
            res[rd] = r
            errs[f"{name}|{rd}"] = a.astype(np.float32)
    out[name] = res
    print(f"[{name:52s}] " + "  ".join(f"{rd} {100*res[rd]['w1']:.1f}/{100*res[rd]['w5']:.1f}" for rd, _, _ in READOUTS), flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(dict(readouts=[r[0] for r in READOUTS], n_events=B, scores=out), open(os.path.join(OUT, "opt_noise_scores.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "opt_noise_errors.npz"), **errs)           # per-shower angle errors [deg]
print("[write]", os.path.join(OUT, "opt_noise_scores.json"), "and opt_noise_errors.npz")

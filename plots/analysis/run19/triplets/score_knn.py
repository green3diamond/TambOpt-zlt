"""Run 19 quick test, kNN 2-of-3 gate: every detector is read out only if at least 2 of {itself, its 2 nearest
neighbours (in the layout, East-North)} are hit (hit = E' > 0.001); otherwise E = 0, untimed, before the recon.
Same noise draws for every readout ('none' = all read out, 'knn' = gated; tri* also 'group' = 2-of-3 within its own
triangle). Clean run 19 recon.
Layouts: grid_n99, disk_r6000_n99, all tri33_* (triplets/layouts.txt) and clu20x5_r1000_n100 (tests/layouts.txt).
Writes knn_scores.json: {layout: {readout: {W: metrics}}}.

    python score_knn.py [--windows 0 128 1000 10000]
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
HIT_MIN = 1e-3                                                                     # the kernel's timing threshold (HIT_DEPOSIT_MIN)
READOUTS = ("none", "knn")

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000])
ap.add_argument("--n-events", type=int, default=5000)
ap.add_argument("--recon-dir", default=C.RECON_FOLDER + "_deepsets")
ap.add_argument("--out", default=os.path.join(T_DIR, "knn_scores.json"))
args = ap.parse_args()

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
T_FALLBACK = None

layouts = {}
for f, pat in ((os.path.join(T_DIR, "layouts.txt"), r"grid_n99|disk_r6000_n99|tri33_r\d+_n99"),
               (os.path.join(C.RUN_LOCATION, "diagnostics", "tests", "layouts.txt"), r"clu20x5_r1000_n100")):
    for line in open(f):
        k, p = line.strip().split("=", 1)
        if re.fullmatch(pat, k):
            layouts[k] = p
print(f"{len(layouts)} triplet layouts, windows {args.windows} ns, recon {args.recon_dir}", flush=True)

out = {}
for li, (name, path) in enumerate(layouts.items()):
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    nd = e.numel()
    d = torch.cdist(torch.stack([e, n], -1), torch.stack([e, n], -1)).fill_diagonal_(float("inf"))
    nn = d.topk(2, largest=False).indices                                          # (nd, 2) nearest neighbours
    tri = name.startswith("tri")                                                   # stored triangle by triangle
    rds = READOUTS + (("group",) if tri else ())
    res = {r: {} for r in rds}
    if tri:
        mates = torch.arange(nd, device=dev).view(-1, 3).repeat_interleave(3, 0)  # (nd, 3) own triangle
        same = float((torch.sort(nn, 1).values == torch.sort(mates[mates != torch.arange(nd, device=dev)[:, None]]
                                                              .view(nd, 2), 1).values).all(1).float().mean())
        res["knn_is_triangle"] = same                                              # share of detectors whose kNN = mates
    with torch.no_grad():
        lab = kern(prim, xy)
        E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
        if T_FALLBACK is None:
            T_FALLBACK = float(t[timed].median()) if timed.any() else 10.0
        t_c = window_centres(t, timed, T_FALLBACK)
        for W in args.windows:
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W))
            En, Tn, Nb, hasbg = add_background(E, t, timed, t_c, W, g)
            ghost = hasbg & ~timed                                                 # background-only detectors
            for rd in rds:
                h = (En > HIT_MIN).float()
                if rd == "none":
                    keep = torch.ones_like(En, dtype=torch.bool)
                elif rd == "knn":
                    keep = (h + h[:, nn[:, 0]] + h[:, nn[:, 1]]) >= 2                # kNN 2-of-3
                else:
                    keep = (h.view(B, -1, 3).sum(-1, keepdim=True) >= 2).expand(B, -1, 3).reshape(B, nd)   # own triangle
                Ek, Tk = torch.where(keep, En, 0.0), torch.where(keep, Tn, 0.0)
                E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, torch.log1p(Ek)[..., None],
                                                                             torch.log1p(Tk)[..., None]], -1)))
                a = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))[0].cpu().numpy()
                trig = ((Ek > 1).sum(1) >= MIN_HIT).cpu().numpy()
                r = dict(n_det=nd, triggered=float(trig.mean()), ang_med=float(np.median(a)),
                         read_det=float(keep.float().sum(1).mean()), ghosts=float((ghost & keep).float().sum(1).mean()),
                         lit_det=float(((Ek > 0) | (Tk > 0)).float().sum(1).mean()))
                for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
                    r[f"notrig_{kk}"] = float(np.mean(a < th))
                    r[f"all_{kk}"] = float(np.mean((a < th) & trig))
                res[rd][f"{W:g}"] = r
    out[name] = res
    s = "  ".join(f"W={w}: " + " ".join(f"{rd} {100*res[rd][w]['notrig_1']:.1f}/{100*res[rd][w]['all_1']:.1f} g{res[rd][w]['ghosts']:.2f}"
                                       for rd in rds) for w in res["none"])
    print(f"[{name:16s}] (1 deg no trig / trig, ghosts) {s}", flush=True)
json.dump(dict(recon_dir=args.recon_dir, windows_ns=args.windows, min_hit=MIN_HIT, n_events=B, scores=out),
          open(args.out, "w"), indent=1)
print("[write]", args.out)

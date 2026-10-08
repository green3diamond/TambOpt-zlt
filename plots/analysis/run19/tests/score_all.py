"""Score every test layout (tests/layouts.txt) with the run's recon on the held-out showers, in one pass.

Per layout: share within 1 / 2.5 / 5 deg (angle only), share good at 5 deg with |dlog10 E| < 0.2, median angle error.
Per shower x layout pair: lit detectors (>= 1 count, kernel), their mean pairwise distance, median nearest-neighbour
spacing and baseline (largest distance), and the angle error. The recon gets exactly the layout's N detectors (no
padding). Writes tests/scores.json and tests/pairs.npz (read by analyze.py).

    python score_all.py [--recon-dir <dir>] [--n-events 1536] [--tag <suffix>]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
GS = DCS.GS
T = os.path.join(C.RUN_LOCATION, "diagnostics", "tests")

ap = argparse.ArgumentParser()
ap.add_argument("--recon-dir", default=C.RECON_FOLDER + "_deepsets")
ap.add_argument("--n-events", type=int, default=1536)
ap.add_argument("--tag", default="", help="suffix for scores_<tag>.json / pairs_<tag>.npz (e.g. another recon)")
args = ap.parse_args()
SUF = f"_{args.tag}" if args.tag else ""

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)


def lit_geometry(xy, lit):
    """xy (nd, 2), lit (B, nd) -> per row: mean pairwise distance, median NN spacing, max distance (nan if < 2 lit)."""
    d = torch.cdist(xy, xy)
    pair = lit[:, :, None] & lit[:, None, :] & ~torch.eye(xy.shape[0], dtype=torch.bool, device=xy.device)
    npair = pair.sum((1, 2)).float()
    nan = torch.full_like(npair, float("nan"))
    mean = torch.where(npair > 0, (d * pair).sum((1, 2)) / npair.clamp(min=1), nan)
    mx = torch.where(npair > 0, (d * pair).amax((1, 2)), nan)
    dn = torch.where(pair, d, torch.full_like(d.expand_as(pair), float("inf"))).amin(2)
    dn = torch.where(lit & torch.isfinite(dn), dn, torch.full_like(dn, float("nan")))
    return mean, torch.nanmedian(dn, dim=1).values, mx


scores, rows, names = {}, {k: [] for k in ("lit", "mean", "nn", "base", "ang", "ev", "lay")}, []
for line in open(os.path.join(T, "layouts.txt")):
    if not line.strip():
        continue
    name, path = line.strip().split("=", 1)
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    pos = torch.stack([e, n], -1)
    xy = pos[None].expand(B, -1, -1)
    with torch.no_grad():
        lab = kern(prim, xy)                                                # (B, nd, 2) log1p E, log1p T
        E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, lab], -1)))
        a, de = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
        lit = torch.expm1(lab[..., 0]) >= 1
        mean, nn_, base = lit_geometry(pos, lit)
    names.append(name)
    for k, v in (("lit", lit.sum(1)), ("mean", mean), ("nn", nn_), ("base", base), ("ang", a)):
        rows[k].append(v.cpu().numpy())
    rows["ev"].append(np.arange(B)); rows["lay"] += [len(names) - 1] * B
    a, de = a.cpu().numpy(), de.cpu().numpy()
    scores[name] = dict(n_det=int(e.numel()), within1=float(np.mean(a < 1)), within2p5=float(np.mean(a < 2.5)),
                        within5=float(np.mean(a < 5)), good5=float(np.mean((a < 5) & (de < 0.2))),
                        ang_med=float(np.median(a)), lit_med=float(np.median(rows["lit"][-1])))
    s = scores[name]
    print(f"[{name:28s}] n={s['n_det']:3d}  1°: {100 * s['within1']:5.1f}%  2.5°: {100 * s['within2p5']:5.1f}%  "
          f"5°: {100 * s['within5']:5.1f}%  median {s['ang_med']:.2f}°  lit {s['lit_med']:.0f}", flush=True)
json.dump(dict(recon_dir=args.recon_dir, n_events=B, scores=scores), open(os.path.join(T, f"scores{SUF}.json"), "w"), indent=1)
np.savez(os.path.join(T, f"pairs{SUF}.npz"), names=np.array(names),
         **{k: (np.concatenate(v) if k != "lay" else np.array(v)) for k, v in rows.items()})
print(f"[write] {T}/scores{SUF}.json, pairs{SUF}.npz ({len(names)} layouts x {B} showers)")

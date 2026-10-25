"""Run 19 ghost events: readout windows with cosmic background muons only (no shower), scored on the validation layouts.
Per window and detector: N_bg ~ Poisson(2 m^2 x rate x W), each muon its own random KE (modules/optimize/background.py),
the earliest muon's time as T, the window centred at --t-centre (default: the fine-tune's median timed T, as 03c). Same
draws for every recon (seeded by layout and window only). Readouts as score_knn.py: all read out, kNN 2-of-3 gate.
Triggered = >= 3 read-out detectors with E' > 1 (the drawn trigger of score_knn.py). Per layout / readout / window:
  ghost_trig     share of ghost windows that trigger (false-trigger probability per window; recon independent)
  lit_det        mean read-out detectors with background
and, over triggered ghosts, what each recon reports: predicted zenith / azimuth percentiles [deg], median predicted
log10 E, share predicted in the training energy range (log10 E 5-7) - ghosts the recon would pass as showers.
Writes ghost_scores.json next to the swgo_trigger scores.

    python plots/analysis/run19/swgotrig/score_ghost.py --layouts <name=path lines> --recon-dirs <dir> [<dir> ...]
                                                        [--windows 1000 10000] [--n-windows 20000]
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
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from modules.optimize.background import add_background, BG_UNITS  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
MIN_HIT = 3
HIT_MIN = 1e-3                                                                     # kNN gate hit, as score_knn.py

ap = argparse.ArgumentParser()
ap.add_argument("--layouts", required=True, help="name=path.pt lines")
ap.add_argument("--recon-dirs", nargs="+", required=True)
ap.add_argument("--windows", type=float, nargs="+", default=[1000, 10000])
ap.add_argument("--n-windows", type=int, default=20000)
ap.add_argument("--batch", type=int, default=5000)
ap.add_argument("--t-centre", type=float, default=None, help="window centre [us]; default: t_fallback_us of a fine-tune")
ap.add_argument("--out", default=os.path.join(D, "ghost_scores.json"))
args = ap.parse_args()

dev = DCS.DEVICE
if args.t_centre is None:
    for rd in args.recon_dirs:
        ck = torch.load(os.path.join(rd, "recon.pt"), map_location="cpu", weights_only=False)
        if "noise_finetune" in ck:
            args.t_centre = float(ck["noise_finetune"]["t_fallback_us"])
            break
    else:
        raise SystemExit("no fine-tuned recon to take t_fallback_us from: pass --t-centre")
recons = {rd: load_models(dev, recon_dir=rd, with_fnn=False)[1].eval() for rd in args.recon_dirs}
layouts = dict(l.strip().split("=", 1) for l in open(args.layouts) if l.strip())
print(f"{len(layouts)} layouts, windows {args.windows} ns, {args.n_windows} ghost windows each, "
      f"centre {args.t_centre:.2f} us, {len(recons)} recons", flush=True)

out = {}
for li, (name, path) in enumerate(layouts.items()):
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    nd = e.numel()
    d = torch.cdist(torch.stack([e, n], -1), torch.stack([e, n], -1)).fill_diagonal_(float("inf"))
    nn = d.topk(2, largest=False).indices
    out[name] = {"none": {}, "knn": {}}
    for W in args.windows:
        g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W) + 17)
        acc = {rd: {r: dict(trig=[], lit=[], th=[], ph=[], le=[]) for r in ("none", "knn")} for rd in recons}
        for b0 in range(0, args.n_windows, args.batch):
            B = min(args.batch, args.n_windows - b0)
            z = torch.zeros(B, nd, device=dev)
            En, Tn, Nb, hasbg = add_background(z, z, z.bool(), torch.full_like(z, args.t_centre), W, g)
            xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
            h = (En > HIT_MIN).float()
            for rdo, keep in (("none", torch.ones_like(h, dtype=torch.bool)),
                              ("knn", (h + h[:, nn[:, 0]] + h[:, nn[:, 1]]) >= 2)):
                Ek, Tk = torch.where(keep, En, 0.0), torch.where(keep, Tn, 0.0)
                trig = (Ek > 1).sum(1) >= MIN_HIT
                inp = torch.cat([xy, torch.log1p(Ek)[..., None], torch.log1p(Tk)[..., None]], -1)
                for rd, rec in recons.items():
                    with torch.no_grad():
                        E_p, th_p, ph_p = primary_to_physical_labels(rec(inp[trig])) if trig.any() else (z[:0, 0],) * 3
                    a = acc[rd][rdo]
                    a["trig"].append(trig.cpu()); a["lit"].append((hasbg & keep).sum(1).float().cpu())
                    a["th"].append(torch.rad2deg(th_p).cpu()); a["ph"].append(torch.rad2deg(ph_p).cpu())
                    a["le"].append(torch.log10(E_p.clamp(min=1.0)).cpu())
        for rdo in ("none", "knn"):
            r = None
            for rd in recons:
                a = {k: torch.cat(v).numpy() for k, v in acc[rd][rdo].items()}
                if r is None:
                    r = dict(n_windows=args.n_windows, n_det=nd, ghost_trig=float(a["trig"].mean()),
                             lit_det=float(a["lit"].mean()), recons={})
                q = lambda x: [float(v) for v in np.percentile(x, [10, 50, 90])] if x.size else [float("nan")] * 3  # noqa: E731
                r["recons"][os.path.basename(rd.rstrip("/"))] = dict(
                    zen_p10_50_90=q(a["th"]), az_p10_50_90=q(a["ph"]),
                    log10E_med=float(np.median(a["le"])) if a["le"].size else float("nan"),
                    in_e_range=float(((a["le"] >= 5) & (a["le"] <= 7)).mean()) if a["le"].size else float("nan"))
            out[name][rdo][f"{W:g}"] = r
        s = " ".join(f"{rdo} {100 * out[name][rdo][f'{W:g}']['ghost_trig']:.1f}%" for rdo in ("none", "knn"))
        print(f"[{name:28s}] W={W:g} ns ghost trigger {s}", flush=True)
json.dump(dict(recon_dirs=args.recon_dirs, windows_ns=args.windows, n_windows=args.n_windows, t_centre_us=args.t_centre,
               min_hit=MIN_HIT, bg_units=BG_UNITS, scores=out), open(args.out, "w"), indent=1)
print("[write]", args.out)

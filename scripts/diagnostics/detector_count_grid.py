"""Reconstruction quality vs the number of detectors in the array (no optimization, no surrogate).

Grid layouts of N detectors over the whole mesh, scored on held-out showers with the run's kernel
and recon. The recon is a DeepSets model with mean pooling over 100 slots, so an N-detector layout
is fed as 100 slots that repeat the N detectors 100/N times (exact for mean pooling, as in
detector_count_scan.py); N must divide 100. Per N: share of showers good at 5 deg and at 2.5 deg
(angle and |dlog10 E| < 0.2), good at 5 deg on angle alone, median and 68% angle error, median
|dlog10 E|, median lit detectors (>= 1 count) per shower, share with fewer than 3 lit.

    python scripts/diagnostics/detector_count_grid.py [--counts 4 5 10 20 25 50 100] [--n-events 1536]
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("dcs", os.path.join(_HERE, "detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DCS)
GS = DCS.GS

import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

INK, GREY, BLUE, ORANGE, GREEN = "#1f2328", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[4, 5, 10, 20, 25, 50, 100])
    ap.add_argument("--n-events", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--recon-dir", default=None)
    ap.add_argument("--lit-min", type=float, default=1.0, help="lit = count > this (0.001 = the kNN gate's hit)")
    ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics", "detector_count"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = DCS.DEVICE
    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                                layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
    _, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
    clouds, B, _, prim = load_events(args.n_events, dev, mountain)
    kern = GS.CachedKernel(clouds, surface, dev, chunk=args.chunk, expected_primary=prim)
    mesh = np.asarray(mountain.centroids_ENU)

    res, lay = {}, {}
    for nd in args.counts:
        assert C.N_DETECTORS % nd == 0, f"{nd} does not divide {C.N_DETECTORS}"
        torch.manual_seed(0)
        e, n = sample_initial_layout_ne(mountain, n_units=nd, scheme="grid")
        e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
        e, n = e.float().to(dev), n.float().to(dev)
        rep = C.N_DETECTORS // nd
        with torch.no_grad():
            counts, parts = GS.predict(e.repeat(rep), n.repeat(rep), prim, kern, recon)
            a, de = DCS.ang_err(parts)
        a, de = a.cpu().numpy(), de.cpu().numpy()
        lit = (counts[:, :nd] > args.lit_min).sum(1).cpu().numpy()
        r = dict(n=nd, events=int(a.size),
                 good5=float(np.mean((a < 5) & (de < 0.2))), good25=float(np.mean((a < 2.5) & (de < 0.2))),
                 good5_angle=float(np.mean(a < 5)), good25_angle=float(np.mean(a < 2.5)),
                 good1_angle=float(np.mean(a < 1)), ang_med=float(np.median(a)),
                 ang_p68=float(np.percentile(a, 68)), de_med=float(np.median(de)),
                 lit_min=args.lit_min, lit_med=float(np.median(lit)), frac_lt3=float(np.mean(lit < 3)),
                 spacing_m=float(np.sqrt(np.ptp(mesh[:, 0]) * np.ptp(mesh[:, 1]) / nd)))
        res[nd] = r
        lay[nd] = (e.cpu().numpy(), n.cpu().numpy())
        print(f"[n={nd:3d}] good5 {r['good5']:.1%}  good2.5 {r['good25']:.1%}  good5(angle) {r['good5_angle']:.1%}  "
              f"angle med {r['ang_med']:.2f} (68% {r['ang_p68']:.1f})  |dlogE| med {r['de_med']:.3f}  "
              f"lit med {r['lit_med']:.0f}  <3 lit {r['frac_lt3']:.0%}  spacing ~{r['spacing_m']:.0f} m", flush=True)
    json.dump(res, open(os.path.join(args.out, "detector_count_grid.json"), "w"), indent=1)

    ns = list(res)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(ns), height_ratios=[1, 1.6], hspace=0.35)
    for i, nd in enumerate(ns):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(mesh[:, 0], mesh[:, 1], s=1, color="#e8e6df", lw=0)
        ax.scatter(*lay[nd], s=12, color=BLUE, lw=0)
        ax.set_title(f"{nd} detectors\n~{res[nd]['spacing_m']:.0f} m apart", fontsize=10, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    sub = gs[1, :].subgridspec(1, 3, wspace=0.28)
    x = np.array(ns)
    a0 = fig.add_subplot(sub[0])
    for key, col, lab in (("good5_angle", GREEN, "within 5° (angle only)"), ("good5", BLUE, "good at 5° (+ energy)"),
                          ("good25", ORANGE, "good at 2.5° (+ energy)")):
        a0.plot(x, [100 * res[k][key] for k in ns], "o-", color=col, lw=2, label=lab)
    a0.set(ylabel="share of showers [%]", title="Well-reconstructed showers")
    a1 = fig.add_subplot(sub[1])
    a1.plot(x, [res[k]["ang_med"] for k in ns], "o-", color=BLUE, lw=2, label="median")
    a1.plot(x, [res[k]["ang_p68"] for k in ns], "o:", color=BLUE, lw=1.5, label="68%")
    a1.set(ylabel="angle error [deg]", title="Angle error", yscale="log")
    a1b = a1.twinx()
    a1b.plot(x, [res[k]["de_med"] for k in ns], "s-", color=ORANGE, lw=1.5, label="median |Δlog10 E|")
    a1b.set_ylabel("median |Δlog10 E|", color=ORANGE)
    a2 = fig.add_subplot(sub[2])
    a2.plot(x, [res[k]["lit_med"] for k in ns], "o-", color=BLUE, lw=2, label=f"median lit (> {args.lit_min:g}) per shower")
    a2b = a2.twinx()
    a2b.plot(x, [100 * res[k]["frac_lt3"] for k in ns], "s-", color=ORANGE, lw=1.5)
    a2b.set_ylabel("showers with < 3 lit [%]", color=ORANGE)
    a2.set(ylabel=f"lit detectors (> {args.lit_min:g}) per shower", title="Detectors seeing a shower")
    for a in (a0, a1, a2):
        a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([str(k) for k in ns])
        a.set_xlabel("detectors in the array (grid)")
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    a0.legend(fontsize=9, frameon=False)
    a1.legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle(f"Reconstruction vs number of detectors (grid over the plane) · {os.path.basename(C.RUN_LOCATION)} · "
                 f"{res[ns[0]]['events']} held-out showers, kernel + the run's recon", fontsize=13, color=INK)
    out = os.path.join(args.out, "detector_count_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()

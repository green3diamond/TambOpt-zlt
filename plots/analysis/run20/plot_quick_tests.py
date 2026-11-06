"""Run 20 quick tests: validation tables of <run 20>/diagnostics/quick/layouts.txt per energy band (val_<recon>_<start>.json
from slurm/run20_quick_val.sh, all held-out chunks pooled) and the construction metrics of each layout.
One section per noise level (no background, 1 us, 10 us); per layout and energy band (log10 E 5.00-5.67 / 5.67-6.33 /
6.33-7.00 / all), with the drawn-count recon and the kNN gate (all read out in the second table): trigger rate [% of all],
1 / 2.5 / 5 deg [% of triggered], New U (u_swgo_drawn). Then the construction metrics (modules/optimize/construction.py
describe) and the cable length. Red = best in the column (Notion colorbox).
Writes <run 20>/diagnostics/quick/quick_tables_notion.md.

    source slurm/run20_guard.sh; [QTAG=quick5] python plots/analysis/run20/plot_quick_tests.py [--recon deepsets_count|deepsets]
QTAG=quick5: the 5 deg tests (diagnostics/quick5).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize.cable import cable_length  # noqa: E402
from modules.optimize.construction import describe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--recon", default="deepsets_count", choices=("deepsets_count", "deepsets"))
args = ap.parse_args()
TAG = os.environ.get("QTAG", "quick")
D = os.path.join(C.RUN_LOCATION, "diagnostics", TAG)
PATHS = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if "=" in l)
SKIP = set(filter(None, os.environ.get("QSKIP", "").split(",")))     # layouts whose files are unreadable (left out)
PATHS = {k: v for k, v in PATHS.items() if k not in SKIP}
DEG = "penalty-only" if TAG == "pen" else "trigger-only" if TAG.startswith("trg") else "5°" if TAG.split("_")[0].endswith("5") else "1°"
OPTN = {"sur": "Sur. SGD", "ker": "Ker. SGD", "trg": "Sur. SGD", "pen": "SGD"}.get(TAG[:3], "SGD")
LABEL = {"grid_n99": "Grid (99)", "grid_n100": "Grid (100, start)", "sgd_nobg_lr1": "SGD 1°, all E (earlier test)"}
LABEL.update({f"{TAG}_{k}": f"{OPTN} {DEG}, {v}" for k, v in (
    ("e500_567", "E 5.00-5.67"), ("e567_633", "E 5.67-6.33"), ("e633_700", "E 6.33-7.00"), ("eall", "all E"),
    ("slope", "+ slope limit"), ("access", "+ access"), ("pads", "+ pads"), ("contour", "+ contour rows"))})
BANDS = [("5-5.667", "E 5.00–5.67"), ("5.667-6.333", "E 5.67–6.33"), ("6.333-7", "E 6.33–7.00"), ("all", "all E")]
NOISE = [("0", "No background"), ("1000", "1 µs background"), ("10000", "10 µs background")]
mark = lambda s: r"$\colorbox{#f8c4c4}{\textbf{" + s + "}}$"                                # noqa: E731


def pooled():
    """{band: {layout: {readout: {window: [trig %, 1, 2.5, 5 % of triggered, New U]}}}}, chunks pooled by shower count."""
    acc = {}
    files = sorted(glob.glob(os.path.join(D, f"val_{args.recon}_*.json")))
    if not files:
        raise SystemExit(f"no {D}/val_{args.recon}_*.json")
    for f in files:
        j = json.load(open(f))
        src = dict(all=j["scores"], **j.get("scores_by_band", {}))
        for band, sc in src.items():
            for lay, by in sc.items():
                for rd in ("none", "knn"):
                    for w, r in by[rd].items():
                        n = r["n_showers"]
                        v = n * np.array([r["triggered"], r["all_1"], r["all_2.5"], r["all_5"], r["u_swgo_drawn"], 1.0,
                                          r["ghosts"]])
                        k = (band, lay, rd, w)
                        acc[k] = acc.get(k, 0.0) + v
    out = {}
    for (band, lay, rd, w), v in acc.items():
        n = v[5]
        out.setdefault(band, {}).setdefault(lay, {}).setdefault(rd, {})[w] = np.r_[
            100 * v[0] / n, 100 * v[1:4] / max(v[0], 1e-9), v[4] / n, v[6] / n]
    return out, len(files)


S, nfiles = pooled()
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surf = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256)
g = torch.load(PATHS["grid_n99"], map_location="cpu")
grid_xy = (g["x"].float(), g["y"].float())
geo = {}
for k, p in PATHS.items():
    L = torch.load(p, map_location="cpu")
    x, y = L["x"].float(), L["y"].float()
    with torch.no_grad():
        geo[k] = dict(describe(x, y, surf, m, grid_xy), cable_km=float(cable_length(x, y, surf)[0]) / 1e3)

rec = "drawn-count recon" if args.recon == "deepsets_count" else "default recon"
md = [f"## Run 20 quick tests ({DEG} utility): validation per energy band ({rec}, drawn counts, count seed 1, {nfiles} chunks)", "",
      "Energy bands in log10 E [GeV]. Trig = share of ALL showers in the band that trigger (≥ 3 read-out detectors with a "
      "drawn particle); 1° / 2.5° / 5° = share of TRIGGERED showers within that angle; New U = mean[trig · (g_a + "
      "0.3(1−g_a)q_a)] over all showers. Layouts: the last step of each 100-step SGD test from the grid. Red = best in "
      "the column."]
keys = [k for k in PATHS]
for w, wname in NOISE:
    for rd, rname in (("knn", "kNN gate"), ("none", "all read out")):
        rows = []
        for k in keys:
            cells = [LABEL.get(k, k)]
            for band, _ in BANDS:
                v = S.get(band, {}).get(k, {}).get(rd, {}).get(w)
                cells += ([f"{v[0]:.1f}", f"{v[1]:.1f}", f"{v[2]:.1f}", f"{v[3]:.1f}", f"{v[4]:.3f}"] if v is not None
                          else ["–"] * 5)
            rows.append(cells)
        for c in range(1, len(rows[0])):
            vals = [float(r[c]) for r in rows if r[c] != "–"]
            if vals:
                best = max(vals)
                for r in rows:
                    r[c] = mark(r[c]) if r[c] != "–" and float(r[c]) == best else r[c]
        head = " | ".join(f"{bn}: Trig | 1° | 2.5° | 5° | New U" for _, bn in BANDS)
        md += ["", f"### {wname}, {rname}", "", f"| Layout | {head} |", "|---" * (1 + 5 * len(BANDS)) + "|"]
        md += ["| " + " | ".join(r) + " |" for r in rows]
GF = os.path.join(D, "ghost_scores.json")             # background-only windows (score_ghost.py), if scored
GH = json.load(open(GF))["scores"] if os.path.exists(GF) else {}
md += ["", "### Ghosts (all energies)", "",
       "Ghost det. = background-only detectors read out per shower (kernel showers + background, drawn counts); "
       "Ghost trig = share of background-only windows that trigger (≥ 3 read-out detectors with a particle; lower is "
       "better).", "",
       "| Layout | 1 µs all: ghost det. | ghost trig [%] | 1 µs kNN: ghost det. | ghost trig [%] | "
       "10 µs all: ghost det. | ghost trig [%] | 10 µs kNN: ghost det. | ghost trig [%] |", "|---" * 9 + "|"]
for k in keys:
    cells = [LABEL.get(k, k)]
    for w in ("1000", "10000"):
        for rd in ("none", "knn"):
            v = S.get("all", {}).get(k, {}).get(rd, {}).get(w)
            g = GH.get(k, {}).get(rd, {}).get(w, {}).get("ghost_trig")
            cells += [f"{v[5]:.2f}" if v is not None else "–", f"{100 * g:.1f}" if g is not None else "–"]
    md.append("| " + " | ".join(cells) + " |")
md += ["", "### Construction metrics", "",
       "Mean terrain slope under the detectors; share above the slope limit (the median slope under the grid); mean 3-D "
       "distance to the access point (lowest point of the region); RMS distance to the nearest of 20 pad centres "
       "(k-means); RMS height off the nearest of 8 constant-height rows; 3-D cable tree length. Lower is easier to build.",
       "", "| Layout | Slope [°] | > limit [%] | Access [m] | Pad RMS [m] | Row RMS [m] | Cable [km] |",
       "|---|---|---|---|---|---|---|"]
for k in keys:
    q = geo[k]
    md.append(f"| {LABEL.get(k, k)} | {q['slope_mean']:.1f} | {100 * q['slope_over']:.0f} | {q['access_m']:.0f} | "
              f"{q['pad_rms']:.0f} | {q['row_rms']:.1f} | {q['cable_km']:.2f} |")
out = os.path.join(D, f"quick_tables_notion{'' if args.recon == 'deepsets_count' else '_default'}.md")
open(out, "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("[write]", out)

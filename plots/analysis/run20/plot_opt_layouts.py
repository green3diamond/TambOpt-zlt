"""Run 20 (Malata): every validated layout (diagnostics/swgotrig/validate_count_layouts.txt: baselines, Stage 4 runs with
and without the cable cost) on the detector region, one figure and one table per noise level (no background, 1 us,
10 us). Each panel: the 266 faces (East-North, coloured by Up), the detectors, the 3-D minimum spanning tree linking
them (modules/optimize/cable.py) and its length, and the validated share of triggered showers within 1 / 2.5 / 5 deg
for the drawn-count recon (valcnt_count score files, all seeds and chunks pooled), all read out and with the kNN gate.
The tables (one file, a section per noise level) add the trigger rates; red = best in the column (Notion colorbox).
Writes <run 20>/diagnostics/opt_layouts_{nobg,1us,10us}.png and opt_layouts_tables_notion.md.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_opt_layouts.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "plots", "analysis", "run19", "swgotrig"))
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402
from modules.optimize.cable import cable_length, mst_edges, points_3d  # noqa: E402
from layout_names import layouts, run_info  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
OUTD = os.path.join(C.RUN_LOCATION, "diagnostics")
PATHS = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "validate_count_layouts.txt")) if "=" in l)
NOISE = [("nobg", "0", "No background"), ("1us", "1000", "1 µs background"), ("10us", "10000", "10 µs background")]
RUN, NSH = run_info(D, C.RUN_LOCATION)
GH = json.load(open(os.path.join(D, "ghost_scores.json")))["scores"]      # background-only windows


def shares(rd, w):
    """{layout: [% of triggered showers within 1, 2.5, 5 deg, trigger rate %, New U]}, drawn-count recon, all seeds and
    chunks pooled (all_X = triggered and within X deg, as a share of all showers; New U = u_swgo_drawn, over all)."""
    num, n = {}, {}
    for f in glob.glob(os.path.join(D, "valcnt_count_s*_*.json")):
        for k, by in json.load(open(f))["scores"].items():
            r = by[rd][w]
            num[k] = num.get(k, 0.0) + r["n_showers"] * np.array([r["all_1"], r["all_2.5"], r["all_5"], r["triggered"],
                                                                   r["u_swgo_drawn"]])
            n[k] = n.get(k, 0.0) + r["n_showers"]
    return {k: np.r_[100 * v[:3] / v[3], 100 * v[3] / n[k], v[4] / n[k]] for k, v in num.items()}


m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256)
verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)   # (F, 3, 3) ENU
panels = layouts(D, short=True)

geo = {}                                                       # layout -> (x, y, MST edges, cable km)
for key, _ in panels:
    lay = torch.load(PATHS[key], map_location="cpu")
    x, y = lay["x"].float(), lay["y"].float()
    with torch.no_grad():
        geo[key] = (x.numpy(), y.numpy(), mst_edges(points_3d(x, y, surface)),
                    float(cable_length(x, y, surface)[0]) / 1e3)

mark = lambda s: r"$\colorbox{#f8c4c4}{\textbf{" + s + "}}$"                                # noqa: E731
md = [f"## {RUN} layouts — triggered showers within 1° / 2.5° / 5° [%], drawn-count recon", "",
      f"{NSH:,} held-out showers, 5 count seeds pooled. 1° / 2.5° / 5° = share of TRIGGERED showers within that angle; "
      "Trig = share of ALL showers that trigger (≥ 3 read-out detectors with a drawn particle, background muons "
      "included); New U = mean[trig · (g_a + 0.3(1−g_a)q_a)] over ALL showers (with background it also counts "
      "background-only triggers). Ghost = share of background-only windows that trigger (same trigger, lower is better). Cable = 3-D minimum-spanning-tree "
      "length. Red = best in the column."]
ncol = 5
nrow = -(-len(panels) // ncol)
xl = (tri[..., 0].min() - 60, tri[..., 0].max() + 60)
yl = (tri[..., 1].min() - 60, tri[..., 1].max() + 60)
for tag, w, title in NOISE:
    allr, knn = shares("none", w), shares("knn", w)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 4.8 * nrow), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for ax, (key, name) in zip(axs.flat, panels):
        xe, ye, (i, j), L = geo[key]
        ax.add_collection(PolyCollection(tri[:, :, :2], array=tri[:, :, 2].mean(1), cmap="Greys", edgecolors="white",
                                         linewidths=0.3))
        ax.add_collection(LineCollection(np.stack([np.c_[xe[i], ye[i]], np.c_[xe[j], ye[j]]], 1),
                                         colors="#e07b39", linewidths=0.9))
        ax.scatter(xe, ye, s=10, c="#2a6fb0", lw=0, zorder=3)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        a, k = allr[key], knn[key]
        ax.set_title(f"{name}, cable {L:.2f} km\nall read out {a[0]:.1f} / {a[1]:.1f} / {a[2]:.1f}%\n"
                     f"kNN gate {k[0]:.1f} / {k[1]:.1f} / {k[2]:.1f}%", fontsize=8.5)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("North [m]")
    for ax in axs[-1]:
        ax.set_xlabel("East [m]")
    fig.suptitle(f"{RUN}, Malata, {title}: layouts (blue), 3-D cable tree (orange). Triggered showers within "
                 "1° / 2.5° / 5° [%], drawn-count recon, all read out and kNN gate", fontsize=11)
    out = os.path.join(OUTD, f"opt_layouts_{tag}.png")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print("[write]", out)

    bg = w != "0"                                               # ghosts only exist with background

    def block(sh, rd, key):
        """Trig, (Ghost), 1 / 2.5 / 5 deg of triggered, New U for one readout."""
        g = [f"{100 * GH[key][rd][w]['ghost_trig']:.1f}"] if bg else []
        return [f"{sh[key][3]:.1f}"] + g + [f"{v:.1f}" for v in sh[key][:3]] + [f"{sh[key][4]:.3f}"]
    rows = [[name, f"{geo[key][3]:.2f}"] + block(allr, "none", key) + block(knn, "knn", key) for key, name in panels]
    per = 6 if bg else 5
    for c in range(2, len(rows[0])):                             # best per metric column (ghosts: lowest)
        low = bg and (c - 2) % per == 1
        best = (min if low else max)(float(r[c]) for r in rows)
        for r in rows:
            r[c] = mark(r[c]) if float(r[c]) == best else r[c]
    head = ["Trig"] + (["Ghost"] if bg else []) + ["1°", "2.5°", "5°", "New U"]
    md += ["", f"### {title}", "",
           "| Layout | Cable [km] | All read out: " + " | ".join(head) + " | kNN gate: " + " | ".join(head) + " |",
           "|---" * (2 + 2 * len(head)) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]
out = os.path.join(OUTD, "opt_layouts_tables_notion.md")
open(out, "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("[write]", out)

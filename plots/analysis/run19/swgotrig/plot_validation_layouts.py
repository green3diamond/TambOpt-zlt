"""Run 19 swgo_trigger validation: the 11 validated layouts in one figure, from the exact files the validation scored
(<run 19>/diagnostics/swgotrig/validate_full_layouts.txt), annotated with where each scored best in the merged validation
(validate_full.json; values compared at the precision of the tables: % to 0.1, U to 0.001; red = outright best, red
outline = tied for best). Same East / North limits [m] and equal aspect on every panel, a dashed 3 km circle around the
plane centre, marker area proportional to detector area (2 m^2; the 20 m^2 run x10), an inset zoom (at most ±1 km around the
centroid) for layouts whose 90 % radius is under 2.5 km. Prints detector count and min / max East and North per layout.
Writes layouts_run19.png (300 dpi) and layouts_run19.pdf next to the scores.

    python plots/analysis/run19/swgotrig/plot_validation_layouts.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig")
LAYOUTS = [("disk_r6000_n99", "Uniform disk 6 km"), ("grid_n99", "Grid"),
           ("tri33_r10_n99", "Triplets r = 10 m on grid"), ("tri33_r25_n99", "Triplets r = 25 m on grid"),
           ("opt_a2_grid", "Opt. Poisson 2 m², grid start"), ("opt_a2_center", "Opt. Poisson 2 m², centre start"),
           ("opt_a20_grid", "Opt. Poisson 20 m², grid start"),
           ("opt_a2_step_grid_8000", "Opt. step 2 m², grid start (8,000 steps)"),
           ("opt_a2_step_center", "Opt. step 2 m², centre start"), ("disk_r1500_n99", "Disk 1.5 km"),
           ("tri33disk_R1000_r25_n99", "Triplets on 1 km disk")]
AREA = {"opt_a20_grid": 20.0}                                                 # m^2; all others 2
CONDS = [("none", "0", "no bkg all"), ("knn", "0", "no bkg kNN"), ("none", "1000", "1 µs all"),
         ("knn", "1000", "1 µs kNN"), ("none", "10000", "10 µs all"), ("knn", "10000", "10 µs kNN")]
METRICS = [("notrig_1", "1°", lambda v: round(100 * v, 1)), ("notrig_2.5", "2.5°", lambda v: round(100 * v, 1)),
           ("notrig_5", "5°", lambda v: round(100 * v, 1)), ("u_two_tier", "Old U", lambda v: round(v, 3)),
           ("u_swgo", "New U Poisson", lambda v: round(v, 3)), ("u_swgo_step", "New U step", lambda v: round(v, 3)),
           ("trig_poisson", "Trig", lambda v: round(100 * v, 1))]

paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "validate_full_layouts.txt")) if l.strip())
scores = json.load(open(os.path.join(D, "validate_full.json")))["scores"]

# wins[key] = {metric label: {"red": [conditions], "tie": [conditions]}}
wins = {k: {} for k, _ in LAYOUTS}
for rd, w, cname in CONDS:
    for mkey, mlab, rnd in METRICS:
        vals = {k: rnd(scores[k][rd][w][mkey]) for k, _ in LAYOUTS}
        best = max(vals.values())
        top = [k for k, v in vals.items() if v == best]
        for k in top:
            wins[k].setdefault(mlab, {"red": [], "tie": []})["red" if len(top) == 1 else "tie"].append(cname)


NOISELESS = ("no bkg all", "no bkg kNN")


def win_lines(k):
    """[(text, kind, green)]: one line per metric, kind (red = outright, tie) and colour group. Green = the trigger
    rate, or a win in a no-background condition; red = a win with background."""
    out = []
    for mlab, kinds in wins[k].items():
        for kind in ("red", "tie"):
            c = kinds[kind]
            groups = ([(c, True)] if mlab == "Trig" else
                      [([x for x in c if x in NOISELESS], True), ([x for x in c if x not in NOISELESS], False)])
            for cs, green in groups:
                if cs:
                    out.append((mlab, tuple(cs), kind, green))
    merged = {}                                                               # metrics with the same conditions share a line
    for mlab, cs, kind, green in out:
        merged.setdefault((cs, kind, green), []).append(mlab)
    lines = [(f"{', '.join(ms)}: {'all 6' if len(cs) == 6 else ', '.join(cs)}", kind, green)
             for (cs, kind, green), ms in merged.items()]
    return lines or [("no wins", None, False)]


m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2]
centre = mesh.mean(0)
xy = {}
print(f"{'layout':42s} {'n':>4s} {'E min':>8s} {'E max':>8s} {'N min':>8s} {'N max':>8s}  file")
for k, name in LAYOUTS:
    L = torch.load(paths[k], map_location="cpu")
    e, n = np.asarray(L["x"], float), np.asarray(L["y"], float)
    xy[k] = (e, n)
    print(f"{name:42s} {e.size:4d} {e.min():8.0f} {e.max():8.0f} {n.min():8.0f} {n.max():8.0f}  {paths[k]}")
allx = np.concatenate([mesh[:, 0]] + [v[0] for v in xy.values()])
ally = np.concatenate([mesh[:, 1]] + [v[1] for v in xy.values()])
pad = 300.0
xlim = (min(allx.min(), centre[0] - 3000) - pad, max(allx.max(), centre[0] + 3000) + pad)
ylim = (min(ally.min(), centre[1] - 3000) - pad, max(ally.max(), centre[1] + 3000) + pad)
th = np.linspace(0, 2 * np.pi, 361)

fig, axs = plt.subplots(4, 3, figsize=(16, 21))
for ax, (k, name) in zip(axs.ravel(), LAYOUTS):
    e, n = xy[k]
    s = 2.0 * AREA.get(k, 2.0)                                                # marker area ~ detector area
    ax.scatter(mesh[:, 0], mesh[:, 1], s=0.2, color="#e3e1da", lw=0, rasterized=True)
    ax.plot(centre[0] + 3000 * np.cos(th), centre[1] + 3000 * np.sin(th), "--", color="#7a7974", lw=0.7)
    ax.scatter(e, n, s=s, color="#2a78d6", lw=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("East [m]", fontsize=8); ax.set_ylabel("North [m]", fontsize=8)
    lines = win_lines(k)
    ax.set_title(f"{name} ({e.size} detectors)", fontsize=10, fontweight="bold", pad=8 + 13 * len(lines))
    for i, (txt, kind, green) in enumerate(lines):                           # one line per 13 pt above the axes
        kw = dict(xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 4 + 13 * (len(lines) - 1 - i)),
                  textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        col = "#1e8449" if green else "#c0392b"
        if kind == "red":
            ax.annotate(txt, color=col, fontweight="bold", **kw)
        elif kind == "tie":
            ax.annotate(txt, color=col, fontweight="bold" if green else "normal",
                        bbox=dict(boxstyle="round,pad=0.12", fc="none", ec=col, lw=1.0), **kw)
        else:
            ax.annotate(txt, color="#7a7974", **kw)
    c = np.array([e.mean(), n.mean()])
    r = np.hypot(e - c[0], n - c[1])
    if np.percentile(r, 90) < 2500:                                           # compact layout: inset zoom
        half = min(r.max() * 1.08, 1000.0)                                    # at most +-1 km, so clusters show
        ins = inset_axes(ax, width="36%", height="36%", loc="upper right", borderpad=0.4)
        ins.scatter(mesh[:, 0], mesh[:, 1], s=1.0, color="#e3e1da", lw=0, rasterized=True)
        ins.scatter(e, n, s=s * 2.5, color="#2a78d6", lw=0)
        ins.set_xlim(c[0] - half, c[0] + half); ins.set_ylim(c[1] - half, c[1] + half); ins.set_aspect("equal")
        ins.tick_params(labelsize=5)
        ins.set_title(f"zoom ±{half / 1e3:.1f} km around the centroid", fontsize=6, pad=2)
axs.ravel()[-1].axis("off")
axs.ravel()[-1].text(0.02, 0.62,
                     "Utilities, per shower, averaged over all showers\n"
                     "(a = angle error [deg], e = |dlog10 E|):\n\n"
                     "1° / 2.5° / 5°: share within that angle\n"
                     "  (no trigger)\n"
                     "Old U (two-tier): g + 0.3 (1-g) q\n"
                     "  g = sig(5-a) sig((0.2-e)/0.05)\n"
                     "  q = 1/(1 + (a/20)^2 + (e/0.5)^2)\n"
                     "New U (swgo_trigger):\n"
                     "  r_b (g_a + 0.3 (1-g_a) q_a)\n"
                     "  g_a = sig(5-a), q_a = 1/(1+(a/20)^2)\n"
                     "  Poisson: r_b = P(>=3 det. with >=1\n"
                     "    particle), p_i = 1 - exp(-mu_i)\n"
                     "  step: r_b = sig(5 (sum sig(10(mu_i-1)) - 3))\n"
                     "Trig: Poisson r_b, averaged\n"
                     "mu_i: expected particles, 2 m^2 detector",
                     fontsize=8, va="top", family="monospace", transform=axs.ravel()[-1].transAxes)
axs.ravel()[-1].text(0.02, 1.15, "Under each title: where the layout scored best\nin the run 19 validation\n"
                     "(21,060 held-out showers; 6 conditions:\n"
                     "no / 1 µs / 10 µs background x all read out / kNN gate).\n"
                     "Bold = outright best; outline = tied (table precision).\n"
                     "Green = trigger rate, or no-background conditions;\n"
                     "red = conditions with background.\n"
                     "Dashed circle: 3 km around the plane centre.\n"
                     "Marker area ~ detector area (2 m²; 20 m² run x10).\n"
                     "Grey: the run 19 plane mesh.", fontsize=8, va="top", transform=axs.ravel()[-1].transAxes)
fig.suptitle("Run 19 validation layouts (exact files scored)", fontsize=15, y=0.995)
fig.subplots_adjust(left=0.06, right=0.98, top=0.915, bottom=0.04, hspace=0.68, wspace=0.28)
for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
    out = os.path.join(D, f"layouts_run19.{ext}")
    fig.savefig(out, facecolor="white", **kw)
    print("[write]", out)

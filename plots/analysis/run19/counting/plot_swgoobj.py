"""Run 19 check of the SWGO-like objective without training (job 54622674): fixed layouts, expected-count trigger
(>= 3 detectors with mu >= 1), noise-free recon input. Reads swgoobj_S{1500,3000,0}.json and swgoobj_layouts.txt in
<RUN_LOCATION>/diagnostics/counting/ and writes there:
    swgoobj_layouts.png   the scored layouts on the run 19 plane mesh (East, North, km)
    swgoobj_results.png   U5 (triggered and within 5 deg, share of all showers in the set) vs disk radius, and the
                          same without the trigger; grid and triangles r = 10 / 100 m as markers

    python plots/analysis/run19/counting/plot_swgoobj.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.join(C.RUN_LOCATION, "diagnostics", "counting")
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3
L = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "swgoobj_layouts.txt")) if l.strip())


def xy(p):
    d = torch.load(p, map_location="cpu")
    return np.asarray(d["x"], float) / 1e3, np.asarray(d["y"], float) / 1e3


# ── layouts ──
SHOW = [(f"disk_r{R}_n99", f"disk R = {R / 1e3:g} km", "#2a78d6") for R in (300, 500, 1000, 1500, 2000, 3000, 6000)] + \
       [("grid_n99", "grid", "#2a78d6"), ("tri33disk_R1000_r10_n99", "33 triangles r = 10 m, R = 1 km", "#c0392b"),
        ("tri33disk_R1000_r100_n99", "33 triangles r = 100 m, R = 1 km", "#c0392b")]
# third row: zoom on the centre (+-0.45 km), where the r = 10-100 m triangles are resolved
ZOOM = [("disk_r1000_n99", "disk R = 1 km (zoom)", "#2a78d6")] + \
       [(f"tri33disk_R1000_r{r}_n99", f"triangles r = {r} m, R = 1 km (zoom)", "#c0392b") for r in (10, 25, 50, 100)]
c0 = np.mean(xy(L["disk_r1000_n99"]), axis=1)
fig, axs = plt.subplots(3, 5, figsize=(22, 14.4))
for i, (ax, (k, t, col)) in enumerate(zip(axs.ravel(), SHOW + ZOOM)):
    ax.scatter(mesh[:, 0], mesh[:, 1], s=0.3 if i < 10 else 4, color="#e3e1da", lw=0, rasterized=True)
    e, n = xy(L[k])
    ax.scatter(e, n, s=8 if i < 10 else 22, color=col, lw=0)
    if i >= 10:
        ax.set_xlim(c0[0] - 0.45, c0[0] + 0.45); ax.set_ylim(c0[1] - 0.45, c0[1] + 0.45)
    ax.set_aspect("equal"); ax.set_title(t, fontsize=11); ax.tick_params(labelsize=8)
for ax in axs[:, 0]:
    ax.set_ylabel("North [km]")
for ax in axs[2]:
    ax.set_xlabel("East [km]")
fig.suptitle("Objective check: 99 detectors per layout (real layouts on the run 19 plane; also scored: disks 100 / 200 / 750 m, "
             "4.5 km, triangles r = 25 / 50 m at R = 0.5 / 0.75 / 1 km, annuli)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(D, "swgoobj_layouts.png"), dpi=110, facecolor="white"); plt.close(fig)

# ── results ──
SETS = [(1500, "showers within 1.5 km", "#2a78d6"), (3000, "within 3 km", "#e08a00"), (0, "whole plane", "#52514e")]
RS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000)
fig, axs = plt.subplots(1, 2, figsize=(16, 6.2))
for S, lab, col in SETS:
    sc = json.load(open(os.path.join(D, f"swgoobj_S{S}.json")))["scores"]
    g = lambda k, f: 100 * sc[k]["none"]["0"][f]                                       # noqa: E731
    n = sc["grid_n99"]["none"]["0"]["n_showers"]
    for ax, f in zip(axs, ("allmu_5", "notrig_5")):
        ax.plot([r / 1e3 for r in RS], [g(f"disk_r{r}_n99", f) for r in RS], "-o", color=col, ms=5,
                label=f"{lab} ({n:,}): even disks")
        ax.axhline(g("grid_n99", f), color=col, ls=":", lw=1.4)
        for r, mk in ((10, "^"), (100, "v")):
            ax.plot([R / 1e3 for R in (500, 750, 1000)], [g(f"tri33disk_R{R}_r{r}_n99", f) for R in (500, 750, 1000)],
                    mk, color=col, ms=8, mfc="none", mew=1.5)
for ax, t in zip(axs, ("With trigger: triggered (≥ 3 det. with μ ≥ 1) and within 5°", "Without trigger: within 5°")):
    ax.set_xscale("log"); ax.set_xlabel("disk radius R [km]"); ax.set_ylabel("share of all showers in the set [%]")
    ax.set_title(t, fontsize=12); ax.grid(alpha=0.3, which="both")
axs[0].plot([], [], "^", color="k", mfc="none", label="33 triangles r = 10 m"); axs[0].plot([], [], "v", color="k", mfc="none", label="33 triangles r = 100 m")
axs[0].plot([], [], ":", color="k", label="grid (dotted, per set)")
axs[0].legend(fontsize=9, loc="upper left")
fig.suptitle("SWGO-like objective, 99 detectors, noise-free recon input (job 54622674)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, "swgoobj_results.png"), dpi=110, facecolor="white"); plt.close(fig)
print("[write]", D, "swgoobj_layouts.png swgoobj_results.png")

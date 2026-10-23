"""Run 19 SWGO-like objective check, donuts and detector area (jobs 54627380, 54628915, 54629473): fixed layouts, trigger
>= 3 detectors with mu >= mu_min, noise-free recon input. mu_min = 1 / 0.1 / 0.01 on the 2 m^2 expected counts is a
2 / 20 / 200 m^2 detector for the trigger (the recon input is unchanged). Reads swgoobjE_S*.json (2 m^2, with energy),
area_mu0p1_S*.json, area_mu0p01_S*.json in <RUN_LOCATION>/diagnostics/counting/ and writes there:
    swgoobj_donut_layouts.png  the 8 donuts (outer 1.5 / 2 km) and the two full disks, zoomed to +-2.2 km
    swgoobj_area.png           triggered and within 5 deg (solid) and also |dlog10 E| < 0.2 (dashed) vs disk radius,
                               per shower set, for the three areas; the grid as a dotted line
    swgoobj_donuts.png         donuts vs inner radius (r_in = 0 = full disk): within 5 deg (top) and 1 deg (bottom)

    python plots/analysis/run19/counting/plot_swgoobj_area.py
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
AREAS = [("2 m²", "swgoobjE", "#2a78d6"), ("20 m²", "area_mu0p1", "#e08a00"), ("200 m²", "area_mu0p01", "#c0392b")]
SETS = [(1500, "showers within 1.5 km (518)"), (3000, "within 3 km (1,784)"), (0, "whole plane (5,000)")]
RS = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4500, 6000)
DON = {1500: (0, 250, 500, 750, 1000), 2000: (0, 250, 500, 1000, 1500)}
SC = {(f, S): json.load(open(os.path.join(D, f"{f}_S{S}.json")))["scores"] for _, f, _ in AREAS for S, _ in SETS}


def g(f, S, k, m):
    return 100 * SC[(f, S)][k]["none"]["0"][m]


def name(Ro, ri):
    return f"disk_r{Ro}_n99" if ri == 0 else f"annulus_rin{ri}_R{Ro}_n99"


# ── donut layouts ──
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3
L = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "swgoobjE_layouts.txt")) if l.strip())
fig, axs = plt.subplots(2, 5, figsize=(22, 9.4))
for i, Ro in enumerate(DON):
    for j, ri in enumerate(DON[Ro]):
        d = torch.load(L[name(Ro, ri)], map_location="cpu")
        e, n = np.asarray(d["x"], float) / 1e3, np.asarray(d["y"], float) / 1e3
        if i == 0 and j == 0:
            c0 = (e.mean(), n.mean())
        ax = axs[i, j]
        ax.scatter(mesh[:, 0], mesh[:, 1], s=2, color="#e3e1da", lw=0, rasterized=True)
        ax.scatter(e, n, s=14, color="#2a78d6" if ri == 0 else "#7b3fb5", lw=0)
        ax.set_xlim(c0[0] - 2.2, c0[0] + 2.2); ax.set_ylim(c0[1] - 2.2, c0[1] + 2.2); ax.set_aspect("equal")
        ax.set_title(f"disk R = {Ro / 1e3:g} km" if ri == 0 else f"donut {ri / 1e3:g}–{Ro / 1e3:g} km", fontsize=11)
        ax.tick_params(labelsize=8)
    axs[i, 0].set_ylabel("North [km]")
for ax in axs[1]:
    ax.set_xlabel("East [km]")
fig.suptitle("Donuts at the scale of the best disk: 99 detectors evenly over r_in – R (real layouts on the run 19 plane)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, "swgoobj_donut_layouts.png"), dpi=110, facecolor="white"); plt.close(fig)

# ── area: score vs disk radius ──
fig, axs = plt.subplots(1, 3, figsize=(20, 6.2))
for ax, (S, lab) in zip(axs, SETS):
    for a, f, col in AREAS:
        x = [r / 1e3 for r in RS]
        ax.plot(x, [g(f, S, f"disk_r{r}_n99", "allmu_5") for r in RS], "-o", color=col, ms=5, label=f"{a}: within 5°")
        ax.plot(x, [g(f, S, f"disk_r{r}_n99", "allmu_5e") for r in RS], "--", color=col, lw=1.3, label=f"{a}: and energy")
        ax.axhline(g(f, S, "grid_n99", "allmu_5"), color=col, ls=":", lw=1.4)
    ax.set_xscale("log"); ax.set_xlabel("disk radius R [km]"); ax.set_title(lab, fontsize=12); ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel("triggered and within 5° [% of all showers in the set]")
axs[0].plot([], [], ":", color="k", label="grid (dotted, per area)")
axs[0].legend(fontsize=9, loc="upper left")
fig.suptitle("Detector area: the best disk grows with the area (trigger ≥ 3 detectors with ≥ 1 expected particle; "
             "energy = |Δlog₁₀E| < 0.2)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, "swgoobj_area.png"), dpi=110, facecolor="white"); plt.close(fig)

# ── donuts vs inner radius ──
fig, axs = plt.subplots(2, 3, figsize=(20, 10.5))
for c, (S, lab) in enumerate(SETS):
    for row, mt in enumerate(("allmu_5", "allmu_1")):
        ax = axs[row, c]
        for a, f, col in AREAS:
            for Ro, ls in ((1500, "-o"), (2000, "--s")):
                ax.plot([ri / 1e3 for ri in DON[Ro]], [g(f, S, name(Ro, ri), mt) for ri in DON[Ro]], ls, color=col, ms=5,
                        label=f"{a}, outer {Ro / 1e3:g} km")
        ax.set_xlabel("inner radius r_in [km] (0 = full disk)"); ax.grid(alpha=0.3)
        ax.set_title(f"{lab}: within {'5' if row == 0 else '1'}°", fontsize=12)
    axs[0, c].set_ylim(bottom=0); axs[1, c].set_ylim(bottom=0)
axs[0, 0].set_ylabel("triggered and within 5° [%]"); axs[1, 0].set_ylabel("triggered and within 1° [%]")
axs[0, 0].legend(fontsize=9, loc="lower left", ncol=2)
fig.suptitle("Donuts: a hole does not pay at 5°; a small hole (0.25 km) is slightly ahead at 1° for every area", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, "swgoobj_donuts.png"), dpi=110, facecolor="white"); plt.close(fig)
print("[write]", D, "swgoobj_donut_layouts.png swgoobj_area.png swgoobj_donuts.png")

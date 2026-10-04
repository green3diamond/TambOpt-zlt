"""Core test figure: 100-detector layouts (top) and held-out kernel scores (bottom):
share of showers good at 5 deg and at 2.5 deg (|dlog10 E| < 0.2), and median angle error."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-plane (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
res = json.load(open(os.path.join(D, "core_test_scores.json")))
paths = dict(l.strip().split("=", 1) for l in open(os.path.join(D, "layouts.txt")) if l.strip())
names = [n for n in paths if n in res]
KIND = {"grid": "#b9b7af"}
for n in names:
    KIND.setdefault(n, "#eb6834" if n.startswith("core_r") else "#eda100" if n.startswith("core")
                    else "#1baf7a" if n.startswith("kern_") else "#2a78d6")
LABEL = {"grid": "grid", "core_r100": "core r=100 m", "core_r200": "core r=200 m", "core_r300": "core r=300 m",
         "core_r500": "core r=500 m", "core300_60_spread40": "60 in 300 m core\n+ 40 spread",
         "core300_40_spread60": "40 in 300 m core\n+ 60 spread", "core300_60_rim40": "60 in 300 m core\n+ 40 on rim",
         "core200_50_spread50": "50 in 200 m core\n+ 50 spread", "opt_current_grid": "optimized, current\n(grid start)",
         "opt_current_center": "optimized, current\n(center start)",
         "opt_twotier_l0p3_grid": "optimized, two-tier\nλ=0.3 (grid start)",
         "opt_twotier_l0p3_center": "optimized, two-tier\nλ=0.3 (center start)"}
for tag, txt in (("current", "current"), ("twotier_l0", "two-tier λ=0"), ("twotier_l0p3", "two-tier λ=0.3")):
    for start in ("grid", "center"):
        LABEL[f"kern_{tag}_{start}"] = f"kernel-optimized,\n{txt} ({start})"

mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(mountain.centroids_ENU)

rows = [("frac_good_5", "showers good at 5°, |Δlog10 E| < 0.2", 100, "%"),
        ("frac_good_t", "showers good at 2.5°, |Δlog10 E| < 0.2", 100, "%"),
        ("ang_med", "median angle error [deg]", 1, "°")]


def draw(sel, title, out, wide=False):
    """Layouts (top) and the three metrics (bars) for the layouts in `sel`, grid first."""
    w_col = 1.45 if wide else 2.4
    fs = 8 if wide else 10
    fig = plt.figure(figsize=(max(w_col * len(sel) + 1, 12), 10))
    gs = fig.add_gridspec(4, len(sel), height_ratios=[1.3, 1, 1, 1], hspace=0.45, wspace=0.08)
    for i, n in enumerate(sel):
        a = fig.add_subplot(gs[0, i])
        L = torch.load(paths[n], map_location="cpu")
        a.scatter(mesh[:, 0], mesh[:, 1], s=1.5, color="#e8e6df", lw=0)
        a.scatter(np.asarray(L["x"]), np.asarray(L["y"]), s=5 if wide else 9, color=KIND[n], lw=0)
        a.set_title(LABEL.get(n, n), fontsize=fs, color=INK)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for r, (key, rtitle, scale, unit) in enumerate(rows):
        a = fig.add_subplot(gs[r + 1, :])
        vals = [res[n][key] * scale for n in sel]
        x = np.arange(len(sel))
        a.bar(x, vals, color=[KIND[n] for n in sel], width=0.7 if wide else 0.6)
        for xx, v in zip(x, vals):
            a.text(xx, v, f"{v:.1f}{unit}", ha="center", va="bottom", fontsize=fs, color=GREY)
        a.axhline(res["grid"][key] * scale, color="#b9b7af", ls="--", lw=1)
        a.set_ylim(0, max(vals) * 1.2)                 # headroom for the value labels
        a.set_xticks(x)
        if wide:
            a.set_xticklabels([LABEL.get(n, n).replace("\n", " ") for n in sel], fontsize=7.5, rotation=25, ha="right")
        else:
            a.set_xticklabels([LABEL.get(n, n) for n in sel], fontsize=9)
        a.set_title(rtitle + "   (dashed: grid)", fontsize=fs + 1, color=INK, loc="left")
        a.grid(axis="y", alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        if r < 2:
            a.set_xticklabels([])
    fig.suptitle(title, fontsize=12, color=INK)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")


EV = "1,536 held-out kernel showers"
draw(names, "Core test, 100 detectors (run 17, 45° plane): hand-made cores and hybrids vs grid and kernel-optimized "
     f"layouts, {EV}\ngreen: optimized on the kernel (kernel in the loop) · all scored with the recon trained on kernel responses",
     os.path.join(D, "core_test.png"), wide=True)
groups = [("cores", "compact cores at the plane centre", lambda n: n.startswith("core_r")),
          ("hybrids", "core + spread / rim hybrids", lambda n: n.startswith("core") and not n.startswith("core_r")),
          ("kernel_optimized", "layouts optimized on the kernel (kernel in the loop)", lambda n: n.startswith("kern_"))]
for tag, what, keep in groups:
    draw(["grid"] + [n for n in names if keep(n)],
         f"Core test, 100 detectors (run 17, 45° plane): {what} vs grid · {EV}\n"
         "scored with the recon trained on 100% kernel responses",
         os.path.join(D, f"core_test_{tag}.png"))

"""Plots for opt_scores.json: the optimised layouts and their starts (opt_layouts.png), the share of all showers
within 1 / 2.5 / 5 deg (opt_accuracy.png) and the share of detected showers only (opt_accuracy_detected.png)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

O = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{O}/opt_scores.json"))
COLS = (("start", "start", "#9b9a95"), ("current", "current", "#2a78d6"), ("twotier_l0", "two-tier λ 0", "#1baf7a"),
        ("twotier_l0p3", "two-tier λ 0.3", "#eb6834"), ("threetier", "three-tier", "#8a5cd1"))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3

fig, axs = plt.subplots(2, 5, figsize=(26, 11))
for i, scheme in enumerate(("grid", "center")):
    for j, (key, lab, col) in enumerate(COLS):
        ax = axs[i, j]; name = f"{key}_{scheme}"
        ax.scatter(*mesh.T, s=0.4, color="#e4e3dd", lw=0)
        ax.set_xlim(mesh[:, 0].min() - 0.3, mesh[:, 0].max() + 0.3); ax.set_ylim(mesh[:, 1].min() - 0.3, mesh[:, 1].max() + 0.3)
        ax.set_aspect("equal"); ax.tick_params(labelsize=8)
        if name not in S:
            ax.set_title(f"{lab} · {scheme} start\n(still running)", fontsize=11, color="#9b9a95"); continue
        r = S[name]; L = torch.load(r["path"], map_location="cpu")
        ax.scatter(np.asarray(L["x"]) / 1e3, np.asarray(L["y"]) / 1e3, s=9, color=col, lw=0)
        ax.set_title(f"{lab} · {scheme} start\n1°: {100*r['all_1']:.1f}%  2.5°: {100*r['all_2.5']:.1f}%  5°: {100*r['all_5']:.1f}%  "
                     f"med {r['ang_med']:.2f}°\ndetected (≥ 3 hit) {100*r['detected']:.1f}%", fontsize=10.5)
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.5)
fig.suptitle("Run 19 optimised layouts (100 detectors, 20,000 L-BFGS steps) and their starts · 5,000 held-out test showers", fontsize=15)
fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.04, wspace=0.18, hspace=0.42); fig.savefig(f"{O}/opt_layouts.png", dpi=85, facecolor="white"); plt.close(fig)


def bars(prefix, fname, ylab, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6.4), sharey=True)
    for ax, scheme in zip(axs, ("grid", "center")):
        names = [(k, lab, col) for k, lab, col in COLS if f"{k}_{scheme}" in S]
        x = np.arange(3); w = 0.8 / max(len(names), 1)
        for j, (k, lab, col) in enumerate(names):
            r = S[f"{k}_{scheme}"]
            v = [100 * r[f"{prefix}_{t}"] for t in ("1", "2.5", "5")]
            xs = x + (j - (len(names) - 1) / 2) * w
            extra = f" (det. {100 * r['detected']:.0f}%)" if prefix == "det" else ""
            ax.bar(xs, v, w * 0.92, color=col, label=lab + extra)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 1, f"{vv:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(["within 1°", "within 2.5°", "within 5°"])
        ax.set(title=f"{scheme} start", ylabel=ylab); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(); fig.savefig(f"{O}/{fname}", dpi=105, facecolor="white"); plt.close(fig)


bars("all", "opt_accuracy.png", "share of all showers [%]", "Run 19 optimisations: share of all 5,000 held-out showers within 1 / 2.5 / 5°")
bars("det", "opt_accuracy_detected.png", "share of detected showers [%]",
     "Run 19 optimisations: share of DETECTED showers only (≥ 3 detectors with > 1 particle) within 1 / 2.5 / 5°")
print("[write] opt_layouts.png opt_accuracy.png opt_accuracy_detected.png")

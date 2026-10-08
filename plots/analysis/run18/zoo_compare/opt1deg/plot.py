"""1 deg optimisations vs their grid / centre starts: layouts and the share of held-out showers within
1 / 2.5 / 5 deg (bars: relative-time recon used in the optimisation; dots: wd 0.03 recon; error bars:
binomial 1 sigma on 1,536 showers). Writes opt1deg.png and opt1deg.json."""
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
TH = (1, 2.5, 5)
NB = 1536
LAY = dict(l.strip().split("=", 1) for l in open(f"{O}/layouts.txt") if l.strip())
S = {tag: {th: json.load(open(f"{O}/{tag}/scores_pad_{th:g}deg.json")) for th in TH} for tag in ("reltime", "wd0.03")}
ITEMS = (("init_grid", "grid start", "#9cc0ec"), ("opt_grid", "optimised from grid", "#2a78d6"),
         ("init_center", "centre start", "#e6a3bb"), ("opt_center", "optimised from centre", "#c23b6b"))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU) / 1e3

fig = plt.figure(figsize=(22, 11))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.15, 2.3], wspace=0.08, hspace=0.35)
out = {}
for row, n in enumerate((100, 10)):
    for j, (key, lab, col) in enumerate(ITEMS):
        name = f"{key}_n{n}"
        L = torch.load(LAY[name], map_location="cpu")
        ax = fig.add_subplot(gs[row, j])
        ax.scatter(mesh[:, 0], mesh[:, 1], s=0.6, color="#ecebe6", lw=0)
        ax.scatter(np.asarray(L["x"]) / 1e3, np.asarray(L["y"]) / 1e3, s=10 if n == 100 else 30, color=col, lw=0)
        ax.set_xlim(mesh[:, 0].min() - 0.2, mesh[:, 0].max() + 0.2); ax.set_ylim(mesh[:, 1].min() - 0.2, mesh[:, 1].max() + 0.2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        r = S["reltime"]
        ax.set_title(f"{n} det. · {lab}\n1°: {100 * r[1][name]['frac_good_t']:.1f}%   2.5°: "
                     f"{100 * r[2.5][name]['frac_good_t']:.1f}%   5°: {100 * r[5][name]['frac_good_t']:.1f}%\n"
                     f"median {r[1][name]['ang_med']:.2f}°", fontsize=10.5)
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.5)
    ax = fig.add_subplot(gs[row, 5])
    x = np.arange(len(TH)); w = 0.2
    for j, (key, lab, col) in enumerate(ITEMS):
        name = f"{key}_n{n}"
        p = np.array([S["reltime"][th][name]["frac_good_t"] for th in TH])
        q = np.array([S["wd0.03"][th][name]["frac_good_t"] for th in TH])
        xs = x + (j - 1.5) * w
        ax.bar(xs, 100 * p, w * 0.92, color=col, label=lab, yerr=100 * np.sqrt(p * (1 - p) / NB), capsize=2,
               error_kw=dict(lw=0.8))
        ax.scatter(xs, 100 * q, s=22, color="white", edgecolor="#1f2328", lw=1, zorder=3,
                   label="wd 0.03 recon" if j == 0 else None)
        for xx, pp in zip(xs, p):
            ax.text(xx, 100 * pp + 1.5, f"{100 * pp:.1f}", ha="center", fontsize=8)
        out[name] = {"reltime": {f"{th:g}": float(v) for th, v in zip(TH, p)},
                     "wd0.03": {f"{th:g}": float(v) for th, v in zip(TH, q)},
                     "median_reltime": S["reltime"][1][name]["ang_med"], "median_wd0.03": S["wd0.03"][1][name]["ang_med"]}
    ax.set_xticks(x); ax.set_xticklabels([f"within {th:g}°" for th in TH])
    ax.set(ylabel="share of held-out showers [%]", ylim=(0, 100), title=f"{n} detectors")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("1° angle-only optimisation (two-tier, λ = 0, 5k steps, relative-time recon) vs its grid and centre starts"
             " · 1,536 held-out showers", fontsize=14)
fig.savefig(f"{O}/opt1deg.png", dpi=110, facecolor="white", bbox_inches="tight")
json.dump(out, open(f"{O}/opt1deg.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:16s} " + "  ".join(f"{th}°: {100 * v['reltime'][th]:5.1f} ({100 * v['wd0.03'][th]:5.1f})" for th in v["reltime"])
          + f"  median {v['median_reltime']:.2f}° ({v['median_wd0.03']:.2f}°)")
print("[write] opt1deg.png opt1deg.json")

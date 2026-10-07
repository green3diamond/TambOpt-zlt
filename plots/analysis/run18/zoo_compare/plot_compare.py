"""Old recon (5 layout strategies) vs zoo recon (9 strategies) on the same held-out showers and layouts.

Reads <tag>/<set>/scores_pad_<theta>deg.json (tag = old | zoo; set = core_test_ndet, clusters, angleonly)
and <tag>/detcount/detector_count_grid.json. Writes:
  compare_scatter.png   every layout: share within 1 / 2.5 / 5 deg, old (x) vs zoo (y), coloured by family
  compare_best.png      best layout per family and detector count, old vs zoo, within 1 and 5 deg
  compare_detcount.png  grid with N detectors: share within 1 / 2.5 / 5 deg vs N, old vs zoo
  compare_summary.json  the numbers behind the best-per-family panel
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
SETS = ("core_test_ndet", "clusters", "angleonly")
THETAS = (1, 2.5, 5)
INK, GREY = "#1f2328", "#52514e"
FAM_COL = {"grid": "#2a78d6", "disk": "#1baf7a", "cluster": "#eb6834", "core": "#c23b6b",
           "hybrid": "#8a5cd1", "optimized": "#7a7a7a"}
REC_COL = {"old": "#9aa4b1", "zoo": "#2a78d6"}


def family(name):
    if name.startswith("kern_"):
        return "optimized"
    if name.startswith("grid"):
        return "grid"
    if name.startswith("disk"):
        return "disk"
    if name.startswith("clu"):
        return "cluster"
    if re.match(r"core_r\d+", name):
        return "core"
    return "hybrid"


def n_det(name, rec):
    m = re.search(r"_n(\d+)$", name)
    return int(m.group(1)) if m else int(rec.get("n_det", 100))


def load(tag):
    out = {}
    for s in SETS:
        for th in THETAS:
            f = os.path.join(Z, tag, s, f"scores_pad_{th:g}deg.json")
            for name, rec in json.load(open(f)).items():
                key = (s, name)
                r = out.setdefault(key, dict(name=name, fam=family(name), n=n_det(name, rec)))
                r[th] = rec["frac_good_t"]
                r["ang_med"] = rec["ang_med"]
    return out


old, zoo = load("old"), load("zoo")
keys = sorted(set(old) & set(zoo))

# 1. scatter
fig, axs = plt.subplots(1, 3, figsize=(19, 6.6))
marks = {20: "o", 50: "s", 100: "^"}
for ax, th in zip(axs, THETAS):
    for k in keys:
        o, z = old[k], zoo[k]
        ax.scatter(100 * o[th], 100 * z[th], s=34, color=FAM_COL[o["fam"]], marker=marks.get(o["n"], "D"),
                   alpha=0.85, lw=0)
    lim = max(max(100 * old[k][th] for k in keys), max(100 * zoo[k][th] for k in keys)) * 1.06
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="old recon: share within θ [%]",
           ylabel="zoo recon: share within θ [%]", title=f"within {th:g}°")
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
h = [plt.Line2D([], [], color=c, marker="o", ls="", label=f) for f, c in FAM_COL.items()]
h += [plt.Line2D([], [], color=GREY, marker=m, ls="", label=f"{n} det.") for n, m in marks.items()]
h += [plt.Line2D([], [], color=GREY, marker="D", ls="", label="other N (grid_nN)")]
axs[2].legend(handles=h, fontsize=9, frameon=False, loc="lower right")
fig.suptitle("Every test layout, scored by both recons on the same 1,536 held-out showers (above the line = zoo better)",
             fontsize=13, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(Z, "compare_scatter.png"), dpi=120, facecolor="white", bbox_inches="tight")

# 2. best per family and N
fams = ["grid", "disk", "cluster", "core", "hybrid", "optimized"]
summary = {}
for tag, d in (("old", old), ("zoo", zoo)):
    for k in keys:
        r = d[k]
        if r["n"] not in (20, 50, 100):
            continue
        for th in THETAS:
            cell = summary.setdefault(f"{r['fam']}|{r['n']}|{th:g}", {})
            if r[th] > cell.get(tag, (-1, ""))[0]:
                cell[tag] = (r[th], r["name"])
fig, axs = plt.subplots(2, 3, figsize=(19, 9), sharey="row")
for i, th in enumerate((1, 5)):
    for j, n in enumerate((20, 50, 100)):
        ax = axs[i, j]
        fs = [f for f in fams if f"{f}|{n}|{th:g}" in summary]
        x = np.arange(len(fs))
        for dx, tag in ((-0.2, "old"), (0.2, "zoo")):
            v = [100 * summary[f"{f}|{n}|{th:g}"].get(tag, (np.nan,))[0] for f in fs]
            ax.bar(x + dx, v, 0.38, color=REC_COL[tag], label=f"{tag} recon")
            for xi, vi in zip(x + dx, v):
                ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(fs)
        ax.set_title(f"{n} detectors · best layout within {th:g}°", fontsize=11)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
        if j == 0:
            ax.set_ylabel(f"share within {th:g}° [%]")
axs[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("Best layout of each family: old recon vs zoo recon", fontsize=13, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(Z, "compare_best.png"), dpi=120, facecolor="white", bbox_inches="tight")
json.dump({k: {t: dict(frac=v[0], layout=v[1]) for t, v in c.items()} for k, c in summary.items()},
          open(os.path.join(Z, "compare_summary.json"), "w"), indent=1)

# 3. detector count
dc = {t: json.load(open(os.path.join(Z, t, "detcount", "detector_count_grid.json"))) for t in ("old", "zoo")}
fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
for ax, (key, th) in zip(axs, (("good1_angle", 1), ("good25_angle", 2.5), ("good5_angle", 5))):
    for tag in ("old", "zoo"):
        ns = sorted(int(n) for n in dc[tag])
        ax.plot(ns, [100 * dc[tag][str(n)][key] for n in ns], "o-", color=REC_COL[tag], lw=2, label=f"{tag} recon")
    ax.set(xscale="log", xlabel="detectors in the array (grid)", ylabel=f"share within {th:g}° [%]",
           title=f"Grid with N detectors · within {th:g}°")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.grid(alpha=0.25, lw=0.5)
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(Z, "compare_detcount.png"), dpi=120, facecolor="white", bbox_inches="tight")

# console summary
for th in THETAS:
    d = np.array([zoo[k][th] - old[k][th] for k in keys])
    print(f"within {th:g}°: zoo - old over {len(keys)} layouts: median {100 * np.median(d):+.1f} pts, "
          f"better on {np.mean(d > 0):.0%}")
    for f in fams:
        df = [zoo[k][th] - old[k][th] for k in keys if old[k]["fam"] == f]
        if df:
            print(f"   {f:9s} n={len(df):2d} median {100 * np.median(df):+.1f} pts")
print("[write] compare_scatter.png compare_best.png compare_detcount.png compare_summary.json")

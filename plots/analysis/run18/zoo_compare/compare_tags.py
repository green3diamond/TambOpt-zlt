"""Compare two scored recons (folders <tag>/<set>/scores_pad_<theta>deg.json, <tag>/detcount/...):
per-family median change, best layout per family and N, grid detector count, and a scatter
compare_scatter_<B>_vs_<A>.png.   python compare_tags.py <A> <B>   (e.g. zoo reltime_lbfgs25)"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
A, B = sys.argv[1], sys.argv[2]
SETS = ("core_test_ndet", "clusters", "angleonly", "disks_n10")
TH = (1, 2.5, 5)
FAM_COL = {"grid": "#2a78d6", "disk": "#1baf7a", "cluster": "#eb6834", "core": "#c23b6b", "hybrid": "#8a5cd1",
           "optimized": "#7a7a7a"}


def fam(n):
    return ("optimized" if n.startswith("kern_") else "grid" if n.startswith("grid") else "disk" if n.startswith("disk")
            else "cluster" if n.startswith("clu") else "core" if re.match(r"core_r\d+", n) else "hybrid")


def ndet(n, r):
    m = re.search(r"_n(\d+)$", n)
    return int(m.group(1)) if m else int(r.get("n_det", 100))


def load(tag):
    out = {}
    for s in SETS:
        for th in TH:
            f = f"{Z}/{tag}/{s}/scores_pad_{th:g}deg.json"
            if not os.path.exists(f):
                continue
            for n, r in json.load(open(f)).items():
                d = out.setdefault((s, n), dict(name=n, fam=fam(n), n=ndet(n, r)))
                d[th] = 100 * r["frac_good_t"]; d["ang"] = r["ang_med"]
    return out


a, b = load(A), load(B)
keys = sorted(set(a) & set(b))
print(f"{len(keys)} layouts scored by both  ({A} -> {B})")
for th in TH:
    d = np.array([b[k][th] - a[k][th] for k in keys])
    print(f"within {th:g}°: median {np.median(d):+.1f} pts, {B} better on {np.mean(d > 0):.0%} " +
          " ".join(f"{f}:{np.median([b[k][th] - a[k][th] for k in keys if a[k]['fam'] == f]):+.1f}"
                   for f in FAM_COL if any(a[k]["fam"] == f for k in keys)))
dm = np.array([b[k]["ang"] - a[k]["ang"] for k in keys])
print(f"median angle error: median change {np.median(dm):+.2f}°")
for nm in (("angleonly", "grid"), ("angleonly", "disk_r3000"), ("angleonly", "disk_r4500"), ("core_test_ndet", "grid_n20"),
           ("core_test_ndet", "grid_n50"), ("clusters", "clu20x5_r500_n100"), ("angleonly", "core_r500")):
    if nm in a and nm in b:
        print(f"  {nm[1]:22s} " + "  ".join(f"{th:g}°: {a[nm][th]:4.1f} -> {b[nm][th]:4.1f}" for th in TH)
              + f"  median {a[nm]['ang']:.2f}° -> {b[nm]['ang']:.2f}°")
for tag in (A, B):
    f = f"{Z}/{tag}/detcount/detector_count_grid.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  grid N ({tag}): " + "  ".join(f"{k}: {100 * v['good1_angle']:.1f}/{100 * v['good5_angle']:.1f}" for k, v in d.items())
              + "   (within 1° / 5°)")
fig, axs = plt.subplots(1, 3, figsize=(19, 6.6))
for ax, th in zip(axs, TH):
    for k in keys:
        ax.scatter(a[k][th], b[k][th], s=30, color=FAM_COL[a[k]["fam"]], lw=0, alpha=0.85)
    lim = 1.06 * max(max(a[k][th] for k in keys), max(b[k][th] for k in keys))
    ax.plot([0, lim], [0, lim], "--", color="#52514e", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=f"{A}: share within θ [%]", ylabel=f"{B}: share within θ [%]",
           title=f"within {th:g}°"); ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
axs[2].legend(handles=[plt.Line2D([], [], color=c, marker="o", ls="", label=f) for f, c in FAM_COL.items()],
              frameon=False, fontsize=9, loc="lower right")
fig.suptitle(f"Every test layout: {A} (x) vs {B} (y), same 1,536 held-out showers (above the line = {B} better)")
fig.tight_layout(); fig.savefig(f"{Z}/compare_scatter_{B}_vs_{A}.png", dpi=120, facecolor="white", bbox_inches="tight")
print(f"[write] compare_scatter_{B}_vs_{A}.png")

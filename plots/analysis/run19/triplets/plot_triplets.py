"""Run 19 triplet test plots, from triplet_scores.json (+ the K x M clusters in ../noise/noise_scores.json):
    triplets_vs_r.png    share within 1 deg (all showers) vs cluster radius, triplets vs 2/4/5-detector clusters,
                         grid and best disk as horizontal lines; rows = N, columns = no noise / 128 ns / 1 us / 10 us,
                         each with (solid) and without (dotted) the >= 3-detector trigger
    triplets_trig.png    triggered share vs cluster radius at W = 0 and 1 us
    triplets_designs.png the triplet layouts (full plane at r = 25 / 100 / 500 m, plus one cluster zoomed for every r)
    compare_designs.png  what they are compared with: grid, best disk (no noise, within 1 deg), 2 / 4 / 5-detector clusters at r = 100 m
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

R = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics"
T = os.path.join(R, "triplets")
S = json.load(open(os.path.join(T, "triplet_scores.json")))["scores"]
S0 = json.load(open(os.path.join(R, "noise", "noise_scores.json")))["scores"]
WS = (("0", "no noise"), ("128", "128 ns"), ("1000", "1 µs"), ("10000", "10 µs"))
ROWS = ((99, 100), (48, 50), (21, 20), (9, 10))                               # (triplet N, main-test N)
COL = dict(tri="#c0392b", x2="#2e86c1", x4="#27ae60", x5="#8e44ad", grid="#555", disk="#e67e22")


def fam(scores, pat):
    out = {}
    for k, v in scores.items():
        mm = re.fullmatch(pat, k)
        if mm:
            out.setdefault(mm.group(1), []).append((int(mm.group(2)), v))
    return {f: sorted(x, key=lambda t: t[0]) for f, x in out.items()}


def best_disk(scores, n, w, key):
    c = [(v[w][key], k) for k, v in scores.items() if re.fullmatch(rf"disk_r\d+_n{n}", k)]
    return max(c) if c else (float("nan"), "")


def panel(ax, n3, n0, w, key, ylab):
    tri = fam(S, rf"(tri\d+)_r(\d+)_n{n3}")
    for f, xs in tri.items():
        ax.plot([r for r, _ in xs], [100 * v[w][key] for _, v in xs], "o-", c=COL["tri"], label=f"{f}×3 (N={n3})")
    for f, xs in fam(S0, rf"(clu\d+x\d)_r(\d+)_n{n0}").items():
        M = "x" + f.split("x")[1]
        ax.plot([r for r, _ in xs], [100 * v[w][key] for _, v in xs], "s--", c=COL.get(M, "k"), ms=4, lw=1, label=f"{f[3:]} (N={n0})")
    g = S[f"grid_n{n3}"][w][key]
    ax.axhline(100 * g, c=COL["grid"], lw=1.2, label=f"grid N={n3}")
    d, dk = best_disk(S, n3, w, key)
    ax.axhline(100 * d, c=COL["disk"], lw=1.2, ls="-.", label=f"best disk {dk.split('_')[1]}")
    ax.set_xscale("log"); ax.set_ylabel(ylab); ax.grid(alpha=.3)


for key, fn, ylab in (("all_1", "triplets_vs_r.png", "within 1° [%] (trigger)"),
                      ("notrig_1", "triplets_vs_r_notrig.png", "within 1° [%] (no trigger)"),
                      ("triggered", "triplets_trig.png", "triggered [%]")):
    fig, axs = plt.subplots(len(ROWS), len(WS), figsize=(17, 13), sharex=True)
    for i, (n3, n0) in enumerate(ROWS):
        for j, (w, wl) in enumerate(WS):
            panel(axs[i, j], n3, n0, w, key, ylab if j == 0 else "")
            if i == 0: axs[i, j].set_title(wl)
            if i == len(ROWS) - 1: axs[i, j].set_xlabel("cluster radius r [m]")
            if j == len(WS) - 1: axs[i, j].legend(fontsize=7, loc="best")
    fig.suptitle(f"Run 19: triplets (3 detectors on a triangle of circumradius r) vs other layouts — {ylab}")
    fig.tight_layout(); fig.savefig(os.path.join(T, fn), dpi=110); plt.close(fig)
    print("[write]", os.path.join(T, fn))


def xy(path):
    L = torch.load(path, map_location="cpu")
    return np.asarray(L["x"], float) / 1000, np.asarray(L["y"], float) / 1000


def paths(f):
    return dict(l.strip().split("=", 1) for l in open(f) if l.strip())


PT, P0 = paths(os.path.join(T, "layouts.txt")), paths(os.path.join(R, "tests", "layouts.txt"))
ge, gn = xy(PT["grid_n99"])
LIM = (ge.min() - .4, ge.max() + .4), (gn.min() - .4, gn.max() + .4)


def plane(ax, path, title, c="k", s=6):
    e, n = xy(path)
    ax.scatter(e, n, s=s, c=c, lw=0)
    ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10); ax.tick_params(labelsize=7)


TRI_R = sorted({int(k.split("_r")[1].split("_")[0]) for k in PT if k.startswith("tri")})
cm = plt.get_cmap("viridis")
fig, axs = plt.subplots(len(ROWS), 4, figsize=(17, 17))
for i, (n3, _) in enumerate(ROWS):
    K = n3 // 3
    for j, r in enumerate((25, 100, 500)):
        plane(axs[i, j], PT[f"tri{K}_r{r}_n{n3}"], f"tri{K}×3, r = {r} m (N = {n3})", s=8)
    ax = axs[i, 3]
    for q, r in enumerate(TRI_R):                                              # the cluster nearest the plane centre
        e, n = xy(PT[f"tri{K}_r{r}_n{n3}"])
        ce, cn = e.reshape(K, 3).mean(1), n.reshape(K, 3).mean(1)
        k = np.argmin(np.hypot(ce - ge.mean(), cn - gn.mean()))
        de, dn = (e.reshape(K, 3)[k] - ce[k]) * 1000, (n.reshape(K, 3)[k] - cn[k]) * 1000
        col = cm(q / max(1, len(TRI_R) - 1))
        ax.fill(de, dn, fc="none", ec=col, lw=1.2); ax.scatter(de, dn, s=14, color=col, label=f"r = {r} m (side {r*3**.5:.0f} m)")
    ax.set_aspect("equal"); ax.set_title("one cluster, zoomed [m]", fontsize=10); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
for ax in axs[-1]: ax.set_xlabel("East [km]")
for ax in axs[:, 0]: ax.set_ylabel("North [km]")
fig.suptitle("Run 19 triplet designs: K cluster centres on an even grid, 3 detectors on an equilateral triangle of circumradius r, random rotation per cluster")
fig.tight_layout(); fig.savefig(os.path.join(T, "triplets_designs.png"), dpi=110); plt.close(fig)
print("[write]", os.path.join(T, "triplets_designs.png"))

fig, axs = plt.subplots(len(ROWS), 5, figsize=(20, 17))
for i, (n3, n0) in enumerate(ROWS):
    _, dk = best_disk(S, n3, "0", "notrig_1")
    plane(axs[i, 0], PT[f"grid_n{n3}"], f"grid (N = {n3})")
    plane(axs[i, 1], PT[dk], f"best disk {dk.split('_')[1][1:]} m (N = {n3})", c=COL["disk"])
    clu = sorted(k for k in P0 if re.fullmatch(rf"clu\d+x\d_r100_n{n0}", k))
    for j in range(3):
        if j < len(clu):
            M = "x" + clu[j].split("x")[1].split("_")[0]
            plane(axs[i, 2 + j], P0[clu[j]], f"{clu[j][3:].split('_')[0]} clusters, r = 100 m (N = {n0})", c=COL.get(M, "k"))
        else:
            axs[i, 2 + j].axis("off")
for ax in axs[-1]: ax.set_xlabel("East [km]")
for ax in axs[:, 0]: ax.set_ylabel("North [km]")
fig.suptitle("Compared with: the grid and best disk at the triplet N, and the 2 / 4 / 5-detector clusters of the main test (shown at r = 100 m; r = 50-1000 m scored)")
fig.tight_layout(); fig.savefig(os.path.join(T, "compare_designs.png"), dpi=110); plt.close(fig)
print("[write]", os.path.join(T, "compare_designs.png"))

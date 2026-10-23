"""Run 19 triplet test plots, from triplet_scores.json (+ the K x M clusters in ../noise/noise_scores.json):
    triplets_vs_r_notrig.png  share within 1 deg (all showers) vs cluster radius, triplets vs 2/4/5-detector clusters,
                         grid and best disk as horizontal lines; rows = N, columns = no noise / 128 ns / 1 us / 10 us
    avg_vs_r_notrig.png  averaged stations (score_avg.py) · local_vs_r_notrig.png  local 2-of-3 trigger (score_local.py)
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


for key, fn, ylab in (("notrig_1", "triplets_vs_r_notrig.png", "within 1° [%]"),):
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

# the same layouts split for slides: full plane only (no zoom column), and one zoomed cluster (N = 99, every r)
fig, axs = plt.subplots(len(ROWS), 3, figsize=(13, 17))
for i, (n3, _) in enumerate(ROWS):
    for j, r in enumerate((25, 100, 500)):
        plane(axs[i, j], PT[f"tri{n3 // 3}_r{r}_n{n3}"], f"tri{n3 // 3}×3, r = {r} m (N = {n3})", s=8)
for ax in axs[-1]: ax.set_xlabel("East [km]")
for ax in axs[:, 0]: ax.set_ylabel("North [km]")
fig.suptitle("Run 19 triplet designs: K cluster centres on an even grid, 3 detectors per triangle of circumradius r")
fig.tight_layout(); fig.savefig(os.path.join(T, "triplets_designs_planes.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
K = 33
for q, r in enumerate(TRI_R):
    e, n = xy(PT[f"tri{K}_r{r}_n{3 * K}"])
    ce, cn = e.reshape(K, 3).mean(1), n.reshape(K, 3).mean(1)
    k = np.argmin(np.hypot(ce - ge.mean(), cn - gn.mean()))
    de, dn = (e.reshape(K, 3)[k] - ce[k]) * 1000, (n.reshape(K, 3)[k] - cn[k]) * 1000
    col = cm(q / max(1, len(TRI_R) - 1))
    ax.fill(de, dn, fc="none", ec=col, lw=1.6); ax.scatter(de, dn, s=30, color=col, label=f"r = {r} m (side {r*3**.5:.0f} m)")
ax.set_aspect("equal"); ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
ax.set_title("One triplet cluster (the one nearest the plane centre, N = 99), every r", fontsize=11)
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(T, "triplets_cluster_zoom.png"), dpi=130); plt.close(fig)
print("[write]", os.path.join(T, "triplets_designs_planes.png"), os.path.join(T, "triplets_cluster_zoom.png"))

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

# Averaged stations (score_avg.py): separate detectors vs one averaged station per triangle vs one detector per centre.
AV = os.path.join(T, "triplet_avg_scores.json")
if os.path.exists(AV):
    A = json.load(open(AV))["scores"]
    for key, fn, ylab in (("notrig_1", "avg_vs_r_notrig.png", "within 1° [%]"),):
        fig, axs = plt.subplots(len(ROWS), len(WS), figsize=(17, 13), sharex=True)
        for i, (n3, _) in enumerate(ROWS):
            K = n3 // 3
            ks = sorted((k for k in A if k.startswith(f"tri{K}_")), key=lambda k: int(k.split("_r")[1].split("_")[0]))
            rs = [int(k.split("_r")[1].split("_")[0]) for k in ks]
            for j, (w, wl) in enumerate(WS):
                ax = axs[i, j]
                ax.plot(rs, [100 * S[k][w][key] for k in ks], "o-", c=COL["tri"], label=f"{K}×3 separate detectors (N = {n3})")
                ax.plot(rs, [100 * A[k]["avg"][w][key] for k in ks], "D-", c="#16a085", label=f"{K} averaged stations")
                if "avg_local" in A[ks[0]]:
                    ax.plot(rs, [100 * A[k]["avg_local"][w][key] for k in ks], "^--", c="#2a78d6", label=f"{K} averaged, 2-of-3 (> 0.001)")
                ax.plot(rs, [100 * A[k]["centres"][w][key] for k in ks], "x:", c="#7f8c8d", label=f"{K} single detectors at the centres")
                ax.axhline(100 * S[f"grid_n{n3}"][w][key], c=COL["grid"], lw=1.2, label=f"grid N = {n3}")
                ax.set_xscale("log"); ax.grid(alpha=.3)
                if j == 0: ax.set_ylabel(ylab)
                if i == 0: ax.set_title(wl)
                if i == len(ROWS) - 1: ax.set_xlabel("triangle circumradius r [m]")
                if j == len(WS) - 1: ax.legend(fontsize=7)
        fig.suptitle(f"Run 19 triplets read out as averaged stations (E and T averaged per triangle) — {ylab}")
        fig.tight_layout(); fig.savefig(os.path.join(T, fn), dpi=110); plt.close(fig)
        print("[write]", os.path.join(T, fn))

# Local 2-of-3 trigger per triangle (score_local.py): none / hit E' > 0.001 / > 1 / >= 1, same noise draws.
LO = os.path.join(T, "triplet_local_scores.json")
if os.path.exists(LO):
    Lc = json.load(open(LO))["scores"]
    RD = (("none", "no local trigger", "o-", COL["tri"]), ("e3", "2 of 3 hit, hit = E' > 0.001", "^-", "#2a78d6"))
    for key, fn, ylab in (("notrig_1", "local_vs_r_notrig.png", "within 1° [%]"),):
        fig, axs = plt.subplots(len(ROWS), len(WS), figsize=(17, 13), sharex=True)
        for i, (n3, _) in enumerate(ROWS):
            K = n3 // 3
            ks = sorted((k for k in Lc if k.startswith(f"tri{K}_")), key=lambda k: int(k.split("_r")[1].split("_")[0]))
            rs = [int(k.split("_r")[1].split("_")[0]) for k in ks]
            for j, (w, wl) in enumerate(WS):
                ax = axs[i, j]
                for rd, lab, mk, col in RD:
                    if rd not in Lc[ks[0]]:
                        continue
                    ax.plot(rs, [100 * Lc[k][rd][w][key] for k in ks], mk, c=col, ms=4, label=lab)
                ax.axhline(100 * S[f"grid_n{n3}"][w][key], c=COL["grid"], lw=1.2, label=f"grid N = {n3}")
                ax.set_xscale("log"); ax.grid(alpha=.3)
                if j == 0: ax.set_ylabel(f"N = {n3}: " + ylab, fontsize=8)
                if i == 0: ax.set_title(wl)
                if i == len(ROWS) - 1: ax.set_xlabel("triangle circumradius r [m]")
                if j == len(WS) - 1: ax.legend(fontsize=7)
        fig.suptitle(f"Run 19 triplets with a local 2-of-3 trigger per triangle (otherwise all 3 zeroed) — {ylab}")
        fig.tight_layout(); fig.savefig(os.path.join(T, fn), dpi=110); plt.close(fig)
        print("[write]", os.path.join(T, fn))

# Clean vs noise-fine-tuned recon on every triplet readout (run_ft.sh): ft_vs_clean.png.
FTF = [os.path.join(T, f) for f in ("triplet_scores_ft.json", "triplet_avg_scores_ft.json", "triplet_local_scores_ft.json")]
if all(os.path.exists(f) for f in FTF) and os.path.exists(AV) and os.path.exists(LO):
    SF, AF, LF = (json.load(open(f))["scores"] for f in FTF)
    RO = (("separate", lambda D, k, w: D[0][k][w], "#c0392b"), ("averaged", lambda D, k, w: D[1][k]["avg"][w], "#16a085"),
          ("averaged 2-of-3", lambda D, k, w: D[1][k]["avg_local"][w], "#2a78d6"),
          ("separate 2-of-3", lambda D, k, w: D[2][k]["e3"][w], "#8e44ad"))
    fig, axs = plt.subplots(3, len(WS), figsize=(17, 11), sharex=True)
    for i, (n3, _) in enumerate(ROWS[:3]):
        K = n3 // 3
        ks = sorted((k for k in A if k.startswith(f"tri{K}_")), key=lambda k: int(k.split("_r")[1].split("_")[0]))
        rs = [int(k.split("_r")[1].split("_")[0]) for k in ks]
        for j, (w, wl) in enumerate(WS):
            ax = axs[i, j]
            for lab, get, col in RO:
                ax.plot(rs, [100 * get((S, A, Lc), k, w)["notrig_1"] for k in ks], "o--", c=col, ms=3, lw=1, alpha=0.6)
                ax.plot(rs, [100 * get((SF, AF, LF), k, w)["notrig_1"] for k in ks], "o-", c=col, ms=4, lw=2, label=lab)
            ax.axhline(100 * S[f"grid_n{n3}"][w]["notrig_1"], c=COL["grid"], lw=1, ls="--")
            ax.axhline(100 * SF[f"grid_n{n3}"][w]["notrig_1"], c=COL["grid"], lw=1.6, label=f"grid N = {n3}")
            ax.set_xscale("log"); ax.grid(alpha=.3)
            if j == 0: ax.set_ylabel(f"N = {n3}: within 1° [%]")
            if i == 0: ax.set_title(wl)
            if i == 2: ax.set_xlabel("triangle circumradius r [m]")
            if j == len(WS) - 1: ax.legend(fontsize=7)
    fig.suptitle("Run 19 triplets: clean recon (dashed) vs recon fine-tuned with background (solid), every readout")
    fig.tight_layout(); fig.savefig(os.path.join(T, "ft_vs_clean.png"), dpi=110); plt.close(fig)
    print("[write]", os.path.join(T, "ft_vs_clean.png"))

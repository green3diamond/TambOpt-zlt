"""Plots for the noise-aware Stage 4 runs (opt_noise_scores.json, opt_noise_errors.npz, the runs' optimize_log.json):
    opt_noise_layouts_kernel.png / _surrogate.png   detector layouts (finished runs + starts), titles: test shares
    opt_noise_radial.png      distribution of detector distance from the plane centre, per run vs its start
    opt_noise_cdf.png         angular-error CDFs on the test showers at 1 us and 10 us with the kNN gate
    opt_noise_utility.png     utility vs step (training batch, 200-step mean; scoring set)
"""
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

R = C.RUN_LOCATION
O = os.path.join(R, "diagnostics", "opt_noise")
D = json.load(open(os.path.join(O, "opt_noise_scores.json")))["scores"]
ERR = np.load(os.path.join(O, "opt_noise_errors.npz"))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3
centre = mesh.mean(0)
RUNS = [k for k in D if k.startswith("twotier_")]


def label(k):
    lam = "λ 0.3" if "_l0p3_" in k else "λ 0"
    init = "grid" if "_grid_" in k else "centre"
    w = "1 µs" if "_bg1us" in k else "10 µs"
    return f"{'surrogate' if k.endswith('_sur') else 'kernel'} · {init} · {lam} · {w}"


def win(k):
    return "1us_knn" if "_bg1us" in k else "10us_knn"


def xy(name):
    path = (os.path.join(R, f"test_v6_run_04_optimize_lbfgs_kernel_{name}", "layout_best.pt") if name.startswith("twotier_")
            else os.path.join(R, "diagnostics", "opt", "layouts", f"{name}.pt") if name.startswith("start_") else None)
    if path is None:
        for line in open(os.path.join(R, "diagnostics", "tests", "layouts.txt")):
            k, p = line.strip().split("=", 1)
            if k == name:
                path = p
    L = torch.load(path, map_location="cpu")
    return np.asarray(L["x"]) / 1e3, np.asarray(L["y"]) / 1e3


def panel(ax, name, title, col):
    e, n = xy(name)
    ax.scatter(*mesh.T, s=0.3, color="#e4e3dd", lw=0)
    ax.scatter(e, n, s=9, color=col, lw=0)
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_xlim(mesh[:, 0].min() - .3, mesh[:, 0].max() + .3); ax.set_ylim(mesh[:, 1].min() - .3, mesh[:, 1].max() + .3)
    ax.set_title(title, fontsize=9)


def acc(name, rd):
    r = D[name][rd]
    return f"{100*r['w1']:.1f} / {100*r['w2.5']:.1f} / {100*r['w5']:.1f}%"


for kind, sel in (("kernel", [k for k in RUNS if not k.endswith("_sur")]), ("surrogate", [k for k in RUNS if k.endswith("_sur")])):
    if not sel:
        continue
    names = ["start_grid", "start_center"] + sorted(sel, key=lambda k: ("_center" in k, "_l0p3" in k, "_bg10us" in k))
    nc = 5
    fig, axs = plt.subplots((len(names) + nc - 1) // nc, nc, figsize=(4.2 * nc, 4.6 * ((len(names) + nc - 1) // nc)), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, k in zip(axs.flat, names):
        ax.axis("on")
        if k.startswith("start_"):
            panel(ax, k, f"start: {k[6:]}\n1 µs gated {acc(k, '1us_knn')}\n10 µs gated {acc(k, '10us_knn')}", "#9b9a95")
        else:
            panel(ax, k, f"{label(k)}\n{win(k)[:-4].replace('us', ' µs')} gated: {acc(k, win(k))}\nno noise: {acc(k, 'clean')}",
                  "#2a78d6" if "_grid_" in k else "#c23b6b")
    fig.suptitle(f"Noise-aware optimisation ({kind} response): best layouts · test showers within 1 / 2.5 / 5° (kernel scoring)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(O, f"opt_noise_layouts_{kind}.png"), dpi=95, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
bins = np.linspace(0, 9, 37)
for ax, init in zip(axs, ("grid", "center")):
    e, n = xy(f"start_{init}")
    ax.hist(np.hypot(e - centre[0], n - centre[1]), bins, histtype="stepfilled", color="#d9d8d3", label=f"start ({init})")
    for k in [k for k in RUNS if f"_{init}_" in k]:
        e, n = xy(k)
        ax.hist(np.hypot(e - centre[0], n - centre[1]), bins, histtype="step", lw=1.6,
                ls="--" if k.endswith("_sur") else "-", label=label(k).replace(f" · {init.replace('center', 'centre')}", ""))
    ax.set(xlabel="distance from the plane centre [km]", title=f"{init} start"); ax.legend(fontsize=7.5, frameon=False)
axs[0].set_ylabel("detectors")
fig.suptitle("Where the detectors went: radial distribution of the best layouts (solid: kernel, dashed: surrogate)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(O, "opt_noise_radial.png"), dpi=100, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
x = np.logspace(-1, 2, 300)
for ax, (rd, wl) in zip(axs, (("1us_knn", "1 µs"), ("10us_knn", "10 µs"))):
    for ref, col in (("grid_n100", "#555"), ("disk_r6000_n100", "#e67e22"), ("start_grid", "#9b9a95"), ("start_center", "#c9c8c3")):
        a = ERR[f"{ref}|{rd}"]
        ax.plot(x, 100 * (a[None] < x[:, None]).mean(1), color=col, lw=1.4, label=ref.replace("_n100", "").replace("disk_r6000", "disk 6 km"))
    for k in [k for k in RUNS if win(k) == rd]:
        a = ERR[f"{k}|{rd}"]
        ax.plot(x, 100 * (a[None] < x[:, None]).mean(1), lw=1.8, ls="--" if k.endswith("_sur") else "-",
                label=label(k).replace(f" · {wl}", ""))
    for th in (1, 2.5, 5):
        ax.axvline(th, color="#ddd", lw=0.8, zorder=0)
    ax.set(xscale="log", xlabel="angular error [deg]", title=f"{wl} window, kNN 2-of-3 gate"); ax.grid(alpha=.2)
    ax.legend(fontsize=7, frameon=False)
axs[0].set_ylabel("showers within the error [%]")
fig.suptitle("Angular-error distributions on the 5,000 test showers (kernel response, clean recon)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(O, "opt_noise_cdf.png"), dpi=100, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, kind in zip(axs, ("kernel", "surrogate")):
    for k in [k for k in RUNS if k.endswith("_sur") == (kind == "surrogate")]:
        L = json.load(open(os.path.join(R, f"test_v6_run_04_optimize_lbfgs_kernel_{k}", "optimize_log.json")))
        u = np.array([r["U"] for r in L["log"]]); kk = 200
        line, = ax.plot(np.arange(kk, len(u) + 1), np.convolve(u, np.ones(kk) / kk, "valid"), lw=1, alpha=.5)
        se = [r["epoch"] for r in L["score_log"]]; su = [r["U_score"] for r in L["score_log"]]
        ax.plot(se, su, "o-", ms=2.5, lw=1.6, color=line.get_color(), label=label(k).replace(f"{kind} · ", ""))
    ax.set(xlabel="step", ylabel="utility U (its own response)", title=f"{kind} response"); ax.grid(alpha=.25); ax.legend(fontsize=7.5, frameon=False)
fig.suptitle("Utility vs step (light: training batch, 200-step mean; dots: scoring set). λ 0 and λ 0.3 have different scales", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(O, "opt_noise_utility.png"), dpi=100, facecolor="white"); plt.close(fig)
print("[write] opt_noise_layouts_*.png opt_noise_radial.png opt_noise_cdf.png opt_noise_utility.png")

"""Run 20b (Malata, 1 deg under 10 us background): the figures of the recent tests, for the run 20b PDF.
  r20b_lr_scan.png      Stage 4 lr scan from the grid (lr 1 / 3 / 10, run 20 settings, no background): scoring-set U
                        every 50 steps (slurm_logs/diag_lr<lr>.log; cancelled at step 150)
  r20b_training.png     the three run 20b runs (first 8 h job, cancelled): scoring-set U relative to the grid, the
                        training-batch U (100-step mean) and the adaptive step size (each run's opt_state.pt)
  r20b_grad_check.png   gradient signal vs noise (scripts/diagnostics/r20_grad_check.py): cosine between batch
                        gradients, |mean| / mean |g| against pure noise, share of detectors with SNR > 2, for run 20
                        (default recon, no noise) and run 20b (drawn-count recon, 10 us + kNN gate + drawn counts)
  r20b_layouts.png      grid start and each run 20b run's layout after the 8 h job, on the detector region
Writes <run 20>/diagnostics/r20b/.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_r20b_tests.py
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402

RL = C.RUN_LOCATION
OUT = os.path.join(RL, "diagnostics", "r20b")
os.makedirs(OUT, exist_ok=True)
S4 = os.path.join(RL, "test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_a2")
T = "_lr3_grid_count_b2048_bg10us_knn23_cnt_acc8_plat"
RUNS = [("1deg_poisson", "1° utility, Poisson trigger", "#2a6fb0", S4 + "_th1_gtrig" + T),
        ("1deg_step", "1° utility, step trigger", "#6b6b6b", S4 + "_step_th1_gtrig" + T),
        ("oldu", "Old U (no trigger term)", "#e07b39", S4 + "_E_notrig" + T)]
GD = os.path.join(RL, "diagnostics")


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── lr scan ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
for lr, col in (("1", "#7fb3e0"), ("3", "#2a6fb0"), ("10", "#b3471b")):
    txt = open(os.path.join(RL, "slurm_logs", f"diag_lr{lr}.log")).read()
    ep = [0] + [int(m) for m in re.findall(r"\[score (\d+)\] U_score", txt)]
    u = [float(re.search(r"init U=([+-][\d.]+)", txt).group(1))] + \
        [float(m) for m in re.findall(r"\] U_score=([+-][\d.]+)", txt)]
    ax.plot(ep, u, marker="o", color=col, lw=1.8, label=f"lr {lr}")
    print(f"lr {lr}: " + ", ".join(f"{e}: {v:.3f}" for e, v in zip(ep, u)))
ax.axhline(u[0], color="#999999", lw=0.8, ls=":")
ax.text(3, u[0] + 0.005, "grid", fontsize=8, color="#666666")
ax.set_xlabel("Adam step")
ax.set_ylabel("U on the scoring set (1,024 showers)")
ax.set_title("Run 20 Stage 4 lr scan from the grid (swgo_trigger, 2,048 showers / step, no background)", fontsize=9.5)
ax.legend(frameon=False)
style(ax)
fig.savefig(os.path.join(OUT, "r20b_lr_scan.png"), dpi=170, facecolor="white")
plt.close(fig)

# ── run 20b training ─────────────────────────────────────────────────────────
W = 100
fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
ST = {}
for key, name, col, d in RUNS:
    st = torch.load(os.path.join(d, "opt_state.pt"), map_location="cpu", weights_only=False)
    ST[key] = st
    sc = st["score_log"]
    e = np.array([s["epoch"] for s in sc]); u = np.array([s["U_score"] for s in sc])
    axs[0].plot(e, 100 * (u / u[0] - 1), color=col, lw=1.6, marker="o", ms=2.5, label=name)
    log = [r for r in st["log"] if not r.get("skipped")]
    ub = np.array([r["U"] for r in log]); eb = np.array([r["epoch"] for r in log])
    c = np.cumsum(np.r_[0.0, ub])
    axs[1].plot(eb, 100 * (ub / u[0] - 1), color=col, lw=0.3, alpha=0.2)
    axs[1].plot(eb[W - 1:], 100 * ((c[W:] - c[:-W]) / W / u[0] - 1), color=col, lw=1.6, label=name)
    lr = [s.get("lr", np.nan) for s in sc]
    axs[2].step(e, lr, where="post", color=col, lw=1.6, label=name)
    print(f"{name}: {st['epoch']} steps, grid U {u[0]:.3f}, best {st['best_u']:.3f}, last {u[-1]:.3f}, lr {lr[-1]}")
axs[0].set_title("Scoring set (fixed 1,024 showers, one background / count draw)", fontsize=10)
axs[0].set_ylabel("U change from the grid [%]")
axs[1].set_title(f"Training batches (8 × 2,048 fresh showers per step; {W}-step mean)", fontsize=10)
axs[1].set_ylabel("U change from the grid's scoring-set U [%]")
axs[2].set_title("Adaptive step size (halved when the scoring-set U stalls)", fontsize=10)
axs[2].set_ylabel("Adam step size [m]")
axs[2].set_yscale("log")
axs[2].axhline(0.03, color="#999999", lw=0.8, ls=":")
axs[2].text(5, 0.033, "floor lr / 100", fontsize=8, color="#666666")
for ax in axs:
    ax.axhline(0, color="#999999", lw=0.6) if ax is not axs[2] else None
    ax.set_xlabel("Adam step")
    style(ax)
axs[0].legend(fontsize=8.5, frameon=False, loc="lower left")
fig.suptitle("Run 20b, Malata: Stage 4 for 1° under 10 µs background + kNN gate, drawn-count recon, drawn counts "
             "(first 8 h job of each run)", fontsize=12)
fig.savefig(os.path.join(OUT, "r20b_training.png"), dpi=170, facecolor="white")
plt.close(fig)

# ── gradient check ───────────────────────────────────────────────────────────
g20 = json.load(open(os.path.join(GD, "grad_check.json")))["results"]
rows = [("Run 20 utility, no noise\n(default recon), grid", g20["grid start"], None),
        ("Run 20 utility, no noise\n(default recon), optimised", g20["opt Poisson grid"], None)]
for key, name, _, _ in RUNS + [("1deg_poisson_nonoise", "1° Poisson, no noise", None, None)]:
    f = os.path.join(GD, f"grad_check_20b_{key}.json")
    if not os.path.exists(f):
        print("[skip]", f)
        continue
    r = json.load(open(f))["results"]
    lab = name.replace(" (no trigger term)", "").replace(" utility,", ",")
    sfx = "" if key.endswith("nonoise") else "\n10 µs + kNN + drawn"
    rows.append((f"{lab}{sfx}, grid", r["grid start"], r["grid start"].get("draws")))
    if "current" in r:
        rows.append((f"{lab}{sfx}, after 8 h", r["current"], r["current"].get("draws")))
y = np.arange(len(rows))[::-1]
fig, axs = plt.subplots(1, 3, figsize=(19, 0.62 * len(rows) + 2), sharey=True, constrained_layout=True)
blue, grey = "#2a6fb0", "#9aa5b1"
for k, (lab, r, dr) in enumerate(rows):
    yy = y[k]
    axs[0].barh(yy + 0.18, r["cos_pairwise_mean"], 0.34, xerr=r["cos_pairwise_std"], color=blue,
                error_kw=dict(lw=0.8), label="different batches" if k == 0 else None)
    if dr:
        axs[0].barh(yy - 0.18, dr["cos_pairwise_mean"], 0.34, xerr=dr["cos_pairwise_std"], color=grey,
                    error_kw=dict(lw=0.8), label="same batch, new bg / count draws" if dr is rows[2][2] else None)
    axs[1].barh(yy + 0.18, r["mean_over_batch_norm"], 0.34, color=blue)
    if dr:
        axs[1].barh(yy - 0.18, dr["mean_over_batch_norm"], 0.34, color=grey)
    axs[2].barh(yy + 0.18, 100 * r["snr_share_above_2"], 0.34, color=blue)
    if dr:
        axs[2].barh(yy - 0.18, 100 * dr["snr_share_above_2"], 0.34, color=grey)
axs[0].axvline(0, color="#444444", lw=0.8)
axs[1].axvline(1 / np.sqrt(8), color="#b3471b", lw=1, ls="--")
axs[1].text(1 / np.sqrt(8) + 0.005, y[-1] - 0.45, "pure noise, 8 batches", fontsize=8, color="#b3471b")
axs[1].axvline(1, color="#444444", lw=0.6, ls=":")
axs[0].set_yticks(y, [r[0] for r in rows], fontsize=8.5)
axs[0].set_title("Cosine between gradients (0 = unrelated, 1 = same)", fontsize=10)
axs[1].set_title("|mean of 8 gradients| / mean |gradient|", fontsize=10)
axs[2].set_title("Detectors with gradient SNR > 2 [%]", fontsize=10)
axs[0].legend(fontsize=8.5, frameon=False, loc="lower right")
for ax in axs:
    style(ax)
fig.suptitle("Stage 4 gradient check: 8 batches of 2,048 held-in showers per layout", fontsize=12)
fig.savefig(os.path.join(OUT, "r20b_grad_check.png"), dpi=170, facecolor="white")
plt.close(fig)

# ── layouts ──────────────────────────────────────────────────────────────────
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)
e0, n0 = sample_initial_layout_ne(m, n_units=C.N_DETECTORS, scheme="grid")
e0, n0 = np.asarray(e0, float), np.asarray(n0, float)
fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
for ax, (key, name, col, _) in zip(axs, RUNS):
    xy = ST[key]["xy"]
    x, yv = xy["x"].float().numpy(), xy["y"].float().numpy()
    mv = np.hypot(x - e0, yv - n0)
    ax.add_collection(PolyCollection(tri[:, :, :2], array=tri[:, :, 2].mean(1), cmap="Greys", edgecolors="white",
                                     linewidths=0.3))
    ax.scatter(e0, n0, s=12, facecolors="none", edgecolors="#999999", lw=0.8, label="grid start")
    for a, b, c, dd in zip(e0, n0, x, yv):
        ax.plot([a, c], [b, dd], color=col, lw=0.6, alpha=0.7)
    ax.scatter(x, yv, s=12, color=col, lw=0, zorder=3, label=f"after {ST[key]['epoch']} steps")
    ax.set_xlim(tri[..., 0].min() - 60, tri[..., 0].max() + 60)
    ax.set_ylim(tri[..., 1].min() - 60, tri[..., 1].max() + 60)
    ax.set_aspect("equal")
    ax.set_title(f"{name}\nmove from the grid: median {np.median(mv):.0f} m, max {mv.max():.0f} m", fontsize=10)
    ax.set_xlabel("East [m]")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    style(ax)
    print(f"{name}: median move {np.median(mv):.1f} m, max {mv.max():.1f} m")
axs[0].set_ylabel("North [m]")
fig.suptitle("Run 20b layouts after the first 8 h job (scoring-set best: grid for the Poisson run, "
             "+1.8% and +0.1% over the grid for step and Old U, early steps)", fontsize=12)
fig.savefig(os.path.join(OUT, "r20b_layouts.png"), dpi=170, facecolor="white")
plt.close(fig)
print("[write]", OUT)

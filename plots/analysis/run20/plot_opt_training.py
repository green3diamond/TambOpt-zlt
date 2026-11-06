"""Run 20 (Malata): the Stage 4 training curves of the swgo_trigger runs (optimize_log.json), one colour per run.
Panels: the utility on each step's fresh batch (without the cable penalty; 100-step running mean, raw steps faint),
the same with the cable penalty subtracted (what Adam maximised), the utility on the fixed scoring set every 250 steps (what picks the best layout; cable runs: minus the penalty, as
optimised), and the 3-D cable length of the cable runs (logged every step). --set 20b: the run 20b runs (1 deg under
10 us background, 8 x 2,048 showers per step) with the adaptive step size in the last panel instead of the cable;
--set 20c: the run 20c runs (1 deg, default recon, expected counts, no background / 1 us + kNN gate). A run without
optimize_log.json (stopped at a time limit) is read from its opt_state.pt.
Writes <run 20>/diagnostics/opt_training[_20b|_20c].png.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_opt_training.py [--set 20b|20c]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402

PRE = os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_")
ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("20", "20b", "20c"), default="20")
SET = ap.parse_args().set
RUNS = [("a2_lr10_grid_b2048", "Opt. Poisson, grid", "#2a6fb0", "-"),
        ("a2_lr10_center_b2048", "Opt. Poisson, centre", "#2a6fb0", "--"),
        ("a20_lr10_grid_b2048", "Opt. Poisson 20 m², grid", "#7fb3e0", "-"),
        ("a2_step_lr10_grid_b2048", "Opt. step, grid", "#6b6b6b", "-"),
        ("a2_step_lr10_center_b2048", "Opt. step, centre", "#6b6b6b", "--"),
        ("a2_lr10_grid_b2048_cab0p05", "Opt. + cable λ 0.05, grid", "#f2a65a", "-"),
        ("a2_lr10_grid_b2048_cab0p1", "Opt. + cable λ 0.1, grid", "#e07b39", "-"),
        ("a2_lr10_grid_b2048_cab0p2", "Opt. + cable λ 0.2, grid", "#b3471b", "-"),
        ("a2_lr10_center_b2048_cab0p1", "Opt. + cable λ 0.1, centre", "#e07b39", "--")]
if SET == "20b":                          # folder patterns (the lr is part of the name)
    T = "_lr*_grid_count_b2048_bg10us_knn23_cnt_acc8_plat"
    RUNS = [("a2_th1_gtrig" + T, "1° under 10 µs bg, Poisson", "#2a6fb0", "-"),
            ("a2_step_th1_gtrig" + T, "1° under 10 µs bg, step", "#6b6b6b", "-"),
            ("a2_E_notrig" + T, "Old U under 10 µs bg", "#e07b39", "-")]
if SET == "20c":
    RUNS = [("a2_th1_lr*_grid_b2048_acc8_plat", "1°, no background", "#2a6fb0", "-"),
            ("a2_th1_gtrig_lr*_grid_b2048_bg1us_knn23_acc16_plat", "1°, 1 µs bg + kNN gate", "#e07b39", "-")]
W = 100


def smooth(v):
    c = np.cumsum(np.r_[0.0, v])
    out = (c[W:] - c[:-W]) / W
    return np.r_[np.full(W - 1, np.nan), out]


fig, axs = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)
for sfx, name, col, ls in RUNS:
    d = next(iter(sorted(glob.glob(PRE + sfx))), "")
    f = os.path.join(d, "optimize_log.json")
    if os.path.exists(f):
        j = json.load(open(f))
    elif os.path.exists(os.path.join(d, "opt_state.pt")):
        import torch
        j = torch.load(os.path.join(d, "opt_state.pt"), map_location="cpu", weights_only=False)
    else:
        print("[skip]", PRE + sfx)
        continue
    log = [r for r in j["log"] if not r.get("skipped")]
    ep = np.array([r["epoch"] for r in log])
    u = np.array([r.get("U_util", r["U"]) for r in log])
    axs[0].plot(ep, u, color=col, lw=0.3, alpha=0.15)
    axs[0].plot(ep, smooth(u), color=col, ls=ls, lw=1.6, label=name)
    ut = np.array([r["U"] for r in log])                         # as optimised (cable runs: minus the penalty)
    axs[1].plot(ep, ut, color=col, lw=0.3, alpha=0.15)
    axs[1].plot(ep, smooth(ut), color=col, ls=ls, lw=1.6, label=name)
    sc = j["score_log"]
    axs[2].plot([s["epoch"] for s in sc], [s["U_score"] for s in sc], color=col, ls=ls, lw=1.6, marker="o", ms=3,
                label=name)
    if SET != "20":
        axs[3].step([s["epoch"] for s in sc], [s.get("lr", np.nan) for s in sc], where="post", color=col, ls=ls,
                    lw=1.6, label=name)
    elif "cable_km" in log[0]:
        axs[3].plot(ep, [r["cable_km"] for r in log], color=col, ls=ls, lw=1.6, label=name)
    print(f"{name}: {len(log)} steps, U_util last-100 mean {np.nanmean(u[-W:]):.3f}, best scoring U {j['best_u']:.3f}")

axs[0].set_title(f"Training set: utility on each step's batch, no cable penalty ({W}-step mean)", fontsize=10)
axs[0].set_ylabel("U (30 × mean reward)")
axs[1].set_title(f"Training set: utility as optimised, cable runs minus the penalty ({W}-step mean)", fontsize=10)
axs[1].set_ylabel("U on the training batches")
axs[2].set_title("Scoring set (fixed 1,024 showers): utility as optimised", fontsize=10)
axs[2].set_ylabel("U on the scoring set")
if SET == "20":
    for ax in axs[1:3]:
        ax.set_ylim(15, 29)                # the centre-start cable run starts far below (detectors closer than d_min)
    axs[2].text(60, 15.4, "centre start + cable: step 0 off scale (spacing penalty)", fontsize=8, color="#666666")
    axs[3].set_title("3-D cable length (cable runs)", fontsize=10)
    axs[3].set_ylabel("Cable [km]")
    axs[3].axhline(9.30, color="#999999", lw=0.8, ls=":")
    axs[3].text(50, 9.35, "grid 9.30 km", fontsize=8, color="#666666")
else:
    axs[3].set_title("Adaptive step size (halved when the scoring-set U stalls)", fontsize=10)
    axs[3].set_ylabel("Adam step size [m]")
    axs[3].set_yscale("log")
for ax in axs:
    ax.set_xlabel("Adam step")
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle({"20": "Run 20, Malata: Stage 4 training (swgo_trigger, 2,048 showers per step)",
              "20b": "Run 20b, Malata: Stage 4 training under 10 µs background + kNN gate (8 × 2,048 showers per step, "
                     "drawn-count recon, drawn counts)",
              "20c": "Run 20c, Malata: Stage 4 for 1° with the default recon on expected counts (no background: "
                     "8 × 2,048 showers per step; 1 µs + kNN gate: 16 × 2,048)"}[SET], fontsize=12)
out = os.path.join(C.RUN_LOCATION, "diagnostics", "opt_training" + ("" if SET == "20" else "_" + SET) + ".png")
fig.savefig(out, dpi=170, facecolor="white")
print("[write]", out)

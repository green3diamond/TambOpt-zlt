"""Utility vs epoch for the kernel-in-the-loop runs (04_optimize_lbfgs_kernel.py --pool main).
Writes utility_vs_epoch.png into each run folder: per-step batch U (raw + rolling mean), the
scoring-set U every 250 steps with the kept best, and the utility components."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
C_BATCH, C_SCORE, C_BEST = "#2a78d6", "#eb6834", "#1f2328"
COMP = {"u_theta": ("#2a78d6", "zenith (100·U_θ)"), "u_phi": ("#1baf7a", "azimuth (100·U_φ)"),
        "u_e": ("#eda100", "energy (250·U_E)")}


def roll(x, w=200):
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="valid")
    return np.arange(w, len(x) + 1), y


for scheme in ("grid", "center"):
    d = os.path.join(R, f"test_v6_run_04_optimize_lbfgs_kernel_recon_utility_lr10_{scheme}")
    j = json.load(open(os.path.join(d, "optimize_log.json")))
    log, sl = j["log"], j["score_log"]
    ep = np.array([e["epoch"] for e in log])
    U = np.array([e["U"] for e in log])
    se = np.array([e["epoch"] for e in sl])
    su = np.array([e["U_score"] for e in sl])
    b = int(np.argmax(su))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    a.plot(ep, U, color=C_BATCH, alpha=0.18, lw=0.6, label="batch U (512 fresh events/step)")
    xr, yr = roll(U)
    a.plot(xr, yr, color=C_BATCH, lw=2, label="batch U, 200-step rolling mean")
    a.plot(se, su, "o-", color=C_SCORE, ms=4, lw=1.6, label="scoring-set U (1,024 separate events)")
    a.plot(se[b], su[b], "*", color=C_BEST, ms=16, label=f"kept best: {su[b]:.2f} at step {se[b]}")
    lo = np.percentile(np.concatenate([U[len(U) // 20:], su[1:]]), 0.5)
    hi = np.percentile(np.concatenate([U, su]), 99.8)
    if scheme == "center":
        a.set_ylim(min(su[0], lo) - 1, hi + 0.5)
    else:
        a.set_ylim(lo - 0.5, hi + 0.5)
    a.set_xlabel("optimization step", color=GREY)
    a.set_ylabel("utility U", color=GREY)
    a.set_title(f"Utility vs step — kernel in the loop, {scheme} start (run 15, Adam lr = 10)", fontsize=11,
                color=INK)
    a.legend(fontsize=8.5, frameon=False, loc="lower right")

    a = ax[1]
    for k, (c, lab) in COMP.items():
        v = np.array([e[k] for e in log])
        xr, yr = roll(v)
        a.plot(xr, yr, color=c, lw=2, label=lab)
    a.set_xlabel("optimization step", color=GREY)
    a.set_ylabel("contribution to U (200-step rolling mean)", color=GREY)
    a.set_title("Utility components (batch)", fontsize=11, color=INK)
    a.legend(fontsize=8.5, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(d, "utility_vs_epoch.png")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"[write] {out}")

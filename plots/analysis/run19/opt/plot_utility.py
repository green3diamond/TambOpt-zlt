"""Utility vs step for the run 19 optimisations: per detector count, one panel per utility; training-batch U
(512 fresh showers per step, 200-step running mean) and the scoring-set U (1,024 fixed showers, every 250 steps),
grid start and centre start. Writes opt_utility_n<N>.png. Utilities have different scales: compare runs within a panel."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

O = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(O))
UT = (("current_lr10", "current"), ("twotier_l0_lr10", "two-tier λ 0"), ("twotier_l0p3_lr10", "two-tier λ 0.3"),
      ("threetier_090301_lr10", "three-tier"))
SC = {"grid": "#2a78d6", "center": "#c23b6b"}
for N in (100, 50, 20):
    SN = "" if N == 100 else f"_n{N}"
    runs = []
    for suf, lab in UT:
        got = {}
        for scheme in ("grid", "center"):
            f = f"{R}/test_v6_run_04_optimize_lbfgs_kernel_{suf}_{scheme}{SN}/optimize_log.json"
            if os.path.exists(f):
                got[scheme] = json.load(open(f))
        if got:
            runs.append((lab, got))
    if not runs:
        continue
    fig, axs = plt.subplots(1, len(runs), figsize=(6.4 * len(runs), 5.4), squeeze=False)
    for ax, (lab, got) in zip(axs[0], runs):
        for scheme, L in got.items():
            ep = np.array([r["epoch"] for r in L["log"]]); u = np.array([r["U"] for r in L["log"]])
            k = 200
            sm = np.convolve(u, np.ones(k) / k, mode="valid")
            ax.plot(ep[k - 1:], sm, color=SC[scheme], lw=1.2, alpha=0.55, label=f"{scheme} start · training batch (200-step mean)")
            se = [r["epoch"] for r in L["score_log"]]; su = [r["U_score"] for r in L["score_log"]]
            ax.plot(se, su, "o-", color=SC[scheme], lw=1.8, ms=3, label=f"{scheme} start · scoring set (best {max(su):.2f})")
        ax.set(xlabel="optimisation step (Adam)", ylabel="utility U", title=f"{lab} · {N} detectors")
        ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Run 19 optimisations, {N} detectors: utility per step (U scales differ between utilities)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{O}/opt_utility{SN or '_n100'}.png", dpi=100, facecolor="white"); plt.close(fig)
    print("[write]", f"{O}/opt_utility{SN or '_n100'}.png")

# 2,048 showers per step with lr 10 -> 1 (5,000 steps) vs 512 per step at lr 10 (20,000 steps), two-tier lambda 0.3, 100 detectors.
# x = showers processed (steps x batch): both runs end at the same 10.24 M, so equal compute.
CMP = (("twotier_l0p3_lr10", "", 512, "512 / step, lr 10, 20,000 steps", "#9b9a95"),
       ("twotier_l0p3_lr10", "_b2048_lrd1", 2048, "2,048 / step, lr 10 → 1, 5,000 steps", "#b0283a"))
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
for ax, scheme in zip(axs, ("grid", "center")):
    for suf, tail, b, lab, col in CMP:
        f = f"{R}/test_v6_run_04_optimize_lbfgs_kernel_{suf}_{scheme}{tail}/optimize_log.json"
        if not os.path.exists(f):
            continue
        L = json.load(open(f))
        ep = np.array([r["epoch"] for r in L["log"]]); u = np.array([r["U"] for r in L["log"]])
        k = max(1, 200 * 512 // b)                                            # same number of showers per mean
        sm = np.convolve(u, np.ones(k) / k, mode="valid")
        ax.plot(ep[k - 1:] * b / 1e6, sm, color=col, lw=1.1, alpha=0.45)
        se = np.array([r["epoch"] for r in L["score_log"]]); su = [r["U_score"] for r in L["score_log"]]
        ax.plot(se * b / 1e6, su, "o-", color=col, lw=1.8, ms=3, label=f"{lab} · best {max(su):.2f}")
    ax.set(xlabel="showers processed [M] (steps × showers per step)", title=f"two-tier λ 0.3 · 100 detectors · {scheme} start")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
axs[0].set_ylabel("utility U (light: training batch; dots: scoring set)")
fig.suptitle("Run 19: bigger batch + decaying step size vs the original setting, at equal compute", fontsize=13)
fig.tight_layout(); fig.savefig(f"{O}/opt_utility_b2048.png", dpi=100, facecolor="white"); plt.close(fig)
print("[write]", f"{O}/opt_utility_b2048.png")

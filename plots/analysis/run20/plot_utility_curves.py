"""Run 20: utility against SGD step for every 1,000-step optimisation (each run's opt_state.pt).
  utility_curves_5deg.png     5 deg utility: energy bands (top) and construction penalties (bottom); surrogate, surrogate
                              optimised at 1 us, kernel. Lines: training-batch utility without the penalty (100-step mean);
                              markers: the scoring set (utility minus penalty, as optimised; every 25 steps)
  utility_curves_other.png    trigger only (energy bands, penalties) and penalty only (penalty value, sign flipped)
  utility_curves_lhs.png      the 12 Latin-hypercube configurations, one line per seed (scoring set)
SKIP = run names to leave out (unreadable folders). Writes <run 20>/diagnostics/r20b/.

    source slurm/run20_guard.sh; SKIP=trg_pads python plots/analysis/run20/plot_utility_curves.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402

PRE = os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_")
OUT = os.path.join(C.RUN_LOCATION, "diagnostics", "r20b")
SKIP = set(filter(None, os.environ.get("SKIP", "").split(",")))
BANDS = [("e500_567", "E 5.00–5.67"), ("e567_633", "E 5.67–6.33"), ("e633_700", "E 6.33–7.00"), ("eall", "all E")]
PENS = [("slope", "slope limit"), ("access", "access"), ("pads", "pads"), ("contour", "contour rows")]
W = 100


def load(name):
    f = PRE + name + "/opt_state.pt"
    if name in SKIP or not os.path.exists(f):
        return None
    st = torch.load(f, map_location="cpu", weights_only=False)
    log = [r for r in st["log"] if not r.get("skipped")]
    ep = np.array([r["epoch"] for r in log])
    u = np.array([r.get("U_util", r["U"]) for r in log])
    pen = np.array([r.get("penalty", 0.0) for r in log])
    sc = st["score_log"]
    return dict(ep=ep, u=u, pen=pen, se=np.array([s["epoch"] for s in sc]), su=np.array([s["U_score"] for s in sc]))


def smooth(v):
    if len(v) < W:
        return v
    c = np.cumsum(np.r_[0.0, v])
    return np.r_[np.full(W - 1, np.nan), (c[W:] - c[:-W]) / W]


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("SGD step", fontsize=8)
    ax.tick_params(labelsize=7)


def draw(ax, d, col, label, batch=True):
    if d is None:
        ax.plot([], [], color=col, label=label + " (missing)")
        ax.text(0.5, 0.5, "file locked / missing", transform=ax.transAxes, ha="center", color="#888888")
        return
    if batch:
        ax.plot(d["ep"], smooth(d["u"]), color=col, lw=1.0, alpha=0.55)
    ax.plot(d["se"], d["su"], color=col, lw=1.6, marker="o", ms=2.5, label=label)


# ── 5 deg utility ────────────────────────────────────────────────────────────
SETS = [("sur5", "surrogate", "#2a6fb0"), ("sur5_1us", "surrogate, 1 µs", "#7fb3e0"), ("ker5", "kernel", "#e07b39")]
fig, axs = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
for r, group in enumerate((BANDS, PENS)):
    for c, (k, name) in enumerate(group):
        ax = axs[r, c]
        for tag, lab, col in SETS:
            draw(ax, load(f"{tag}_{k}"), col, lab)
        style(ax, ("" if r == 0 else "+ ") + name)
axs[0, 0].set_ylabel("U (30 × mean reward)")
axs[1, 0].set_ylabel("U (penalty runs: markers = U − penalty)")
axs[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("5° Poisson utility, SGD from the 100-detector grid: lines = training-batch utility without penalty "
             f"({W}-step mean), markers = scoring set as optimised (every 25 steps)", fontsize=12)
fig.savefig(os.path.join(OUT, "utility_curves_5deg.png"), dpi=150, facecolor="white")
plt.close(fig)

# ── trigger only, penalty only ───────────────────────────────────────────────
fig, axs = plt.subplots(3, 4, figsize=(20, 13), constrained_layout=True)
for c, (k, name) in enumerate(BANDS):
    draw(axs[0, c], load(f"trg_{k}"), "#e07b39", "trigger only")
    style(axs[0, c], f"trigger only: {name}")
for c, (k, name) in enumerate(PENS):
    draw(axs[1, c], load(f"trg_{k}"), "#e07b39", "trigger only")
    style(axs[1, c], f"trigger only + {name}")
    d = load(f"pen_{k}")
    ax = axs[2, c]
    if d is not None:
        ax.plot(d["ep"], -d["pen"], color="#5b8c3a", lw=1.4, label="− penalty (training)")
        ax.plot(d["se"], d["su"], color="#5b8c3a", lw=0, marker="o", ms=2.5, label="scoring set")
    style(ax, f"penalty only: {name}")
    if c == 0:
        ax.legend(fontsize=8, frameon=False)
axs[0, 0].set_ylabel("U = 30 × P(trigger)")
axs[1, 0].set_ylabel("U − penalty")
axs[2, 0].set_ylabel("− penalty (units of U)")
axs[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Trigger only (no reconstruction term) and penalty only (no utility), surrogate, SGD from the 100-detector grid",
             fontsize=12)
fig.savefig(os.path.join(OUT, "utility_curves_other.png"), dpi=150, facecolor="white")
plt.close(fig)

# ── Latin-hypercube repeats ──────────────────────────────────────────────────
CFG = [(f"sur5_lhs{{s}}_{k}", f"5°: {n}") for k, n in BANDS] + [(f"sur5_lhs{{s}}_{k}", f"5° + {n}") for k, n in PENS] + \
      [(f"pen_lhs{{s}}_{k}", f"penalty only: {n}") for k, n in PENS]
cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
fig, axs = plt.subplots(3, 4, figsize=(20, 13), constrained_layout=True)
for ax, (pat, name) in zip(axs.flat, CFG):
    for i, s in enumerate((1, 2, 3, 4, 5)):
        d = load(pat.format(s=s))
        if d is not None:
            ax.plot(d["se"], d["su"], color=cols[i], lw=1.4, label=f"seed {s}")
    style(ax, name)
axs[0, 0].legend(fontsize=8, frameon=False)
for ax in axs[:, 0]:
    ax.set_ylabel("scoring-set U (− penalty)")
fig.suptitle("Five Latin-hypercube starts (surrogate, SGD 1,000 steps): scoring-set U as optimised, one line per seed",
             fontsize=12)
fig.savefig(os.path.join(OUT, "utility_curves_lhs.png"), dpi=150, facecolor="white")
plt.close(fig)
print("[write]", OUT, "utility_curves_{5deg,other,lhs}.png")

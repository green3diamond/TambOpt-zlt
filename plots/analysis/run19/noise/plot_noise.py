"""Plots for noise_scores.json: disks vs radius per window (noise_disks.png), key layouts per detector count and window
(noise_best.png), background-only detectors and triggered share vs window (noise_bgonly.png)."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{H}/noise_scores.json")); S = D["scores"]
WS = [f"{w:g}" for w in D["windows_ns"]]
WLAB = {"0": "no noise", "128": "128 ns", "1000": "1 µs", "10000": "10 µs"}
LS = {"0": "-", "128": "--", "1000": ":", "10000": "-."}
NC = {10: "#c23b6b", 20: "#eb6834", 50: "#1baf7a", 100: "#2a78d6"}
TH = (("1", "1°"), ("2.5", "2.5°"), ("5", "5°"))

def disks_fig(pre, fname, ylab, extra):
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.4))
    for ax, (k, lab) in zip(axs, TH):
        for n, col in NC.items():
            rs = sorted(int(re.fullmatch(r"disk_r(\d+)_n\d+", x).group(1)) for x in S if re.fullmatch(rf"disk_r\d+_n{n}", x))
            for w in WS:
                ax.plot([r / 1e3 for r in rs], [100 * S[f"disk_r{r}_n{n}"][w][f"{pre}_{k}"] for r in rs], LS[w], color=col,
                        lw=2 if w == "0" else 1.4, label=f"{n} det." if w == "0" else None)
        ax.set(xscale="log", xlabel="disk radius [km]", ylabel=f"within {lab}{ylab} [%]", title=f"Within {lab}")
        ax.grid(alpha=0.25, lw=0.5)
    for w in WS:
        axs[1].plot([], [], LS[w], color="#52514e", label=WLAB[w])
    axs[0].legend(frameon=False, fontsize=9); axs[1].legend(frameon=False, fontsize=9, title="window")
    fig.suptitle("Centred disks with cosmic-ray background (1,826 /m²/s, 2 m² detectors, window centred on the front) · " +
                 extra + " · 5,000 showers", fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{H}/{fname}", dpi=110, facecolor="white"); plt.close(fig)



disks_fig("all", "noise_disks.png", " and triggered", "trigger ≥ 3 detectors > 1 particle")
disks_fig("notrig", "noise_disks_notrig.png", "", "no trigger")

# key layouts per N: grid, best disk (no noise), best optimised (no noise)
keys = []
for n in (100, 50, 20):
    disks = [x for x in S if re.fullmatch(rf"disk_r\d+_n{n}", x)]
    opts = [x for x in S if x.startswith("opt:") and not x.startswith("opt:start") and (x.endswith(f"_n{n}") if n != 100 else "_n" not in x)]
    keys += [(n, f"grid_n{n}", "grid"), (n, max(disks, key=lambda x: S[x]["0"]["all_1"]), "best disk")]
    if opts:
        keys.append((n, max(opts, key=lambda x: S[x]["0"]["all_1"]), "best optimised"))
WC = {"0": "#2a78d6", "128": "#1baf7a", "1000": "#eb6834", "10000": "#c23b6b"}
fig, axs = plt.subplots(1, 3, figsize=(22, 6.6), sharey=False)
for ax, (k, lab) in zip(axs, TH):
    x = np.arange(len(keys)); w_ = 0.8 / len(WS)
    for j, w in enumerate(WS):
        ax.bar(x + (j - (len(WS) - 1) / 2) * w_, [100 * S[nm][w][f"all_{k}"] for _, nm, _ in keys], w_ * 0.92, color=WC[w], label=WLAB[w])
    ax.set_xticks(x); ax.set_xticklabels([f"{n} det.\n{t}\n{nm.replace('opt:', '').replace(f'_n{n}', '')}" for n, nm, t in keys], fontsize=8)
    ax.set(ylabel=f"within {lab} and triggered [%]", title=f"Within {lab}"); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Key layouts with background, per window (best disk / optimised chosen without noise)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{H}/noise_best.png", dpi=105, facecolor="white"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.4))
wx = [float(w) for w in WS if w != "0"]
for n, col in NC.items():
    g = S.get(f"grid_n{n}")
    if g is None:
        continue
    axs[0].plot(wx, [g[f"{w:g}"]["bg_only"] for w in wx], "o-", color=col, lw=2, label=f"grid, {n} det.")
    axs[1].plot([1] + wx, [100 * g[w]["triggered"] for w in WS], "o-", color=col, lw=2, label=f"grid, {n} det.")
axs[0].set(xscale="log", yscale="log", xlabel="window [ns]", ylabel="background-only detectors per shower", title="Detectors firing on background alone")
axs[1].set(xscale="log", xlabel="window [ns] (1 = no noise)", ylabel="triggered showers [%]", title="Trigger ≥ 3 detectors with > 1 particle")
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{H}/noise_bgonly.png", dpi=110, facecolor="white"); plt.close(fig)
print("[write] noise_disks.png noise_best.png noise_bgonly.png")

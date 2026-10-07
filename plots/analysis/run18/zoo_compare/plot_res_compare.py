"""Angular resolution vs lit detectors for two recons, laid out exactly like lit_resolution.png (same panels,
titles, axis labels and y-ranges), spread layouts. python plot_res_compare.py <tagA> <tagB> ('zoo' = untagged)
-> lit_resolution_<B>_vs_<A>.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
A, B = sys.argv[1], sys.argv[2]
YLIM_ANG, YLIM_F1 = (0.9, 20.0), (0.0, 18.0)          # as lit_resolution.py
ld = lambda t: json.load(open(f"{Z}/lit_resolution{'' if t == 'zoo' else '_' + t}.json"))  # noqa: E731
J = {A: ld(A), B: ld(B)}
labs = J[A]["bins"]; x = np.arange(len(labs))
COL = {A: "#9aa4b1", B: "#2a78d6"}
fig, axs = plt.subplots(1, 3, figsize=(21, 6.4))
for t, j in J.items():
    r = j["spread"]; ok = [i for i, s in enumerate(r) if s]
    for key, ls in (("med", "-"), ("p68", "--")):
        axs[0].plot(x[ok], [r[i][key] for i in ok], "o" + ls, color=COL[t], lw=2,
                    label=f"{t} · {'median' if key == 'med' else '68%'}")
        axs[0].fill_between(x[ok], *np.array([r[i][f"{key}_band"] for i in ok]).T, color=COL[t], alpha=0.15, lw=0)
    axs[1].plot(x[ok], [100 * r[i]["f1"] for i in ok], "o-", color=COL[t], lw=2, label=f"{t} · within 1°")
    axs[1].fill_between(x[ok], *(100 * np.array([r[i]["f1_band"] for i in ok])).T, color=COL[t], alpha=0.15, lw=0)
    axs[2].bar(x[ok] + (-0.2 if t == A else 0.2), [r[i]["n_showers"] for i in ok], 0.38, color=COL[t], label=t)
axs[0].axhline(1, color="#c23b6b", lw=1.2, ls=":"); axs[0].text(0, 1.05, " 1°", color="#c23b6b")
axs[0].set(yscale="log", ylim=YLIM_ANG, ylabel="angle error [deg]", title="Angular resolution vs lit detectors (median, 68%)")
axs[1].set(ylim=YLIM_F1, ylabel="share within 1° [%]", title="Share within 1° (band: 16–84% over shower resamples)")
axs[2].set(ylabel="distinct showers in the bin (of 1,536)", title="How many different showers each bin holds")
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45); ax.set_xlabel("detectors with ≥ 1 count")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"{A} (grey) vs {B} (blue) · spread layouts (grids, optimized, disks ≥ 2 km, 500 m clusters) · "
             "same showers and layouts", fontsize=12.5)
fig.tight_layout()
out = f"{Z}/lit_resolution_{B}_vs_{A}.png"
fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight"); print("[write]", out)

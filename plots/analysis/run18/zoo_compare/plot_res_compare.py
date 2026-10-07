"""Angular resolution vs lit detectors for two recons (lit_resolution[_<tag>].json), spread and all layouts.
    python plot_res_compare.py <tagA> <tagB>     ('zoo' = the untagged files)  -> lit_resolution_<B>_vs_<A>.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
A, B = sys.argv[1], sys.argv[2]
ld = lambda t: json.load(open(f"{Z}/lit_resolution{'' if t == 'zoo' else '_' + t}.json"))  # noqa: E731
J = {A: ld(A), B: ld(B)}
labs = J[A]["bins"]; x = np.arange(len(labs))
COL = {A: "#9aa4b1", B: "#2a78d6"}
fig, axs = plt.subplots(1, 3, figsize=(21, 6.2))
for t, j in J.items():
    for grp, ls in (("spread", "-"), ("all", "--")):
        r = j[grp]; ok = [i for i, s in enumerate(r) if s]
        axs[0].plot(x[ok], [r[i]["med"] for i in ok], "o" + ls, color=COL[t], lw=2, label=f"{t} · {grp}")
        axs[0].fill_between(x[ok], *np.array([r[i]["med_band"] for i in ok]).T, color=COL[t], alpha=0.15, lw=0)
        axs[1].plot(x[ok], [r[i]["p68"] for i in ok], "o" + ls, color=COL[t], lw=2, label=f"{t} · {grp}")
        axs[2].plot(x[ok], [100 * r[i]["f1"] for i in ok], "o" + ls, color=COL[t], lw=2, label=f"{t} · {grp}")
        axs[2].fill_between(x[ok], *(100 * np.array([r[i]["f1_band"] for i in ok])).T, color=COL[t], alpha=0.15, lw=0)
for ax, yl, tt in zip(axs, ("median angle error [deg]", "68% angle error [deg]", "share within 1° [%]"),
                      ("Median error vs lit detectors", "68% error vs lit detectors", "Within 1° vs lit detectors")):
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45); ax.set(xlabel="detectors with ≥ 1 count", ylabel=yl, title=tt)
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_yscale("log"); axs[1].set_yscale("log"); axs[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"Angular resolution vs lit detectors: {A} (grey) vs {B} (blue) · solid = spread layouts, dashed = all · "
             "bands: 16–84% over shower resamples", fontsize=12.5)
fig.tight_layout()
out = f"{Z}/lit_resolution_{B}_vs_{A}.png"
fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight"); print("[write]", out)

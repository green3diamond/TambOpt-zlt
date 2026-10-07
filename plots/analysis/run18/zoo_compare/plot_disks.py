"""Disks of N detectors centred on the plane: share of showers within 1 / 2.5 / 5 deg vs disk radius, per N,
old recon (dashed) vs zoo recon (solid). Reads <tag>/{core_test_ndet,angleonly}/scores_pad_<theta>deg.json
(disk_r<R>_n<N> for N = 20, 50; disk_r<R> for N = 100). Writes disks_vs_radius.png.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
INK, GREY = "#1f2328", "#52514e"
N_COL = {20: "#eb6834", 50: "#1baf7a", 100: "#2a78d6"}


def disks(tag, th):
    out = {}
    for s in ("core_test_ndet", "angleonly"):
        for name, r in json.load(open(f"{Z}/{tag}/{s}/scores_pad_{th:g}deg.json")).items():
            m = re.fullmatch(r"disk_r(\d+)(?:_n(\d+))?", name)
            if m:
                out.setdefault(int(m.group(2) or 100), {})[int(m.group(1))] = 100 * r["frac_good_t"]
    return out


fig, axs = plt.subplots(1, 3, figsize=(20, 6.4))
for ax, th in zip(axs, (1, 2.5, 5)):
    for tag, ls, lw in (("old", "--", 1.4), ("zoo", "-", 2.2)):
        for n, d in sorted(disks(tag, th).items()):
            rs = sorted(d)
            ax.plot([r / 1e3 for r in rs], [d[r] for r in rs], ls, marker="o", ms=4 if tag == "zoo" else 3,
                    color=N_COL[n], lw=lw, alpha=1 if tag == "zoo" else 0.6,
                    label=f"{n} det. · {tag} recon")
    ax.axvline(3.0, color=GREY, lw=0.8, ls=":")
    ax.text(3.0, ax.get_ylim()[1] * 0.98, " half-width of the plane (3 km)", fontsize=8, color=GREY, va="top")
    ax.set(xscale="log", xlabel="disk radius [km] (detectors uniform in the disk, centred on the plane)",
           ylabel=f"showers within {th:g}° [%]", title=f"Perfect showers: within {th:g}°")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 3, 5, 10]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "3", "5", "10"])
    ax.grid(alpha=0.25, lw=0.5)
axs[0].legend(fontsize=8.5, frameon=False, ncol=2)
fig.suptitle("Disk size and number of detectors vs share of well-reconstructed showers (1,536 held-out showers; "
             "beyond 3 km the detectors off the plane snap to its nearest edge)", fontsize=12.5, color=INK)
fig.tight_layout()
out = f"{Z}/disks_vs_radius.png"
fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight")
for th in (1, 2.5, 5):
    z = disks("zoo", th)
    print(f"within {th:g}°: " + "  ".join(f"N={n}: best r={max(d, key=d.get)} m {max(d.values()):.1f}%" for n, d in sorted(z.items())))
print("[write]", out)

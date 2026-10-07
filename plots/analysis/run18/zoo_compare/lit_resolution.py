"""Angular resolution vs number of lit detectors (zoo recon), from lit_vs_angle_pairs.npz.

Per lit-count bin (detectors with >= 1 count): median and 68% angle error and the share within 1 deg, for all
distinct layouts and for spread layouts (grids, optimized, disks >= 2 km, 500 m clusters). Error bands: 200
bootstrap resamples of the 1,536 showers (a shower's pairs move together, so one big shower cannot carry a
bin). Also the number of distinct showers per bin. Writes lit_resolution.png and lit_resolution.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Z = os.path.dirname(os.path.abspath(__file__))
P = np.load(f"{Z}/lit_vs_angle_pairs.npz")
lit, ang, kind, ev = P["lit"], P["ang"], P["kind"], P["ev"]
n_ev = int(ev.max()) + 1
EDGES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 10 ** 9]
LABS = ["0", "1", "2", "3", "4", "5", "6-7", "8-9", "10-12", "13-15", "16-19", "20-29", "30+"]
rng = np.random.default_rng(0)
BOOT = [np.bincount(rng.integers(0, n_ev, n_ev), minlength=n_ev) for _ in range(200)]


def stats(sel):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = sel & (lit >= lo) & (lit < hi)
        a, e = ang[k], ev[k]
        if a.size < 50:
            out.append(None); continue
        med, p68, f1 = np.median(a), np.percentile(a, 68), np.mean(a < 1)
        bs = []
        for w in BOOT:
            ww = w[e].astype(float)
            if ww.sum() == 0:
                continue
            o = np.argsort(a); cw = np.cumsum(ww[o]) / ww.sum()
            bs.append((a[o][np.searchsorted(cw, 0.5)], a[o][np.searchsorted(cw, 0.68)], np.sum(ww * (a < 1)) / ww.sum()))
        lo_, hi_ = np.percentile(np.array(bs), [16, 84], axis=0)
        out.append(dict(n_pairs=int(a.size), n_showers=int(np.unique(e).size), med=float(med), p68=float(p68),
                        f1=float(f1), med_band=[float(lo_[0]), float(hi_[0])], p68_band=[float(lo_[1]), float(hi_[1])],
                        f1_band=[float(lo_[2]), float(hi_[2])]))
    return out


res = {"all": stats(np.ones_like(ang, bool)), "spread": stats(kind == "spread")}
fig, axs = plt.subplots(1, 3, figsize=(21, 6.4))
x = np.arange(len(LABS))
COL = {"all": "#1f2328", "spread": "#2a78d6"}
for g, r in res.items():
    ok = [i for i, s in enumerate(r) if s]
    for ax, key, ls in ((axs[0], "med", "-"), (axs[0], "p68", "--"), (axs[1], "f1", "-")):
        y = np.array([r[i][key] for i in ok]) * (100 if key == "f1" else 1)
        b = np.array([r[i][f"{key}_band"] for i in ok]) * (100 if key == "f1" else 1)
        ax.plot(x[ok], y, "o" + ls, color=COL[g], lw=2, label=f"{g} · {'median' if key == 'med' else '68%' if key == 'p68' else 'within 1°'}")
        ax.fill_between(x[ok], b[:, 0], b[:, 1], color=COL[g], alpha=0.15, lw=0)
    axs[2].bar(x[ok] + (-0.2 if g == "all" else 0.2), [r[i]["n_showers"] for i in ok], 0.38, color=COL[g], label=g)
axs[0].axhline(1, color="#c23b6b", lw=1.2, ls=":"); axs[0].text(0, 1.05, " 1°", color="#c23b6b")
axs[0].set(yscale="log", ylabel="angle error [deg]", title="Angular resolution vs lit detectors (median, 68%)")
axs[1].set(ylabel="share within 1° [%]", title="Share within 1° (band: 16–84% over shower resamples)")
axs[2].set(ylabel="distinct showers in the bin (of 1,536)", title="How many different showers each bin holds")
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(LABS, rotation=45); ax.set_xlabel("detectors with ≥ 1 count")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"Zoo recon · {len(P['names'])} distinct layouts × {n_ev} held-out showers · spread = grids, optimized, "
             "disks ≥ 2 km, 500 m clusters", fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{Z}/lit_resolution.png", dpi=120, facecolor="white", bbox_inches="tight")
json.dump(dict(bins=LABS, **res), open(f"{Z}/lit_resolution.json", "w"), indent=1)
for g, r in res.items():
    print(g)
    for l, s in zip(LABS, r):
        if s:
            print(f"  {l:>6}: median {s['med']:5.2f}° [{s['med_band'][0]:.2f}-{s['med_band'][1]:.2f}]  68% {s['p68']:5.2f}°  "
                  f"within 1° {100 * s['f1']:4.1f}%  showers {s['n_showers']:4d}  pairs {s['n_pairs']}")
print("[write] lit_resolution.png lit_resolution.json")

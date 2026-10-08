"""Run 19 layout tests: plots and tables from tests/scores.json and tests/pairs.npz (score_all.py). CPU only.

    disks_vs_radius.png    share within 1 / 2.5 / 5 deg vs disk radius, one line per detector count
    density_area.json      the density-vs-area grid (detectors x radius) and the two-path comparison, per threshold
    lit_resolution.png     median / 68% error and share within 1 deg vs lit detectors (bands: shower resamples)
    lit_geometry.png       median error and share within 1 deg vs the lit detectors' spacing / baseline
    best_per_family.png    best layout of each family per detector count, within 1 and 5 deg
    layout_catalog_p<k>.png  every layout, grouped by detector count, ranked by share within 1 deg

    python analyze.py [--tag <suffix>]
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
SUF = f"_{args.tag}" if args.tag else ""
T = os.path.join(C.RUN_LOCATION, "diagnostics", "tests")
RUN = os.path.basename(C.RUN_LOCATION.rstrip("/"))
S = json.load(open(f"{T}/scores{SUF}.json"))["scores"]
P = np.load(f"{T}/pairs{SUF}.npz")
names = list(P["names"]); lit, ang, lay, ev = P["lit"], P["ang"], P["lay"], P["ev"]
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
centre = mesh[:, :2].mean(0)
HALF_N = 0.5 * (m.n_max - m.n_min)                    # the plane's smaller horizontal half-width
TH = (1, 2.5, 5)
KEY = {1: "within1", 2.5: "within2p5", 5: "within5"}
FAMS = ("grid", "disk", "cluster", "core", "hybrid")
COL = {"grid": "#2a78d6", "disk": "#1baf7a", "cluster": "#eb6834", "core": "#c23b6b", "hybrid": "#8a5cd1"}
N_COL = {10: "#c23b6b", 20: "#eb6834", 50: "#1baf7a", 100: "#2a78d6"}


def family(x):
    return ("grid" if x.startswith("grid") else "disk" if x.startswith("disk") else "cluster" if x.startswith("clu")
            else "core" if re.match(r"core_r\d+", x) else "hybrid")


def ndet(x):
    return S[x]["n_det"]


def kind(x):
    """spread / compact / mid, with the disk thresholds scaled to the plane (run 18: >= 2 km / <= 500 m)."""
    f, r = family(x), re.search(r"_r(\d+)", x)
    r = int(r.group(1)) if r else 0
    if f == "grid":
        return "spread"
    if f == "disk":
        return "spread" if r >= 0.95 * HALF_N else ("compact" if r <= 0.25 * HALF_N else "mid")
    if f == "cluster":
        return "spread" if r >= 500 else ("compact" if r <= 200 else "mid")
    return "compact" if f == "core" else "mid"


kinds = np.array([kind(x) for x in names])[lay]

# ── disks vs radius + density/area grid ───────────────────────────────────────
disks = {}
for x in names:
    mm = re.fullmatch(r"disk_r(\d+)_n(\d+)", x)
    if mm:
        disks[(int(mm.group(2)), int(mm.group(1)))] = x
radii = sorted({r for _, r in disks}); counts = sorted({n for n, _ in disks})
u = np.random.default_rng(0).uniform([m.east_lo, m.n_min], [m.east_hi, m.n_max], (400000, 2))
rr = np.hypot(*(u - centre).T)
cover = {r: float(np.mean(rr < r)) for r in radii}
fig, axs = plt.subplots(1, 3, figsize=(20, 6.4))
grid_out = {}
for ax, th in zip(axs, TH):
    tab = {n: {r: 100 * S[disks[(n, r)]][KEY[th]] for r in radii} for n in counts}
    grid_out[f"{th:g}"] = {str(n): {str(r): v for r, v in d.items()} for n, d in tab.items()}
    for n in counts:
        ax.plot([r / 1e3 for r in radii], [tab[n][r] for r in radii], "o-", color=N_COL[n], lw=2, label=f"{n} det.")
    ax.axvline(HALF_N / 1e3, color="#52514e", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="disk radius [km] (centred on the plane)", ylabel=f"showers within {th:g}° [%]",
           title=f"Within {th:g}°")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].legend(frameon=False)
fig.suptitle(f"{RUN}: disk size and detector count vs share within 1 / 2.5 / 5° (dotted: N–S half-width "
             f"{HALF_N / 1e3:.1f} km; detectors off the plane snap to its edge)", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{T}/disks_vs_radius{SUF}.png", dpi=120, facecolor="white", bbox_inches="tight")
json.dump(dict(radii=radii, counts=counts, cover=cover, grid=grid_out), open(f"{T}/density_area{SUF}.json", "w"), indent=1)

# ── lit detectors: resolution and geometry ────────────────────────────────────
rng = np.random.default_rng(0)
B = int(ev.max()) + 1
BOOT = [np.bincount(rng.integers(0, B, B), minlength=B) for _ in range(200)]


def binned(x, edges, sel):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = sel & (x >= lo) & (x < hi)
        a, e = ang[k], ev[k]
        if a.size < 50:
            out.append(None); continue
        bs = []
        for w in BOOT:
            ww = w[e].astype(float)
            o = np.argsort(a); cw = np.cumsum(ww[o]) / max(ww.sum(), 1)
            bs.append((a[o][np.searchsorted(cw, 0.5)], a[o][np.searchsorted(cw, 0.68)], np.sum(ww * (a < 1)) / max(ww.sum(), 1)))
        lo_, hi_ = np.percentile(np.array(bs), [16, 84], axis=0)
        out.append(dict(med=np.median(a), p68=np.percentile(a, 68), f1=np.mean(a < 1), showers=np.unique(e).size,
                        mb=(lo_[0], hi_[0]), fb=(lo_[2], hi_[2])))
    return out


LIT_E = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 10 ** 9]
LIT_L = ["0", "1", "2", "3", "4", "5", "6-7", "8-9", "10-12", "13-15", "16-19", "20-29", "30+"]
fig, axs = plt.subplots(1, 3, figsize=(21, 6.4))
x = np.arange(len(LIT_L))
for g, col in (("all", "#1f2328"), ("spread", "#2a78d6")):
    r = binned(lit, LIT_E, np.ones_like(ang, bool) if g == "all" else kinds == "spread")
    ok = [i for i, s_ in enumerate(r) if s_]
    for key, ls in (("med", "-"), ("p68", "--")):
        axs[0].plot(x[ok], [r[i][key] for i in ok], "o" + ls, color=col, lw=2, label=f"{g} · {'median' if key == 'med' else '68%'}")
    axs[0].fill_between(x[ok], *np.array([r[i]["mb"] for i in ok]).T, color=col, alpha=0.15, lw=0)
    axs[1].plot(x[ok], [100 * r[i]["f1"] for i in ok], "o-", color=col, lw=2, label=f"{g} · within 1°")
    axs[1].fill_between(x[ok], *(100 * np.array([r[i]["fb"] for i in ok])).T, color=col, alpha=0.15, lw=0)
    axs[2].bar(x[ok] + (-0.2 if g == "all" else 0.2), [r[i]["showers"] for i in ok], 0.38, color=col, label=g)
axs[0].axhline(1, color="#c23b6b", lw=1.2, ls=":")
axs[0].set(yscale="log", ylim=(0.9, 20), ylabel="angle error [deg]", title="Angular resolution vs lit detectors (median, 68%)")
axs[1].set(ylim=(0, 18), ylabel="share within 1° [%]", title="Share within 1° (band: 16–84% over shower resamples)")
axs[2].set(ylabel=f"distinct showers in the bin (of {B})", title="How many different showers each bin holds")
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(LIT_L, rotation=45); ax.set_xlabel("detectors with ≥ 1 count")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"{RUN} · {len(names)} layouts × {B} held-out showers · spread = grids, disks ≥ {0.95 * HALF_N / 1e3:.1f} km, "
             "500 m+ clusters", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{T}/lit_resolution{SUF}.png", dpi=120, facecolor="white", bbox_inches="tight")

D_E = [0, 50, 100, 200, 300, 450, 600, 800, 1200, 1e9]
D_L = ["<50", "50-100", "100-200", "200-300", "300-450", "450-600", "600-800", "800-1200", "1200+"]
fig, axs = plt.subplots(2, 3, figsize=(21, 11))
x = np.arange(len(D_L))
for j, (key, title) in enumerate((("mean", "mean distance between lit detectors"),
                                  ("nn", "median nearest-neighbour spacing of lit detectors"),
                                  ("base", "baseline: largest distance between lit detectors"))):
    for gname, (lo, hi), col in (("3–5 lit", (3, 6), "#eb6834"), ("6–15 lit", (6, 16), "#2a78d6"), ("16+ lit", (16, 10 ** 6), "#8a5cd1")):
        r = binned(P[key], D_E, (lit >= lo) & (lit < hi))
        ok = [i for i, s_ in enumerate(r) if s_]
        if not ok:
            continue
        axs[0, j].plot(x[ok], [r[i]["med"] for i in ok], "o-", color=col, lw=2, label=gname)
        axs[0, j].fill_between(x[ok], *np.array([r[i]["mb"] for i in ok]).T, color=col, alpha=0.15, lw=0)
        axs[1, j].plot(x[ok], [100 * r[i]["f1"] for i in ok], "o-", color=col, lw=2, label=gname)
        axs[1, j].fill_between(x[ok], *(100 * np.array([r[i]["fb"] for i in ok])).T, color=col, alpha=0.15, lw=0)
    for i, (yl, ys, ylim) in enumerate((("median angle error [deg]", "log", (0.9, 20)), ("share within 1° [%]", "linear", (0, 25)))):
        axs[i, j].set(ylabel=yl, yscale=ys, ylim=ylim, xlabel=f"{title} [m]")
        axs[i, j].set_xticks(x); axs[i, j].set_xticklabels(D_L, rotation=45)
        axs[i, j].grid(alpha=0.25, lw=0.5); axs[i, j].legend(frameon=False, fontsize=9)
    axs[0, j].set_title(title.capitalize())
fig.suptitle(f"{RUN} · lit = ≥ 1 count · {len(names)} layouts × {B} showers · bands: 16–84% over shower resamples", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{T}/lit_geometry{SUF}.png", dpi=120, facecolor="white", bbox_inches="tight")

# ── best per family ───────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 4, figsize=(24, 9), sharey="row")
for i, th in enumerate((1, 5)):
    for j, n in enumerate((10, 20, 50, 100)):
        fs = [f for f in FAMS if any(family(x) == f and ndet(x) == n for x in names)]
        best = [max((x for x in names if family(x) == f and ndet(x) == n), key=lambda x: S[x][KEY[th]]) for f in fs]
        v = [100 * S[b][KEY[th]] for b in best]
        axs[i, j].bar(range(len(fs)), v, color=[COL[f] for f in fs])
        for k, (vv, b) in enumerate(zip(v, best)):
            axs[i, j].text(k, vv, f"{vv:.1f}\n{b.replace(f'_n{n}', '')}", ha="center", va="bottom", fontsize=7.5)
        axs[i, j].set_xticks(range(len(fs))); axs[i, j].set_xticklabels(fs)
        axs[i, j].set_title(f"{n} detectors · best layout within {th:g}°", fontsize=11); axs[i, j].grid(alpha=0.25, lw=0.5, axis="y")
    axs[i, 0].set_ylabel(f"share within {th:g}° [%]")
fig.suptitle(f"{RUN}: best layout of each family per detector count", fontsize=13)
fig.tight_layout(); fig.savefig(f"{T}/best_per_family{SUF}.png", dpi=120, facecolor="white", bbox_inches="tight")

# ── catalogue: grouped by detector count, ranked by share within 1 deg ────────
nn, base = P["nn"], P["base"]
ROWS, COLS = 5, 9
groups = [[c] for c in (100, 50, 20)] + [[25, 10, 5, 4]]
pages = []
for g in groups:
    items = [(r, x) for c in g for r, x in enumerate(sorted([x for x in names if ndet(x) == c],
                                                           key=lambda x: (-S[x]["within1"], -S[x]["within5"])), start=1)]
    for k in range(0, len(items), ROWS * COLS):
        pages.append((g, items[k:k + ROWS * COLS]))
for pg, (g, items) in enumerate(pages, start=1):
    fig, axs = plt.subplots(ROWS, COLS, figsize=(33.5, 18))
    for ax in axs.flat:
        ax.axis("off")
    for ax, (rank, x_) in zip(axs.flat, items):
        L = torch.load([l.split("=", 1)[1].strip() for l in open(f"{T}/layouts.txt") if l.startswith(x_ + "=")][0], map_location="cpu")
        k = lay == names.index(x_)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(mesh[:, 0] / 1e3, mesh[:, 1] / 1e3, s=0.4, color="#ecebe6", lw=0)
        ax.scatter(np.asarray(L["x"]) / 1e3, np.asarray(L["y"]) / 1e3, s=6, color=COL[family(x_)], lw=0)
        ax.set_aspect("equal")
        s_ = S[x_]
        ax.set_title(f"#{rank} of {ndet(x_)} det.  ·  {x_}\n1°: {100 * s_['within1']:.1f}%   5°: {100 * s_['within5']:.1f}%   "
                     f"median {s_['ang_med']:.2f}°\nlit {s_['lit_med']:.0f}   NN {np.nanmedian(nn[k]):.0f} m   "
                     f"baseline {np.nanmedian(base[k]):.0f} m", fontsize=10)
        for sp in ax.spines.values():
            sp.set_color(COL[family(x_)]); sp.set_linewidth(1.2)
    fig.suptitle(f"{RUN} · {' / '.join(map(str, g))} detectors: layouts ranked by share within 1° · frame: grid blue, "
                 "disk green, cluster orange, core red, hybrid purple\nlit = median detectors with ≥ 1 count per shower · "
                 "NN spacing / baseline = medians over showers with ≥ 2 lit", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.3)
    fig.savefig(f"{T}/layout_catalog{SUF}_p{pg}.png", dpi=80, facecolor="white"); plt.close(fig)

for th in TH:
    print(f"within {th:g}°: best disk radius " + "  ".join(
        f"N={n}: {max(radii, key=lambda r: S[disks[(n, r)]][KEY[th]]) / 1e3:g} km "
        f"({100 * max(S[disks[(n, r)]][KEY[th]] for r in radii):.1f}%)" for n in counts))
print(f"[write] {T}: disks_vs_radius, density_area, lit_resolution, lit_geometry, best_per_family, layout_catalog ({len(pages)} pages)")

"""Paper 2, fixed-layout experiments (disk size, disk with a hole, clustering) from the score JSONs of score_all.py:
extd_<run>_*.json (flow matching; 12 km: ext<run>_*.json, all showers) and extc8ft_<run>_*.json (CORSIKA 8, fine-tuned
recon). -> <out>/figures/results_2/exp19.pdf, exp20.pdf

    python scripts/paper2/plot_exp.py --out <overleaf clone> [--res <score dir>]
Columns: share within 1 and 5 deg; median angular error; detected and trigger share. Solid: flow matching; dashed: C8.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="paper folder (Overleaf clone)")
ap.add_argument("--res", default=os.environ.get("PAPER2_RES", "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/"
                                                "zdimitrov/tmp_rings/res2"), help="score JSON folder")
args = ap.parse_args()
D = args.res
FIG = os.path.join(args.out, "figures", "results_2"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "axes.grid": True,
                     "grid.alpha": 0.25})
SRC = {"fm": ("extd_", "-", "flow matching"), "c8": ("extc8ft_", "--", "CORSIKA 8")}   # triggered metrics
sys.path.insert(0, HERE)
from trigsel import trig_view


def load(run):
    out = {}
    for k, (pre, _, _) in SRC.items():
        d = {}
        ps = glob.glob(f"{D}/{pre}{run}_*.json") or (glob.glob(f"{D}/ext{run}_*.json") if k == "fm" else [])
        for p in ps:                                     # fm falls back to the older all-shower scoring
            d.update(json.load(open(p)))
        # reference: the uniform grid from the main scoring (same showers; shared layouts agree exactly)
        main = (f"scoresc8ft_{run}_*.json" if k == "c8" else
                f"scoresd_{run}_*.json" if glob.glob(f"{D}/scoresd_{run}_*.json") else f"scores{run}_*.json")
        for p in sorted(glob.glob(f"{D}/{main}")):
            if "c8test" not in p:
                r = json.load(open(p))
                if "ugrid" in r:
                    d["ugrid"] = r["ugrid"]
        out[k] = trig_view(d, acc=run != 19)
    return out


def series(d, pat, key_fn):
    """[(x, record)] of layouts whose name matches pat, sorted by x."""
    pts = [(key_fn(m), d[n]) for n in d for m in [re.fullmatch(pat, n)] if m]
    return sorted(pts, key=lambda t: t[0])


MARKERS = ("o", "s", "^", "D", "v", "P")


def src_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls=ls, label=nm) for _, ls, nm in SRC.values()]
    return h + [Line2D([], [], color="0.5", lw=0.8, label="uniform grid")]


def series_legend(a, fams, ncol, extra=()):
    """legend of the series of one column, below its bottom panel; `extra`: more handles (the line-style key)"""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker=MARKERS[i % len(MARKERS)], ms=3.5, lw=1, label=lab)
         for i, (lab, c, _, _) in enumerate(fams)] + list(extra)
    a.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=ncol, fontsize=6.5, frameon=False,
             handlelength=1.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)


def draw(axrow, data, fams, xlabel, logx, grid=True, legend=True):
    """fams: list of (label, colour, pattern, x-function). axrow: (accuracy, median, trigger) axes; None skips a panel."""
    acc, med, trg = axrow
    for src, (_, ls, sname) in SRC.items():
        d = data[src]
        for i, (lab, c, pat, fx) in enumerate(fams):
            pts = series(d, pat, fx)
            if not pts:
                continue
            x = np.array([p[0] for p in pts])
            g = lambda k: np.array([p[1][k] for p in pts])
            mk = MARKERS[i % len(MARKERS)]                               # one marker per series
            acc.plot(x, 100 * g("w1"), ls=ls, marker=mk, ms=2.5, color=c, label=f"{lab}" if src == "fm" else None)
            acc.plot(x, 100 * g("w5"), ls=ls, marker=mk, ms=2.5, color=c, alpha=0.45)
            if med is not None:
                med.plot(x, g("med"), ls=ls, marker=mk, ms=2.5, color=c)
            trg.plot(x, 100 * g("trig"), ls=ls, marker=mk, ms=2.5, color=c)
        if grid and "ugrid" in d:
            r = d["ugrid"]
            for a, v in ((acc, 100 * r["w1"]), (med, r["med"]), (trg, 100 * r["trig"])):
                if a is not None:
                    a.axhline(v, color="0.5", ls=ls, lw=0.8)
    for a in (acc, med, trg):
        if a is None:
            continue
        a.set_xlabel(xlabel)
        if logx:
            a.set_xscale("log")
    acc.set_ylabel(r"within $1^\circ$ (dark), $5^\circ$ (light) [%]")
    if med is not None:
        med.set_ylabel(r"median error [$^\circ$]"); med.set_yscale("log")
    trg.set_ylabel("trigger [%]")
    if legend:
        acc.legend(loc="best")


def src_legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls=ls, label=nm) for _, ls, nm in SRC.values()]
    h.append(Line2D([], [], color="0.5", lw=0.8, label="uniform grid"))
    fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=7.5)


# ── run 19 ──────────────────────────────────────────────────────────────────
d19 = load(19)
if d19["fm"] or d19["c8"]:
    fig, ax = plt.subplots(2, 3, figsize=(10, 4.8), layout="constrained")   # columns: experiments; rows: accuracy, trigger
    col = lambda j: (ax[0, j], None, ax[1, j])
    disk = [("disk, 99 detectors", "tab:blue", r"disk_r(\d+)", lambda m: int(m[1]) / 1000)]
    draw(col(0), d19, disk, "disk radius [km]", True, legend=False)
    series_legend(ax[1, 0], [], 1, extra=src_handles())       # single series: only the line-style key
    ax[0, 0].set_title("(a) disk size")
    hole = []
    for R, c in ((6000, "tab:red"), (2000, "tab:orange"), (1500, "tab:olive")):
        hole.append((f"annulus, outer {R / 1000:g} km", c, rf"annulus_rin(\d+)_R{R}_n\d+", lambda m: int(m[1]) / 1000))
    draw(col(1), d19, hole, "inner (hole) radius [km]", False, legend=False)
    series_legend(ax[1, 1], hole, 1)
    ax[0, 1].set_title("(b) disk with a hole")
    clus = [("33 triplets, uniform", "tab:green", r"tri33_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 0.5 km disk", "tab:cyan", r"tri33disk_R500_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 0.75 km disk", "tab:blue", r"tri33disk_R750_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 1 km disk", "tab:purple", r"tri33disk_R1000_r(\d+)", lambda m: int(m[1])),
            ("20 clusters of 5", "tab:brown", r"clu20x5_r(\d+)", lambda m: int(m[1])),
            ("25 clusters of 4", "tab:pink", r"clu25x4_r(\d+)", lambda m: int(m[1]))]
    draw(col(2), d19, clus, "cluster size [m]", True, legend=False)
    series_legend(ax[1, 2], clus, 2)
    ax[0, 2].set_title("(c) clustering")
    fig.savefig(f"{FIG}/exp19.pdf")

# ── run 20 ──────────────────────────────────────────────────────────────────
d20 = load(20)
if d20["fm"] or d20["c8"]:
    fig, ax = plt.subplots(2, 2, figsize=(10 * 2 / 3, 4.8), layout="constrained")   # as exp19: accuracy / trigger
    col = lambda j: (ax[0, j], None, ax[1, j])
    disk = [("disk, 99 detectors", "tab:blue", r"disk_r(\d+)", lambda m: int(m[1]))]
    draw(col(0), d20, disk, "disk radius [m]", False, legend=False)
    series_legend(ax[1, 0], [], 1, extra=src_handles())       # single series: only the line-style key
    ax[0, 0].set_title("(a) disk size")
    clus = [("33 triplets, uniform", "tab:green", r"tri33_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 150 m disk", "tab:cyan", r"tri33disk_R150_r(\d+)", lambda m: int(m[1]))]
    draw(col(1), d20, clus, "cluster size [m]", False, legend=False)
    series_legend(ax[1, 1], clus, 1)
    ax[0, 1].set_title("(b) clustering")
    fig.savefig(f"{FIG}/exp20.pdf")
# ── the layouts themselves: one row per experiment, real layout files on the detector region ─────────────────────
import torch
SITES = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites"


def name_of(n):
    m = re.fullmatch(r"disk_r(\d+)", n)
    if m:
        return f"disk {int(m[1]) / 1000:g} km" if int(m[1]) >= 1000 else f"disk {m[1]} m"
    m = re.fullmatch(r"annulus_rin(\d+)_R(\d+)_n\d+", n)
    if m:
        return f"annulus {int(m[1]) / 1000:g}–{int(m[2]) / 1000:g} km" if int(m[1]) else f"disk {int(m[2]) / 1000:g} km"
    m = re.fullmatch(r"tri33disk_R(\d+)_r(\d+)", n)
    if m:
        return f"33 triplets ({m[2]} m) in {int(m[1]) / 1000:g} km disk" if int(m[1]) >= 1000 else \
            f"33 triplets ({m[2]} m) in {m[1]} m disk"
    m = re.fullmatch(r"tri33_r(\d+)", n)
    if m:
        return f"33 triplets ({m[1]} m)"
    m = re.fullmatch(r"clu(\d+)x(\d+)_r(\d+)", n)
    if m:
        return f"{m[1]} clusters of {m[2]} ({m[3]} m)"
    return n


def layout_rows(run, rows, fname):
    """rows: [(row title, [layout names])]; each panel: the layout on the detector region (grey), title: name and the
    share within 1 and 5 deg of the flow-matching showers (all showers on the 12 km plane, triggered on Malata)"""
    d = load(run)["fm"]
    tris = np.load(f"{SITES}/{'plane' if run == 19 else 'malata'}.npz")["tris"]
    sc = 1000.0 if run == 19 else 1.0
    cE, cN = tris[:, :, 0].mean(1) / sc, tris[:, :, 1].mean(1) / sc
    ncol = max(len(r[1]) for r in rows)
    rh = 1.8 if run == 19 else 2.05                                  # row height [in]; run 19: tight rows
    fig, ax = plt.subplots(len(rows), ncol, figsize=(1.75 * ncol, rh * len(rows)), layout="constrained", squeeze=False)
    if run == 19:
        fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
    for i, (title, names) in enumerate(rows):
        for j in range(ncol):
            a = ax[i, j]
            if j >= len(names) or names[j] not in d:
                a.axis("off"); continue
            n = names[j]; r = d[n]
            L = torch.load(r["path"], map_location="cpu", weights_only=False)
            a.scatter(cE, cN, s=(0.3 if run == 19 else 1), color="0.85", lw=0, rasterized=True)
            a.scatter(np.asarray(L["x"]) / sc, np.asarray(L["y"]) / sc, s=2.5, color="0.15", lw=0)
            a.set_title(f"{name_of(n)}\n1°: {100 * r['w1']:.1f}%  5°: {100 * r['w5']:.1f}%", fontsize=6.5)
            a.set_aspect("equal"); a.grid(False); a.tick_params(labelsize=5.5)
        ax[i, 0].set_ylabel(f"{title}\nNorth [{'km' if run == 19 else 'm'}]", fontsize=7)
    fig.savefig(f"{FIG}/{fname}.pdf", dpi=300)
    plt.close(fig)


layout_rows(19, [("(a) disk size", ["disk_r500", "disk_r1000", "disk_r2000", "disk_r3000", "disk_r6000", "disk_r9000"]),
                 ("(b) disk with a hole", ["annulus_rin0_R6000_n100", "annulus_rin2000_R6000_n100",
                                           "annulus_rin3000_R6000_n100", "annulus_rin4000_R6000_n100",
                                           "annulus_rin500_R2000_n99", "annulus_rin1500_R2000_n99"]),
                 ("(c) clustering", ["tri33_r10", "tri33_r100", "tri33disk_R500_r25", "tri33disk_R1000_r25",
                                     "clu20x5_r500", "clu25x4_r500"])], "exp19_layouts")
layout_rows(20, [("(a) disk size", ["disk_r100", "disk_r150", "disk_r200", "disk_r300", "disk_r400"]),
                 ("(b) clustering", ["tri33_r10", "tri33_r25", "tri33disk_R150_r25"])], "exp20_layouts")
print("ok")

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
        out[k] = trig_view(d, acc=run != 19)
    return out


def series(d, pat, key_fn):
    """[(x, record)] of layouts whose name matches pat, sorted by x."""
    pts = [(key_fn(m), d[n]) for n in d for m in [re.fullmatch(pat, n)] if m]
    return sorted(pts, key=lambda t: t[0])


def draw(axrow, data, fams, xlabel, logx, grid=True):
    """fams: list of (label, colour, pattern, x-function)."""
    for src, (_, ls, sname) in SRC.items():
        d = data[src]
        for lab, c, pat, fx in fams:
            pts = series(d, pat, fx)
            if not pts:
                continue
            x = np.array([p[0] for p in pts])
            g = lambda k: np.array([p[1][k] for p in pts])
            mk = "o" if len(pts) > 1 else "s"
            axrow[0].plot(x, 100 * g("w1"), ls=ls, marker=mk, ms=2.5, color=c,
                          label=f"{lab}" if src == "fm" else None)
            axrow[0].plot(x, 100 * g("w5"), ls=ls, marker=mk, ms=2.5, color=c, alpha=0.45)
            axrow[1].plot(x, g("med"), ls=ls, marker=mk, ms=2.5, color=c)
            axrow[2].plot(x, 100 * g("trig"), ls=ls, marker=mk, ms=2.5, color=c)
        if grid and "grid" in d:
            r = d["grid"]
            for a, vals in ((axrow[0], (100 * r["w1"],)), (axrow[1], (r["med"],)), (axrow[2], (100 * r["trig"],))):
                for v in vals:
                    a.axhline(v, color="0.5", ls=ls, lw=0.8)
    for a in axrow:
        a.set_xlabel(xlabel)
        if logx:
            a.set_xscale("log")
    axrow[0].set_ylabel(r"within $1^\circ$ (dark), $5^\circ$ (light) [%]")
    axrow[1].set_ylabel(r"median error [$^\circ$]"); axrow[1].set_yscale("log")
    axrow[2].set_ylabel("trigger [%]")
    axrow[0].legend(loc="best")


def src_legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls=ls, label=nm) for _, ls, nm in SRC.values()]
    h.append(Line2D([], [], color="0.5", lw=0.8, label="grid"))
    fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=7.5)


# ── run 19 ──────────────────────────────────────────────────────────────────
d19 = load(19)
if d19["fm"] or d19["c8"]:
    fig, ax = plt.subplots(3, 3, figsize=(10, 9.4), layout="constrained")
    draw(ax[0], d19, [("disk, 99 detectors", "tab:blue", r"disk_r(\d+)", lambda m: int(m[1]) / 1000)],
         "disk radius [km]", True)
    ax[0, 1].set_title("(a) disk size")
    hole = []
    for R, c in ((6000, "tab:red"), (2000, "tab:orange"), (1500, "tab:olive")):
        hole.append((f"annulus, outer {R / 1000:g} km", c, rf"annulus_rin(\d+)_R{R}_n\d+", lambda m: int(m[1]) / 1000))
    draw(ax[1], d19, hole, "inner (hole) radius [km]", False)
    ax[1, 1].set_title("(b) disk with a hole")
    clus = [("33 triplets, uniform", "tab:green", r"tri33_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 0.5 km disk", "tab:cyan", r"tri33disk_R500_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 0.75 km disk", "tab:blue", r"tri33disk_R750_r(\d+)", lambda m: int(m[1])),
            ("33 triplets in 1 km disk", "tab:purple", r"tri33disk_R1000_r(\d+)", lambda m: int(m[1])),
            ("20 clusters of 5", "tab:brown", r"clu20x5_r(\d+)", lambda m: int(m[1])),
            ("25 clusters of 4", "tab:pink", r"clu25x4_r(\d+)", lambda m: int(m[1]))]
    draw(ax[2], d19, clus, "cluster size [m]", True)
    ax[2, 1].set_title("(c) clustering")
    src_legend(fig)
    fig.savefig(f"{FIG}/exp19.pdf")

# ── run 20 ──────────────────────────────────────────────────────────────────
d20 = load(20)
if d20["fm"] or d20["c8"]:
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
    draw(ax, d20, [("disk, 99 detectors", "tab:blue", r"disk_r(\d+)", lambda m: int(m[1])),
                   ("33 triplets, uniform", "tab:green", r"tri33_r(\d+)", lambda m: 400.0),
                   ("33 triplets in 150 m disk", "tab:purple", r"tri33disk_R150_r(\d+)", lambda m: 150.0)],
         "disk radius [m]", False)
    ax[1].set_title("disk size (triplets: at the radius they cover)")
    src_legend(fig)
    fig.savefig(f"{FIG}/exp20.pdf")
print("ok")

"""Run 20 SGD tests (slurm/run20_sgd_test.sh): the layout of each run at its latest checkpoint (opt_state.pt), against the
grid start, on the detector region. Rows: Poisson trigger, step trigger; columns: no background lr 0.3 / 1 / 3 m, 1 us +
kNN gate lr 1 m. Each panel: grid start (grey rings), current positions, the move of each detector, and the scoring-set U
(grid, best, latest). The detector region is drawn flat (light grey faces). --set sparse: the step-trigger runs of run20_sgd_sparse_test.sh (rows: 20-detector grid start,
33 triplets of 25 m start; columns lr 1 / 3 m), with the median distance to the nearest and second-nearest detector.
--set center: run20_sgd_center_test.sh (100 detectors from the centre start; rows Poisson / step trigger, columns lr 3 /
10 m), with the grid's spacing for reference.
--set energy / construct: the quick tests (slurm/run20_quick_tests.sh) from grid_n99: per energy band, per construction
penalty.
Writes <run 20>/diagnostics/r20b/sgd_layouts[_sparse|_center|_energy|_construct].png.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_sgd_layouts.py [--set sparse|center|energy|construct]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, sample_initial_layout_ne  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402

PRE = os.path.join(C.RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_kernel_")
ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=("grid", "sparse", "center", "energy", "construct", "construct_cmp"), default="grid")
ap.add_argument("--final-only", action="store_true", help="only the final positions (no start rings or moves); _final.png")
A = ap.parse_args()
SET, FINAL = A.set, A.final_only
TRI = os.path.join(C.RUN_LOCATION, "diagnostics", "triplets", "layouts", "tri33_r25_n99.pt")
if SET == "grid":       # (folder prefix, row name, colour, start: detector count or layout file)
    ROWS = [("sgdtest_", "Poisson trigger", "#2a6fb0", 100), ("sgdtest_step_", "step trigger", "#e07b39", 100)]
    COLS = [("nobg_lr0p3", "no bg, lr 0.3 m"), ("nobg_lr1", "no bg, lr 1 m"), ("nobg_lr3", "no bg, lr 3 m"),
            ("1us_lr1", "1 µs + kNN, lr 1 m")]
elif SET == "center":   # start (scheme, detectors)
    ROWS = [("sgdtest_center_poisson_", "Poisson trigger, centre start", "#2a6fb0", ("center", 100)),
            ("sgdtest_center_step_", "step trigger, centre start", "#e07b39", ("center", 100))]
    COLS = [("lr3", "lr 3 m"), ("lr10", "lr 10 m")]
elif SET == "construct_cmp":   # construction penalties: 5 deg utility (sur5) against trigger only (trg), surrogate
    ROWS = [("sur5_", "5° utility", "#2a6fb0", ("grid", 100)), ("trg_", "trigger only", "#e07b39", ("grid", 100)),
            ("pen_", "penalty only", "#5b8c3a", ("grid", 100))]
    COLS = [("slope", "slope limit"), ("access", "access distance"), ("pads", "station pads (K 20)"),
            ("contour", "contour rows (8)")]
elif SET in ("energy", "construct"):
    GRID99 = os.path.join(C.RUN_LOCATION, "diagnostics", "triplets", "layouts", "grid_n99.pt")
    QTAG = os.environ.get("QTAG", "quick")                 # quick5: the 5 deg tests
    ROWS = [(QTAG + "_", "", "#2a6fb0" if SET == "energy" else "#e07b39",
             ("grid", 100) if QTAG.startswith(("sur", "ker", "trg")) else GRID99)]   # surrogate runs: the 100-detector grid
    COLS = ([("e500_567", "log10 E 5.00-5.67"), ("e567_633", "log10 E 5.67-6.33"), ("e633_700", "log10 E 6.33-7.00"),
             ("eall", "all energies")] if SET == "energy" else
            [("slope", "slope limit"), ("access", "access distance"), ("pads", "station pads (K 20)"),
             ("contour", "contour rows (8)")])
else:
    ROWS = [("sgdtest_step_n20_", "20-det. grid start", "#2a6fb0", 20),
            ("sgdtest_step_tri25_", "triplets 25 m start", "#e07b39", TRI)]
    COLS = [("lr1", "lr 1 m"), ("lr3", "lr 3 m")]


def nn12(x, y):
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    np.fill_diagonal(d, np.inf)
    d = np.sort(d, 1)
    return np.median(d[:, 0]), np.median(d[:, 1])

m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)


def start(spec):
    if isinstance(spec, str):
        lay = torch.load(spec, map_location="cpu")
        return lay["x"].float().numpy(), lay["y"].float().numpy()
    scheme, n = spec if isinstance(spec, tuple) else ("grid", spec)
    return tuple(np.asarray(v, float) for v in sample_initial_layout_ne(m, n_units=n, scheme=scheme))

fig, axs = plt.subplots(len(ROWS), len(COLS), figsize=(4.6 * len(COLS), 5.6 * len(ROWS)), constrained_layout=True,
                        squeeze=False)
for r, (pre, rname, col, spec) in enumerate(ROWS):
    e0, n0 = start(spec)
    for c, (key, cname) in enumerate(COLS):
        ax = axs[r, c]
        ax.add_collection(PolyCollection(tri[:, :, :2], facecolors="#f2f2f2", edgecolors="#d0d0d0",
                                         linewidths=0.3))
        ax.set_xlim(tri[..., 0].min() - 60, tri[..., 0].max() + 60)
        ax.set_ylim(tri[..., 1].min() - 60, tri[..., 1].max() + 60)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        f = os.path.join(PRE + pre + key, "opt_state.pt")
        if pre + key in os.environ.get("QSKIP", "").split(",") or not os.path.exists(f):     # QSKIP: unreadable runs
            ax.set_title(f"{rname}, {cname}\n(no checkpoint)", fontsize=9)
            continue
        st = torch.load(f, map_location="cpu", weights_only=False)
        x, y = st["xy"]["x"].float().numpy(), st["xy"]["y"].float().numpy()
        sc = st["score_log"]
        mv = np.hypot(x - e0, y - n0)
        if not FINAL:
            ax.scatter(e0, n0, s=14, facecolors="none", edgecolors="#8a8a8a", lw=0.8, label="start")
            ax.add_collection(LineCollection(np.stack([np.c_[e0, n0], np.c_[x, y]], 1), colors=col, linewidths=0.8))
        ax.scatter(x, y, s=16 if FINAL else 10, color=col, lw=0, zorder=3, label=f"step {st['epoch']}")
        u0, ul = sc[0]["U_score"], sc[-1]["U_score"]
        ax.set_title(f"{rname + ', ' if rname else ''}{cname}\nscoring U: start {u0:.2f}, best {st['best_u']:.2f}, "
                     f"step {sc[-1]['epoch']} {ul:.2f} ({100 * (ul / u0 - 1):+.1f}%)\n"
                     f"move: median {np.median(mv):.1f} m, max {mv.max():.0f} m"
                     + ("\nnearest / 2nd nearest: {:.0f} / {:.0f} m -> {:.0f} / {:.0f} m".format(*nn12(e0, n0), *nn12(x, y))
                        if SET != "grid" else "")
                     + ("\n(grid: {:.0f} / {:.0f} m, U {})".format(*nn12(*start(100)), "15.05" if "poisson" in pre else "14.43")
                        if SET == "center" else ""), fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
for ax in axs[:, 0]:
    ax.set_ylabel("North [m]")
for ax in axs[-1]:
    ax.set_xlabel("East [m]")
fig.suptitle(("Run 20 quick tests" if SET in ("energy", "construct", "construct_cmp") else "Run 20 SGD tests")
             + ": layouts at the latest checkpoint" + ("\n" if SET != "grid" else " ")
             + ("(5°" if os.environ.get("QTAG", "").split("_")[0].endswith("5") and SET in ("energy", "construct") else "(1°")
             + " utility, default recon, expected counts, "
             "4 × 2,048 showers per step" + {"sparse": ", step trigger, no background)", "center": ", no background)",
              "energy": ", Poisson trigger, no background; 8 x 2,048 loaded per step, a band keeps ~1/3)",
              "construct": ", Poisson trigger, no background, one construction penalty each)"}.get(SET, ")"), fontsize=12)
out = os.path.join(C.RUN_LOCATION, "diagnostics", "r20b", "sgd_layouts" + ("" if SET == "grid" else "_" + SET)
                   + ({"quick5": "_5deg", "sur5": "_sur5deg", "sur5_1us": "_sur5deg_1us", "ker5": "_ker5deg", "trg": "_trigonly",
                       "trg_1us": "_trigonly_1us"}.get(os.environ.get("QTAG"), "")
                      if SET in ("energy", "construct") else "")
                   + ("_final" if FINAL else "") + ".png")
if SET == "construct_cmp":
    fig.suptitle("Run 20: final layouts with one construction penalty each, Stage 2 surrogate, SGD 1,000 steps from the "
                 "100-detector grid, no background\nrows: 5° Poisson utility + penalty; trigger only + penalty (no "
                 "reconstruction term); the penalty alone (no utility)", fontsize=12)
    out = os.path.join(C.RUN_LOCATION, "diagnostics", "r20b", "sgd_layouts_construct_sur5_vs_trigonly"
                       + ("_final" if FINAL else "") + ".png")
fig.savefig(out, dpi=170, facecolor="white")
print("[write]", out)

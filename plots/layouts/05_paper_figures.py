"""Regenerate every figure destined for the Elsevier paper, at the \\textwidth
fraction it's placed at, by importing and re-calling the code that already
draws each one — nothing here recomputes a plot from scratch.

Three sources, three ways of reusing them:

* `geometry_plots.py` — the three malata/tau notebook figures, already lifted
  out of `notebooks/malata_tau_geometry.ipynb` and
  `notebooks/explore_trigger_counts.ipynb` into functions parametrized by
  figsize + fontsize. This script only supplies the data (replicating the
  notebooks' own setup cell — see `load_malata_context` below) and the
  paper-scaled fontsizes.
* `02_plot_nn_target_vs_pred.py` (Deepsets conditional density / calibration /
  recon scatter) — filename starts with a digit so it's loaded by path, same
  as `explore_trigger_counts.ipynb` already does for `opt_plotting.py`. Its
  FS_*/FIGSIZE_* module constants are overridden in place before each call
  (its own convention: "05_paper_figures.py scales these up before calling
  in", per its header comment) so the *rendering* code is untouched — only
  the type scale changes.
* `opt_plotting.plot_density_heatmap` (detector placement density) — reused
  the same way, reading saved ensemble-optimizer runs from disk exactly like
  `replot_de_ensemble_up.py` does (`load_ensemble_run` below mirrors its
  `_load_dir`), so no optimizer re-run is needed.

`paper_style.py` holds the only actual "new" logic: converting a \\textwidth
fraction + a figure's existing (screen-tuned) figsize into the fontsize that
reads at the right size once LaTeX shrinks the artwork down for print.

Output: <outdir>/<name>.pdf, default `constants.PAPER_FIGURES_DIR` (the
out-of-repo figures archive — these are regenerable artifacts and are no
longer tracked in git), plus the
`includegraphics.tex` Results section referencing them. `prune_unused_outputs`
removes anything else in <outdir> at the end of a run (intermediates, and any
figure no longer referenced in `RESULTS_SECTION`).

Run from the v6 folder:
    python plots/layouts/05_paper_figures.py                    # everything
    python plots/layouts/05_paper_figures.py --only geometry     # the 5 malata/tau figures
    python plots/layouts/05_paper_figures.py --only deepsets     # conditional/calibration/recon
    python plots/layouts/05_paper_figures.py --only density      # the 4 layout-density heatmaps
    python plots/layouts/05_paper_figures.py --textwidth-pt 384.1  # elsarticle 1p (single column)
"""
import argparse
import importlib.util as _ilu
import math
import os
import sys

# `_HERE` is this file's own directory; `_V6` the repo root, found by walking up
# to the _pathfix.py marker instead of counting parents. Counting is what broke
# the old plots/single_species/ scripts: written for plots/*.py, they resolved
# the "repo root" to plots/ and could not import `modules` at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import PAPER_FIGURES_DIR
from plots.lib import paper_style as ps
from plots.lib import geometry_plots as gp


def _load_by_path(name, relpath):
    """Load another plots/ script by path, relative to plots/.

    Not an import: `02_nn_target_vs_pred` starts with a digit, so it is not a
    legal module name. opt_plotting goes the same way for symmetry.
    """
    spec = _ilu.spec_from_file_location(name, os.path.join(_V6, "plots", relpath))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Every figure: (output name, \textwidth fraction, drawn-figure width [in]).
# The width feeds paper_style.paper_fontsize together with the fraction; it is
# NOT the on-page size, it's the figsize the underlying function already draws
# at (see each function's own `figsize=`/`FIGSIZE_*` default).
# --------------------------------------------------------------------------- #
FIGURE_SPECS = {
    "mountain_3d":                    0.8,
    "tau_arrival_directions":         1.0,
    "showers_overlaid_mountain":      1.0,
    "fnn_conditional_electron":       1.0,
    "fnn_conditional_muon":           1.0,
    # The paper uses the CONDITIONAL density (P(pred|target) with the mean and
    # +/-1 sigma overlay), not the raw joint hexbin: the joint is dominated by
    # where the targets happen to sit, so it reads as quality when it is really
    # showing the prior. plot_recon_dual renders both; prune_unused_outputs
    # keeps whichever RESULTS_SECTION references.
    "recon_conditional":              1.0,
    # Intermediates only -- pasted two-up (not re-rendered) into optimized*.png
    # at frac=0.75 total, so each source panel's own final on-page width is
    # ~0.75/2 -- that's the frac their font sizing should target, not 0.5.
    "layout_density_k6_center":       0.375,
    "layout_density_k6_grid":         0.375,
    "layout_density_activation_center": 0.375,
    "layout_density_activation_grid":   0.375,
    "optimized_simple":               0.75,
    "optimized":                       0.75,
}


# --------------------------------------------------------------------------- #
# Group C -- geometry figures. Data-loading mirrors the notebooks' own setup
# cells; the drawing itself is entirely `geometry_plots.py`.
# --------------------------------------------------------------------------- #
def load_malata_context():
    """notebooks/malata_tau_geometry.ipynb cells 2+3+7 (mountain, tau corpus,
    layouts, local terrain mesh, detector-region triangles)."""
    from modules.geometry.mountain import load_tr_mountain, _ecef_to_enu
    from modules.geometry import SurfaceUpMap
    from modules.layouts.strategies import (
        layout_grid, layout_center_gaussian, layout_uniform_random, layout_latin_hypercube)
    from modules.constants import (
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        EAST_ENTRY, LAYER_EAST_DX, N_PLANES, N_DETECTORS,
        TAU_WHOLESKY_PATH, LOG_E_MIN, LOG_E_MAX)

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    mE, mN, mU = mtn.centroids_ENU[:, 0], mtn.centroids_ENU[:, 1], mtn.centroids_ENU[:, 2]
    surface = SurfaceUpMap.from_mountain(mtn).to("cpu")

    with h5py.File(TAU_WHOLESKY_PATH, "r") as f:
        tpos = f["position"][...]
        tEne = f["energy"][...]
    tE_e, tN_n, tU_u = tpos[0], tpos[1], tpos[2]
    inband = (tEne >= 10 ** LOG_E_MIN) & (tEne <= 10 ** LOG_E_MAX)

    rng = np.random.default_rng(0)
    layouts = {
        "grid":     layout_grid(mtn, N_DETECTORS, rng=rng),
        "center":   layout_center_gaussian(mtn, N_DETECTORS, sigma=400.0, rng=rng),
        "uniform":  layout_uniform_random(mtn, N_DETECTORS, rng=rng),
        "latin_hc": layout_latin_hypercube(mtn, N_DETECTORS, rng=rng),
    }
    layouts = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in layouts.items()}

    def det_up(N, E):
        return surface(torch.as_tensor(N, dtype=torch.float32),
                       torch.as_tensor(E, dtype=torch.float32)).numpy()

    LOC_R = 15000.0
    with h5py.File(GEOMETRY_PATH_RESOLVED, "r") as _f:
        _g = _f[GEOMETRY_GROUP]
        _V = _g["vertices"][...]; _F = _g["faces"][...] - 1
        _det = _g[DET_KEY][...] - 1; _loc = _g["location"][...]
    _vE, _vN, _vU = _ecef_to_enu(_V, float(_loc[0]), float(_loc[1]))
    _sane = (np.hypot(_vE, _vN) < LOC_R) & (np.abs(_vU) < 8000)
    _tri = _F[:, _sane[_F].all(axis=0)]
    terr = np.stack([_vE[_tri], _vN[_tri], _vU[_tri]], axis=-1).transpose(1, 0, 2)
    on_terr = np.hypot(tE_e, tN_n) < LOC_R

    def add_terrain(ax, alpha=0.5, r_lim=LOC_R):
        keep = (np.abs(terr[:, :, 0]) <= r_lim).all(1) & (np.abs(terr[:, :, 1]) <= r_lim).all(1)
        tris_ = terr[keep]
        up = tris_[:, :, 2].mean(1)
        nrm = matplotlib.colors.Normalize(up.min(), up.max())
        ax.add_collection3d(Poly3DCollection(
            tris_, facecolors=matplotlib.cm.terrain(nrm(up)), edgecolors="none", alpha=alpha))
        zt = tris_[:, :, 2]
        ax.set_xlim(-r_lim, r_lim); ax.set_ylim(-r_lim, r_lim)
        ax.set_zlim(float(zt.min()), float(zt.max()))
        ax.set_box_aspect((2 * r_lim, 2 * r_lim, float(np.ptp(zt))))

    # Detector-region triangles (malata_tau_geometry.ipynb cell 7), separate
    # from `terr` above (the wider local terrain patch cells 3/9 draw with
    # `add_terrain`) -- only `plot_mountain_3d` needs these.
    with h5py.File(GEOMETRY_PATH_RESOLVED, "r") as f:
        g = f[GEOMETRY_GROUP]
        _verts = g["vertices"][...]; _faces = g["faces"][...] - 1
        _det = g[DET_KEY][...] - 1; _loc = g["location"][...]
    _tri_ecef = _verts[:, _faces[:, _det]]
    _enu = _ecef_to_enu(_tri_ecef.reshape(3, -1), float(_loc[0]), float(_loc[1]))
    _E3, _N3, _U3 = (_enu[i].reshape(3, -1) for i in range(3))
    tris = np.stack([_E3, _N3, _U3], axis=-1).transpose(1, 0, 2)
    tri_up = _U3.mean(axis=0)

    with h5py.File(TAU_WHOLESKY_PATH, "r") as f:
        tdir = f["direction"][...]

    # section 9's `rod(i)`: a schematic rod of points along shower i's axis, drawn in
    # ENU at its decay vertex (C8 plane l at (l + 1) x 500 m along the axis).
    NLAY, PER, SIG = N_PLANES, 30, 20.0
    _lay = np.repeat(np.arange(NLAY), PER).astype(np.float32)
    _rng_rod = np.random.default_rng(0)

    def rod(i, rng=_rng_rod):
        p = np.array([tE_e[i], tN_n[i], tU_u[i]], dtype=np.float32)
        d = (tdir[:, i] / np.linalg.norm(tdir[:, i])).astype(np.float32)
        s = _lay * LAYER_EAST_DX
        east = p[0] + s * d[0] + rng.normal(0, SIG, _lay.size)
        north = p[1] + s * d[1] + rng.normal(0, SIG, _lay.size)
        up = p[2] + (_lay + 1.0) * LAYER_EAST_DX * d[2]
        return np.stack([east, north, up], axis=1)

    return dict(tris=tris, tri_up=tri_up, layouts=layouts, det_up=det_up,
               mE=mE, mN=mN, mU=mU, on_terr=on_terr, tE_e=tE_e, tN_n=tN_n, tU_u=tU_u,
               inband=inband, LOC_R=LOC_R, add_terrain=add_terrain, rod=rod,
               TAU_WHOLESKY_PATH=TAU_WHOLESKY_PATH)


def make_geometry_figures(outdir, frac_by_name, textwidth_pt):
    malata = load_malata_context()

    def pf(target_pt, drawn_w, frac):
        return ps.paper_fontsize(target_pt, drawn_w, frac, textwidth_pt)

    # rc_context fills in axis xlabel/ylabel/zlabel and tick fontsize for every
    # call below that doesn't take its own fs_label/fs_tick kwarg -- those
    # elements had no explicit fontsize before and rode matplotlib's fixed
    # ~10pt default, which is why a legend scaled up for print ended up
    # visually bigger than the (unscaled) axis title next to it.
    frac = frac_by_name["mountain_3d"]
    with matplotlib.rc_context(rc=ps.paper_rc(13, frac, textwidth_pt)):
        fig = gp.plot_mountain_3d(malata["tris"], malata["tri_up"], malata["layouts"],
                                  malata["det_up"], malata["mE"], malata["mN"], malata["mU"],
                                  figsize=(13, 9),
                                  fs_title=pf(ps.TARGET_TITLE_PT, 13, frac),
                                  fs_label=pf(ps.TARGET_LABEL_PT, 13, frac),
                                  fs_tick=pf(ps.TARGET_TICK_PT, 13, frac),
                                  fs_legend=pf(ps.TARGET_LEGEND_PT, 13, frac))
        ps.savefig_paper_formats(fig, os.path.join(outdir, "mountain_3d"), formats=("pdf",))
        plt.close(fig)

    frac = frac_by_name["tau_arrival_directions"]
    with matplotlib.rc_context(rc=ps.paper_rc(30, frac, textwidth_pt)):
        fig = gp.plot_tau_arrival_directions(
            malata["TAU_WHOLESKY_PATH"], malata["on_terr"], malata["tE_e"], malata["tN_n"],
            malata["tU_u"], malata["mE"], malata["mN"], malata["mU"], malata["tri_up"],
            malata["add_terrain"], figsize=(30, 13),
            fs_title=pf(ps.TARGET_TITLE_PT, 30, frac), fs_label=pf(ps.TARGET_LABEL_PT, 30, frac),
            fs_tick=pf(ps.TARGET_TICK_PT, 30, frac), fs_legend=pf(ps.TARGET_LEGEND_PT, 30, frac))
        ps.savefig_paper_formats(fig, os.path.join(outdir, "tau_arrival_directions"),
                                 formats=("pdf",))
        plt.close(fig)

    frac = frac_by_name["showers_overlaid_mountain"]
    with matplotlib.rc_context(rc=ps.paper_rc(15, frac, textwidth_pt)):
        fig = gp.plot_showers_overlaid(
            malata["rod"], malata["add_terrain"], malata["tE_e"], malata["tN_n"], malata["tU_u"],
            malata["inband"], malata["mE"], malata["mN"], malata["mU"], malata["LOC_R"],
            figsize=(15, 6.8),
            fs_title=pf(ps.TARGET_TITLE_PT, 15, frac), fs_label=pf(ps.TARGET_LABEL_PT, 15, frac),
            fs_tick=pf(ps.TARGET_TICK_PT, 15, frac), fs_legend=pf(ps.TARGET_LEGEND_PT, 15, frac))
        ps.savefig_paper_formats(fig, os.path.join(outdir, "showers_overlaid_mountain"),
                                 formats=("pdf",))
        plt.close(fig)

# --------------------------------------------------------------------------- #
# Group A -- Deepsets surrogate figures, via 02_plot_nn_target_vs_pred.py.
# --------------------------------------------------------------------------- #
def make_deepsets_figures(outdir, frac_by_name, textwidth_pt):
    nnplot = _load_by_path("nn_target_vs_pred", "training/02_nn_target_vs_pred.py")

    def pf(target_pt, drawn_w, frac):
        return ps.paper_fontsize(target_pt, drawn_w, frac, textwidth_pt)

    primary, xy, E_true, T_true, strat_ids = nnplot._load_corpus()
    species_ids = torch.load(os.path.join(nnplot.TRAINING_DATASET_FOLDER, "species_ids.pt")).long()

    cond_w = nnplot.FIGSIZE_CONDITIONAL[0]     # 13
    for tag, species_val in nnplot.SPECIES_TAGS:
        idx = torch.nonzero(species_ids == species_val).squeeze(-1)
        if idx.numel() == 0:
            print(f"[skip] no {tag} rows in corpus")
            continue
        fnn = nnplot.load_species_fnn(tag)
        val_idx = nnplot.shower_level_val_idx(strat_ids[idx], nnplot.VAL_FRAC, nnplot.FNN_VAL_SEED)

        cond_frac = frac_by_name[f"fnn_conditional_{tag}"]
        nnplot.FS_PANEL_TITLE = pf(ps.TARGET_PANEL_TITLE_PT, cond_w, cond_frac)
        nnplot.FS_LEGEND = pf(ps.TARGET_LEGEND_PT, cond_w, cond_frac)
        nnplot.FS_SUPTITLE = pf(ps.TARGET_TITLE_PT, cond_w, cond_frac)
        # rc_context covers this figure's xlabel/ylabel ("target"/"prediction")
        # and tick numbers, which _conditional_panel never gave a fontsize.
        with matplotlib.rc_context(rc=ps.paper_rc(cond_w, cond_frac, textwidth_pt)):
            nnplot._render_fnn_conditional(
                fnn, primary[idx], xy[idx], E_true[idx], T_true[idx], val_idx,
                os.path.join(outdir, f"fnn_conditional_{tag}.png"), formats=("pdf",))


    recon_w = nnplot.FIGSIZE_RECON[0]          # 18
    recon_frac = frac_by_name["recon_conditional"]
    nnplot.FS_LEGEND = pf(ps.TARGET_LEGEND_PT, recon_w, recon_frac)
    nnplot.FS_SUPTITLE_SCATTER = pf(ps.TARGET_TITLE_PT, recon_w, recon_frac)
    with matplotlib.rc_context(rc=ps.paper_rc(recon_w, recon_frac, textwidth_pt)):
        nnplot.plot_recon_dual(output_path=os.path.join(outdir, "recon_target_vs_pred.png"),
                               formats=("pdf",))


# --------------------------------------------------------------------------- #
# Group B -- detector placement density, from saved ensemble-optimizer runs.
# Mirrors replot_de_ensemble_up.py's `_load_dir`, generalized to any 04
# optimizer's opt_dir (they all save the identical layouts_all/layout_best/
# layout_mean triple -- see that script's own docstring).
# --------------------------------------------------------------------------- #
RUN_LOCATION = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
                "detector_optimization_v6/07_750k_primaires_meanvar")
DEFAULT_DENSITY_RUNS = {
    # K=6: run 7 (6 chains on top of run 6). (run 6's K=1 single-chain result
    # is not used by any figure the paper references, so it's not generated.)
    "layout_density_k6_center": os.path.join(
        RUN_LOCATION, "run 7 6 chains on top of run 6",
        "test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_center"),
    "layout_density_k6_grid": os.path.join(
        RUN_LOCATION, "run 7 6 chains on top of run 6",
        "test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_grid"),
    # lbfgs_activation optimizer, K=1 (top-level "current" run, not inside a
    # numbered run folder -- it has no ensemble counterpart on disk).
    "layout_density_activation_center": os.path.join(
        RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_activation_center"),
    "layout_density_activation_grid": os.path.join(
        RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_activation_grid"),
}

# Colorbar range every layout_density figure shares -- pinned, not
# auto-scaled: an auto vmax follows whatever run happens to have the hottest
# cell, so different runs land on different scales and the ones with a lower
# peak wash out to near-black. This matches replot_de_ensemble_up.py's own
# --vmax default.
DENSITY_VMAX = 0.2


def load_ensemble_run(opt_dir):
    la = torch.load(os.path.join(opt_dir, "layouts_all.pt"), map_location="cpu")
    lb = torch.load(os.path.join(opt_dir, "layout_best.pt"), map_location="cpu")
    aligned = np.asarray(la["aligned"])
    best_x = np.asarray(lb["x"]); best_y = np.asarray(lb["y"])
    return aligned, best_x, best_y


def make_density_figures(outdir, frac_by_name, textwidth_pt, run_dirs, vmax=DENSITY_VMAX):
    from modules.geometry import load_tr_mountain
    from modules.geometry import SurfaceUpMap
    from modules.constants import (
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES)

    optplt = _load_by_path("opt_plotting", "lib/opt_plotting.py")
    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to("cpu")

    fig_w = 14.0
    for name, opt_dir in run_dirs.items():
        if not os.path.isdir(opt_dir):
            print(f"[skip] {name}: run dir not found ({opt_dir})")
            continue
        frac = frac_by_name[name]
        optplt.FS_TITLE = ps.paper_fontsize(ps.TARGET_TITLE_PT, fig_w, frac, textwidth_pt)
        optplt.FS_LABEL = ps.paper_fontsize(ps.TARGET_LABEL_PT, fig_w, frac, textwidth_pt)
        optplt.FS_TICK = ps.paper_fontsize(ps.TARGET_TICK_PT, fig_w, frac, textwidth_pt)
        optplt.FS_LEGEND = ps.paper_fontsize(ps.TARGET_LEGEND_PT, fig_w, frac, textwidth_pt)

        aligned, best_x, best_y = load_ensemble_run(opt_dir)
        optplt.plot_density_heatmap(aligned, best_x, best_y, mountain,
                                    os.path.join(outdir, f"{name}.png"),
                                    surface=surface, fig_w=fig_w, dpi=ps.DEFAULT_DPI,
                                    vmax=vmax, formats=("png",))

    # Paper composites: center + grid scheme side by side for one optimizer
    # run each, matching the two-subplot "Optimization run" figures. Built
    # from the PNGs only (compositing a vector PDF page-by-page isn't a plain
    # paste); combine_side_by_side writes both an image and a PDF wrapping it.
    combine_side_by_side(
        os.path.join(outdir, "layout_density_activation_center.png"),
        os.path.join(outdir, "layout_density_activation_grid.png"),
        os.path.join(outdir, "optimized_simple"))
    combine_side_by_side(
        os.path.join(outdir, "layout_density_k6_center.png"),
        os.path.join(outdir, "layout_density_k6_grid.png"),
        os.path.join(outdir, "optimized"))


def combine_side_by_side(path_a, path_b, out_path_no_ext):
    """Paste two already-rendered PNGs into one side-by-side composite --
    used for the "Optimization run" figures, each of which is one optimizer
    run's center-scheme and grid-scheme density heatmap shown as a pair.
    Plain image concatenation (not a re-plotted 1x2 subplot): the two source
    PNGs are independently generated, full-resolution `plot_density_heatmap`
    outputs, so pasting keeps every pixel instead of re-rendering at a
    shared, smaller canvas size. Writes only `{out_path_no_ext}.pdf` (Pillow
    wraps the raster as a single-page PDF); the source PNGs stay PNG since
    compositing needs raster input, but nothing needs a PNG copy of the
    composite itself -- the paper includes the PDF."""
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        print(f"[skip] {out_path_no_ext}: missing source ({path_a!r} or {path_b!r})")
        return
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # two 400dpi print-size panels legitimately exceed the default bomb-detection limit
    a, b = Image.open(path_a), Image.open(path_b)
    h = min(a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h))
    b = b.resize((round(b.width * h / b.height), h))
    combo = Image.new("RGBA", (a.width + b.width, h), (255, 255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (a.width, 0))
    out = f"{out_path_no_ext}.pdf"
    combo.convert("RGB").save(out)
    print(f"[paper] wrote {out}")


RESULTS_SECTION = r"""\section{Results}
\subsection{Decay locations}
\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{figures/results/tau_arrival_directions.pdf}
    \caption{Caption}
    \label{fig:decay-locations}
\end{figure}

\subsection{Matching decay locations to surrogate outputs}


\subsection{Mountain slope}

\begin{figure}
    \centering
    \includegraphics[width=0.8\linewidth]{figures/results/mountain_3d.pdf}
    \caption{Caption}
    \label{fig:mountain-slope}
\end{figure}

\subsection{Overlaying showers on mountain slope}
\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{figures/results/showers_overlaid_mountain.pdf}
    \caption{Caption}
    \label{fig:shower-over-mountain-global}
\end{figure}

\subsection{Detector response model}

\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{figures/results/fnn_conditional_electron.pdf}
    \caption{Caption}
    \label{fig:fnn-electron-conditional}
\end{figure}


\subsection{Reconstruction network}
\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{figures/results/recon_conditional.pdf}
    \caption{Caption}
    \label{fig:recon-target-vs-output}
\end{figure}

\subsection{Optimization run}

\begin{figure}
    \centering
    \includegraphics[width=0.75\linewidth]{figures/results/optimized_simple.pdf}
    \caption{Caption}
    \label{fig:optimized-simple}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.75\linewidth]{figures/results/optimized.pdf}
    \caption{Caption}
    \label{fig:optimized}
\end{figure}
"""


def write_includegraphics(outdir, frac_by_name):
    """Writes the paper's curated Results section, with every placeholder
    filename resolved to the actual figure this script produces:

        decay_location_3d          -> tau_arrival_directions
        mountain_slope              -> mountain_3d
        shower_over_mountain_global -> showers_overlaid_mountain
        fnn_electron_conditional    -> fnn_conditional_electron
        recon_target_vs_output      -> recon_conditional
        optimized_simple            -> layout_density_activation_{center,grid} composite
        optimized                   -> layout_density_k6_{center,grid} composite

    The plain (unsuffixed) "shower_over_mountain" figure block from the
    original draft had nothing to point at once "global"/"activation" were
    resolved to two real, distinct figures, so it's dropped rather than
    guessed. \\label{}s were also de-duplicated (the draft reused
    fig:placeholder everywhere, which is a LaTeX compile error).

    Every filename here uses this script's own output names, so what you get
    is a corrected copy of the section text you'd paste into the paper's
    Results section, backed by the files already sitting in `outdir`.

    All \\includegraphics targets reference `.pdf` -- see `_savefig_multi`
    in `02_plot_nn_target_vs_pred.py` / the `formats=` plumbing throughout
    this file: every figure the paper includes is written as PDF only."""
    path = os.path.join(outdir, "includegraphics.tex")
    with open(path, "w") as f:
        f.write(RESULTS_SECTION)
    print(f"[paper] wrote {path}")


def prune_unused_outputs(outdir):
    """Delete every file in `outdir` that isn't one of RESULTS_SECTION's own
    \\includegraphics targets (all .pdf -- this script writes every figure
    the paper includes as PDF only, no PNG sibling) or includegraphics.tex
    itself -- e.g. the individual
    layout_density_{k6,activation}_{center,grid} PNGs (intermediates the
    optimized*.pdf composites are pasted from, kept as PNG since compositing
    needs raster input). Self-maintaining: add or remove a figure from
    RESULTS_SECTION and the next run's cleanup follows automatically."""
    import re
    needed = {"includegraphics.tex"}
    for stem in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/results/([^}]+)\}",
                           RESULTS_SECTION):
        needed.add(stem)

    if not os.path.isdir(outdir):
        return
    for name in sorted(os.listdir(outdir)):
        if name not in needed:
            path = os.path.join(outdir, name)
            if os.path.isfile(path):
                os.remove(path)
                print(f"[paper] removed {path} (not referenced in RESULTS_SECTION)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["geometry", "deepsets", "density"], default=None,
                    help="render just one group (default: all three)")
    ap.add_argument("--outdir", default=PAPER_FIGURES_DIR,
                    help="where the PDFs land (default: the out-of-repo "
                         "figures archive, modules.constants.PAPER_FIGURES_DIR)")
    ap.add_argument("--textwidth-pt", type=float, default=ps.TEXTWIDTH_PT,
                    help="\\the\\textwidth from the Elsevier .tex, in pt "
                         f"(default {ps.TEXTWIDTH_PT:g}, elsarticle 5p/3p two-column)")
    ap.add_argument("--k6-center", default=DEFAULT_DENSITY_RUNS["layout_density_k6_center"])
    ap.add_argument("--k6-grid", default=DEFAULT_DENSITY_RUNS["layout_density_k6_grid"])
    ap.add_argument("--activation-center",
                    default=DEFAULT_DENSITY_RUNS["layout_density_activation_center"])
    ap.add_argument("--activation-grid",
                    default=DEFAULT_DENSITY_RUNS["layout_density_activation_grid"])
    ap.add_argument("--density-vmax", type=float, default=DENSITY_VMAX,
                    help="colorbar upper limit shared by every layout_density "
                         f"figure (default {DENSITY_VMAX:g}); <=0 auto-scales per run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 72)
    print(f"v6/plots/layouts/05_paper_figures.py  textwidth={args.textwidth_pt:g}pt  -> {args.outdir}")
    print("=" * 72)

    if args.only in (None, "geometry"):
        make_geometry_figures(args.outdir, FIGURE_SPECS, args.textwidth_pt)
    if args.only in (None, "deepsets"):
        make_deepsets_figures(args.outdir, FIGURE_SPECS, args.textwidth_pt)
    if args.only in (None, "density"):
        run_dirs = {
            "layout_density_k6_center": args.k6_center,
            "layout_density_k6_grid": args.k6_grid,
            "layout_density_activation_center": args.activation_center,
            "layout_density_activation_grid": args.activation_grid,
        }
        vmax = args.density_vmax if args.density_vmax and args.density_vmax > 0 else None
        make_density_figures(args.outdir, FIGURE_SPECS, args.textwidth_pt, run_dirs, vmax=vmax)

    write_includegraphics(args.outdir, FIGURE_SPECS)
    prune_unused_outputs(args.outdir)


if __name__ == "__main__":
    main()

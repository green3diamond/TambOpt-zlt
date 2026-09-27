#!/usr/bin/env python
"""
Poster figure: CORSIKA vs ML shower observables, all primaries overlaid.

Layout : 1 row x 5 observable columns ("All" showers only)
         every panel contains electron + photon + muon
         solid  = CORSIKA        dashed = ML        colour = primary
Theme  : dark poster background (#0B132B)
"""

import gc
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

# ------------------------------------------------------------------ config

BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training"

DATASETS = [
    dict(
        label = r"$e^-$",
        color = "#5BC8FF",                       # cyan
        ml    = f"{BASE}/best_ckpts/20260519_185649_Electron-Allshower/samples01.h5",
        sim   = f"{BASE}/h5_files_v3/combined_electrons_test.h5",
    ),
    dict(
        label = r"$\gamma$",
        color = "#FFC857",                       # amber
        ml    = f"{BASE}/best_ckpts/20260724_074020_Photon-Allshower/samples01.h5",
        sim   = f"{BASE}/h5_files_v3/combined_photons_test.h5",
    ),
    dict(
        label = r"$\mu^\pm$",
        color = "#FF6B9D",                       # magenta
        ml    = f"{BASE}/best_ckpts/20260520_160031_Muons-Allshower/samples03.h5",
        sim   = f"{BASE}/h5_files_v3/combined_muons_test.h5",
    ),
]

OUT = f"{BASE}/h5_files_v3/shower_observables_poster_1x5.pdf"

NUM_LAYERS = 24
US         = 1e6     # seconds -> microseconds
THRESHOLD  = 1e-4    # GeV
N_RADIAL   = 35
N_EBINS    = 60
N_TBINS    = 40
SEED       = 42

# ------------------------------------------------------------------ theme

BG       = "#0B132B"
FG       = "#E8ECF5"
MUTED    = "#8FA0BF"
GRIDC    = "#1C2541"

mpl.rcParams.update({
    "figure.facecolor":  BG,
    "savefig.facecolor": BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "text.color":        FG,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        GRIDC,
    "legend.facecolor":  BG,
    "legend.edgecolor":  MUTED,
    "legend.framealpha": 0.85,
    "font.family":       "DejaVu Sans",
    "axes.linewidth":    1.1,
    "pdf.fonttype":      42,
})

FS_TITLE = 17
FS_LABEL = 15
FS_TICK  = 12
FS_LEG   = 15

SIM_LS, ML_LS = "-", "--"
SIM_LW, ML_LW = 2.4, 2.2
BAND_ALPHA    = 0.16


# ------------------------------------------------------------------ loaders

def load_file(path):
    print(f"  Reading {path} ...")
    with h5py.File(path, "r") as f:
        pdg   = f["pdg"][:]
        raw   = f["showers"][:]
        shape = f["shape"][:]

    N, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"    {N} showers, {max_pts} max pts, {ncols} cols")

    lengths  = np.array([len(r) // ncols for r in raw], dtype=np.int32)
    flat_all = np.concatenate([np.asarray(r, dtype=np.float32) for r in raw])
    hits_all = flat_all.reshape(-1, ncols)

    pts = np.zeros((N, max_pts, ncols), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for i, (start, L) in enumerate(zip(offsets[:-1], lengths)):
        if L > 0:
            pts[i, :L] = hits_all[start:start + L]

    del raw, flat_all, hits_all
    gc.collect()

    pts[..., 3] = np.where(pts[..., 3] >= THRESHOLD, pts[..., 3], 0.0)
    return pts, pdg, ncols


# ------------------------------------------------------------------ observables

def compute_longitudinal(pts, num_layers=NUM_LAYERS):
    N = pts.shape[0]
    out = np.zeros((N, num_layers), dtype=np.float64)
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    energies   = pts[..., 3].astype(np.float64)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts.shape[1], axis=1)
    np.add.at(out, (shower_idx, layer_idx), energies)
    return out


def cog_centred(pts):
    """Above-threshold hits re-centred per shower on the energy-weighted CoG.
    Returns (dist, energy, time, mean_hit_time_per_shower)."""
    has_t = pts.shape[2] >= 5
    d_all, e_all, t_all, mht = [], [], [], []
    for i in range(pts.shape[0]):
        e = pts[i, :, 3].astype(np.float64)
        m = e > 0
        if not m.any():
            continue
        e_h = e[m]
        x_h = pts[i, :, 0][m].astype(np.float64)
        y_h = pts[i, :, 1][m].astype(np.float64)
        s   = e_h.sum()
        xc  = (x_h * e_h).sum() / s
        yc  = (y_h * e_h).sum() / s
        d_all.append(np.sqrt((x_h - xc) ** 2 + (y_h - yc) ** 2))
        e_all.append(e_h)
        if has_t:
            t_h = pts[i, :, 4][m].astype(np.float64)
            t_all.append(t_h)
            mht.append(t_h.mean())
    return (np.concatenate(d_all),
            np.concatenate(e_all),
            np.concatenate(t_all) if has_t else None,
            np.asarray(mht) * US if has_t else None)


def radial_energy_profile(dist, ener, n_events, r_max, num_bins=N_RADIAL):
    mean,  edges, _ = binned_statistic(dist, ener, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(dist, ener, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(dist, ener, bins=num_bins, statistic="count", range=(0, r_max))
    return edges, mean * count / n_events, (std / np.sqrt(count.clip(min=1))) * (count / n_events)


def radial_time_profile(dist, times, r_max, num_bins=N_RADIAL):
    if times is None:
        return None, None, None
    mean,  edges, _ = binned_statistic(dist, times, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(dist, times, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(dist, times, bins=num_bins, statistic="count", range=(0, r_max))
    return edges, mean, std / np.sqrt(count.clip(min=1))


def capped_indices(n_sim, n_ml, seed=SEED):
    n = min(n_sim, n_ml)
    rng = np.random.default_rng(seed)
    return (rng.choice(n_sim, size=n, replace=False),
            rng.choice(n_ml,  size=n, replace=False), n)


# ------------------------------------------------------------------ pass 1: reduce

reduced = []
for cfg in DATASETS:
    print(f"\n=== {cfg['label']} ===")
    print("Loading Simulated...")
    s_pts, s_pdg, s_ncols = load_file(cfg["sim"])
    print("Loading ML...")
    m_pts, m_pdg, m_ncols = load_file(cfg["ml"])

    s_idx, m_idx, n = capped_indices(len(s_pdg), len(m_pdg))
    s_sel, m_sel = s_pts[s_idx], m_pts[m_idx]
    del s_pts, m_pts
    gc.collect()

    print("  Reducing ...")
    rec = dict(cfg)
    rec["n"] = n
    rec["s_long"] = compute_longitudinal(s_sel)
    rec["m_long"] = compute_longitudinal(m_sel)
    rec["s_d"], rec["s_e"], rec["s_t"], rec["s_mht"] = cog_centred(s_sel)
    rec["m_d"], rec["m_e"], rec["m_t"], rec["m_mht"] = cog_centred(m_sel)
    reduced.append(rec)

    del s_sel, m_sel
    gc.collect()

# global binning so the three primaries are directly comparable
R_CAP = 5000.0   # hard upper limit on the radial axis [m]
R_MAX = min(float(np.percentile(
    np.concatenate([r[k] for r in reduced for k in ("s_d", "m_d")]), 99)), R_CAP)
E_MIN = 1e0
E_MAX = float(max(r[k].max() for r in reduced for k in ("s_e", "m_e")))
EBINS = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_EBINS)
E_CTR = np.sqrt(EBINS[:-1] * EBINS[1:])

_has_t = all(r["s_mht"] is not None and r["m_mht"] is not None for r in reduced)
if _has_t:
    _t = np.concatenate([r[k] for r in reduced for k in ("s_mht", "m_mht")])
    TBINS = np.linspace(_t.min(), _t.max(), N_TBINS)
    T_CTR = 0.5 * (TBINS[:-1] + TBINS[1:])

print(f"\nGlobal R_MAX = {R_MAX:.1f} m,  E_MAX = {E_MAX:.3g} GeV\n")


# ------------------------------------------------------------------ panels

fig = plt.figure(figsize=(28, 6.8))
outer_gs = gridspec.GridSpec(
    1, 5, figure=fig, wspace=0.26,
    top=0.815, bottom=0.105, left=0.040, right=0.995,
)

# vertical budget above the axes:
#   0.815 - 0.875  panel titles
#   0.885 - 0.985  legend band


def make_panel(cell):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=cell, height_ratios=[3, 1], hspace=0.06)
    ax  = fig.add_subplot(inner[0])
    axr = fig.add_subplot(inner[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    for a in (ax, axr):
        a.set_facecolor(BG)
        a.tick_params(labelsize=FS_TICK, colors=MUTED)
        for sp in a.spines.values():
            sp.set_color(MUTED)
        a.grid(True, color=GRIDC, lw=0.7, alpha=0.85)
        a.set_axisbelow(True)
    axr.axhline(1.0, color=MUTED, lw=1.0, ls="--", alpha=0.9)
    axr.set_ylabel("ML / Sim", fontsize=FS_TICK, color=FG)
    return ax, axr


def curve(ax, x, y, err, color, ls, lw, edges=None, steps=False):
    """Draw one curve + its +/-1 sigma band."""
    if edges is not None:
        ax.stairs(y, edges, color=color, lw=lw, linestyle=ls)
        ax.stairs(y + err, edges, baseline=y - err, color=color,
                  alpha=BAND_ALPHA, fill=True)
    else:
        ax.plot(x, y, color=color, lw=lw, ls=ls,
                drawstyle="steps-mid" if steps else "default")
        ax.fill_between(x, y - err, y + err, color=color, alpha=BAND_ALPHA,
                        step="mid" if steps else None, lw=0)


def ratio(axr, x, sim, ml, sim_err, ml_err, color, steps=False):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sim > 0, ml / sim, np.nan)
        e = r * np.sqrt(
            np.where(sim > 0, (sim_err / np.where(sim > 0, sim, 1)) ** 2, 0) +
            np.where(ml  > 0, (ml_err  / np.where(ml  > 0, ml,  1)) ** 2, 0))
    axr.plot(x, r, color=color, lw=1.9,
             drawstyle="steps-mid" if steps else "default")
    axr.fill_between(x, r - e, r + e, color=color, alpha=0.22,
                     step="mid" if steps else None, lw=0)


layers = np.arange(1, NUM_LAYERS + 1)

# ---- Col 0: Longitudinal energy profile --------------------------------
ax, axr = make_panel(outer_gs[0, 0])
for r in reduced:
    c, n = r["color"], r["n"]
    sm, se = r["s_long"].mean(0), r["s_long"].std(0) / np.sqrt(n)
    mm, me = r["m_long"].mean(0), r["m_long"].std(0) / np.sqrt(n)
    curve(ax, layers, sm, se, c, SIM_LS, SIM_LW, steps=True)
    curve(ax, layers, mm, me, c, ML_LS,  ML_LW,  steps=True)
    ratio(axr, layers, sm, mm, se, me, c, steps=True)
ax.set_yscale("log")
ax.set_ylabel("Mean Energy [GeV]", fontsize=FS_LABEL)
ax.set_xticks(list(np.arange(1, NUM_LAYERS + 1, 4)) + [NUM_LAYERS])
ax.set_title("Longitudinal Energy Profile", fontsize=FS_TITLE, pad=10)
axr.set_ylim(0.8, 1.2); axr.set_yticks([0.8, 1.0, 1.2])
axr.set_xlabel("Observing Plane", fontsize=FS_LABEL)

# ---- Col 1: Cell energy spectrum ---------------------------------------
ax, axr = make_panel(outer_gs[0, 1])
for r in reduced:
    c = r["color"]
    sc, _ = np.histogram(r["s_e"], bins=EBINS)
    mc, _ = np.histogram(r["m_e"], bins=EBINS)
    curve(ax, None, sc.astype(float), np.sqrt(sc), c, SIM_LS, SIM_LW, edges=EBINS)
    curve(ax, None, mc.astype(float), np.sqrt(mc), c, ML_LS,  ML_LW,  edges=EBINS)
    ratio(axr, E_CTR, sc.astype(float), mc.astype(float),
          np.sqrt(sc), np.sqrt(mc), c)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(E_MIN, E_MAX)
ax.set_ylabel("Number of Cells", fontsize=FS_LABEL)
ax.set_title("Cell Energy Spectrum", fontsize=FS_TITLE, pad=10)
axr.set_xscale("log"); axr.set_ylim(0.5, 1.5); axr.set_yticks([0.5, 1.0, 1.5])
axr.set_xlabel("Cell Energy [GeV]", fontsize=FS_LABEL)

# ---- Col 2: Radial energy profile --------------------------------------
ax, axr = make_panel(outer_gs[0, 2])
for r in reduced:
    c, n = r["color"], r["n"]
    edges, sr, sre = radial_energy_profile(r["s_d"], r["s_e"], n, R_MAX)
    _,     mr, mre = radial_energy_profile(r["m_d"], r["m_e"], n, R_MAX)
    sr *= 1e3; sre *= 1e3; mr *= 1e3; mre *= 1e3     # GeV -> MeV
    r_ctr = 0.5 * (edges[:-1] + edges[1:])
    curve(ax, None, sr, sre, c, SIM_LS, SIM_LW, edges=edges)
    curve(ax, None, mr, mre, c, ML_LS,  ML_LW,  edges=edges)
    ratio(axr, r_ctr, sr, mr, sre, mre, c, steps=True)
ax.set_yscale("log"); ax.set_xlim(0, R_MAX)
ax.set_ylabel("Mean Energy [MeV]", fontsize=FS_LABEL)
ax.set_title("Radial Energy Profile", fontsize=FS_TITLE, pad=10)
axr.set_ylim(0.5, 1.5); axr.set_yticks([0.5, 1.0, 1.5])
axr.set_xlabel("Radial Distance [m]", fontsize=FS_LABEL)

# ---- Col 3: Mean hit time per shower -----------------------------------
ax, axr = make_panel(outer_gs[0, 3])
if _has_t:
    for r in reduced:
        c = r["color"]
        sc, _ = np.histogram(r["s_mht"], bins=TBINS)
        mc, _ = np.histogram(r["m_mht"], bins=TBINS)
        curve(ax, None, sc.astype(float), np.sqrt(sc), c, SIM_LS, SIM_LW, edges=TBINS)
        curve(ax, None, mc.astype(float), np.sqrt(mc), c, ML_LS,  ML_LW,  edges=TBINS)
        ratio(axr, T_CTR, sc.astype(float), mc.astype(float),
              np.sqrt(sc), np.sqrt(mc), c)
    ax.set_ylabel("Number of Showers", fontsize=FS_LABEL)
    ax.set_title("Mean Hit Time per Shower", fontsize=FS_TITLE, pad=10)
    axr.set_ylim(0.5, 1.5); axr.set_yticks([0.5, 1.0, 1.5])
    axr.set_xlabel(r"Mean $t$ [$\mu$s]", fontsize=FS_LABEL)
else:
    ax.text(0.5, 0.5, "No time data", ha="center", va="center", color=MUTED)
    axr.set_visible(False)

# ---- Col 4: Radial time profile ----------------------------------------
ax, axr = make_panel(outer_gs[0, 4])
drew = False
for r in reduced:
    c = r["color"]
    edges_t, st, ste = radial_time_profile(r["s_d"], r["s_t"], R_MAX)
    _,       mt, mte = radial_time_profile(r["m_d"], r["m_t"], R_MAX)
    if edges_t is None or mt is None:
        continue
    st *= US; ste *= US; mt *= US; mte *= US
    rt_ctr = 0.5 * (edges_t[:-1] + edges_t[1:])
    curve(ax, None, st, ste, c, SIM_LS, SIM_LW, edges=edges_t)
    curve(ax, None, mt, mte, c, ML_LS,  ML_LW,  edges=edges_t)
    ratio(axr, rt_ctr, st, mt, ste, mte, c, steps=True)
    drew = True
if drew:
    ax.set_xlim(0, R_MAX)
    ax.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=FS_LABEL)
    ax.set_title("Radial Time Profile", fontsize=FS_TITLE, pad=10)
    axr.set_ylim(0.5, 1.5); axr.set_yticks([0.5, 1.0, 1.5])
    axr.set_xlabel("Radial Distance [m]", fontsize=FS_LABEL)
else:
    ax.text(0.5, 0.5, "No time data", ha="center", va="center", color=MUTED)
    axr.set_visible(False)


# ------------------------------------------------------------------ legend

handles  = [Line2D([], [], color=r["color"], lw=3.0,
                   label=f"{r['label']}  ({r['n']})") for r in reduced]
handles += [Line2D([], [], color=FG, lw=2.6, ls=SIM_LS, label="CORSIKA"),
            Line2D([], [], color=FG, lw=2.4, ls=ML_LS,  label="ML")]

leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.935),
                 ncol=len(handles), fontsize=FS_LEG, labelcolor=FG,
                 frameon=True, facecolor=BG, edgecolor=MUTED,
                 handlelength=2.6, columnspacing=2.2,
                 borderaxespad=0.0, borderpad=0.7)
leg.get_frame().set_linewidth(1.0)

plt.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
plt.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight",
            facecolor=fig.get_facecolor(), dpi=200)
print(f"Saved → {OUT}")
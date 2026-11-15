"""Shared figure functions lifted out of the malata/tau geometry notebooks.

Each function here is the exact plotting code that used to be inline in one
notebook cell (`malata_tau_geometry.ipynb` / `explore_trigger_counts.ipynb`),
parametrized by `figsize` and a few font sizes so the same function renders
both the notebook's on-screen version (defaults match what the cell used to
produce) and a paper-sized version (05_paper_figures.py passes a smaller
figsize + larger fontsize so the text reads correctly once LaTeX scales the
figure back down to its `\\includegraphics` width).

Every legend here is drawn ONCE, at the bottom of the figure via
`fig.legend(...)` (outside every axes' bbox), never per-axes: the plotted
artists still carry their `label=` kwargs (so `get_legend_handles_labels()`
can find them), but nothing calls `ax.legend()` — an in-axes legend sits on
top of the plotted data, which is the opposite of what a legend is for.
`savefig_paper`'s `bbox_inches="tight"` expands the saved canvas to include
the bottom legend rather than clipping it.

Setup/data-loading that a LATER, untouched cell in the same notebook also
depends on (e.g. `rod()` in section 9, `panel()` in explore_trigger_counts'
detector-patterns cell) is deliberately NOT moved in here — it stays inline in
the notebook and is passed to these functions as an argument, so nothing that
another cell needs disappears.

Does not call matplotlib.use(...) — the caller's backend (inline in a
notebook, Agg in a script) is left untouched.
"""
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def tidy_3d_axes(ax, fs_tick=None, fs_label=None, nbins=4):
    """Fewer, evenly-spaced tick numbers on all three 3D axes, with
    tick/label padding that scales with the figure's own font size instead
    of matplotlib's small fixed default. 3D tick labels crowd each other and
    the axis label next to them much faster than 2D ones do -- a fixed pad
    that looked fine at screen size stays that same fixed size once fonts
    are scaled 2-4x for print, so the (now much bigger) numbers run into
    whatever is next to them.

    `fs_tick`/`fs_label` default to the CURRENT rcParams so this still works
    inside a bare `paper_rc` rc_context: mplot3d's z axis doesn't reliably
    pick up the `xtick.labelsize`/`ytick.labelsize` rcParams the way the x/y
    axes do, so every axis is set explicitly here rather than left to
    inherit -- silently keeping z ticks at matplotlib's unscaled default is
    exactly the earlier "legend bigger than title" bug in a different spot.

    The z axis specifically needs much more labelpad than x/y: mplot3d
    offsets the z label by a fixed distance from the axis LINE, not from the
    tick numbers next to it, so at any nontrivial tick fontsize the numbers'
    own width eats into that fixed offset and the label collides with (or at
    print scale, prints on top of / off the edge of the canvas past) the
    tick numbers. x/y labels sit below their (horizontal) tick text and
    don't have this problem.
    """
    if fs_tick is None:
        fs_tick = matplotlib.rcParams.get("xtick.labelsize", 10)
        if not isinstance(fs_tick, (int, float)):
            fs_tick = matplotlib.rcParams["font.size"]
    if fs_label is None:
        fs_label = matplotlib.rcParams.get("axes.labelsize", 12)
        if not isinstance(fs_label, (int, float)):
            fs_label = matplotlib.rcParams["font.size"]
    pad = max(3.0, 0.5 * fs_tick)
    labelpad = max(4.0, 1.0 * fs_label) + 0.4 * fs_tick
    # z is the outlier: unlike x/y, pushing it out much further than this
    # doesn't separate it from the tick numbers -- matplotlib's 3D tight-bbox
    # calculation for the z axis stops tracking it and it gets cropped/lost
    # off the saved canvas entirely instead. Same value as x/y is the largest
    # that reliably stays visible.
    z_labelpad = labelpad
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=nbins))
    for a in ("x", "y", "z"):
        ax.tick_params(axis=a, labelsize=fs_tick, pad=pad)
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = z_labelpad


def plot_mountain_3d(tris, tri_up, layouts, det_up, mE, mN, mU, *,
                     figsize=(13, 9), fs_title=14, fs_label=12, fs_tick=10,
                     fs_legend=10, view=(28, -125)):
    """malata_tau_geometry.ipynb section 1b: mountain surface + grid detectors.

    `tris`/`tri_up` are the detector-region triangles (ENU) and their mean Up,
    already rebuilt by the caller (h5py + `_ecef_to_enu`) — kept there rather
    than recomputed here because section 6 (surface normals) reuses the same
    `tris` array as a notebook global."""
    norm = matplotlib.colors.Normalize(tri_up.min(), tri_up.max())
    poly = Poly3DCollection(tris, facecolors=matplotlib.cm.terrain(norm(tri_up)),
                            edgecolors="k", linewidths=0.1, alpha=0.95)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(poly)
    Eg, Ng = layouts["grid"]
    ax.scatter(Eg, Ng, det_up(Ng, Eg) + 110, c="black", s=22, marker="^",
              depthshade=False, label="grid detectors")
    ax.set_xlim(mE.min(), mE.max()); ax.set_ylim(mN.min(), mN.max())
    ax.set_zlim(mU.min(), mU.max())
    ax.set_box_aspect((np.ptp(mE), np.ptp(mN), np.ptp(mU)))
    ax.set_xlabel("East [m]", fontsize=fs_label)
    ax.set_ylabel("North [m]", fontsize=fs_label)
    ax.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax, fs_tick, fs_label)
    ax.set_title("Malata mountain (3D)", fontsize=fs_title)
    ax.view_init(elev=view[0], azim=view[1])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="terrain"); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Up [m]", fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
              fontsize=fs_legend, ncol=1, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def plot_tau_arrival_directions(tau_wholesky_path, on_terr, tE_e, tN_n, tU_u,
                                mE, mN, mU, tri_up, add_terrain, *,
                                figsize=(30, 13), fs_title=14, fs_label=12,
                                fs_tick=10, fs_legend=9, r_lim=5000.0,
                                view=(22, -155), seed=0):
    """malata_tau_geometry.ipynb section 2b: tau travel-direction vector field.

    Self-contained (reloads `direction` from `tau_wholesky_path` itself, same
    as the source cell) — no other cell in the notebook depends on values this
    one used to leave behind: the two-panel variant right after it reloads its
    own copy of `tdir`/`dE`/`dN`/`dU`/`idx`/`q` independently."""
    with h5py.File(tau_wholesky_path, "r") as f:
        tdir = f["direction"][...]
    dE, dN, dU = tdir[0], tdir[1], tdir[2]

    idx = np.where(on_terr & (np.abs(tE_e) <= r_lim) & (np.abs(tN_n) <= r_lim))[0]
    qrng = np.random.default_rng(seed)
    q = qrng.choice(idx, size=min(500, idx.size), replace=False)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    add_terrain(ax1, alpha=0.45, r_lim=r_lim)
    ax1.quiver(tE_e[q], tN_n[q], tU_u[q], dE[q], dN[q], dU[q],
              length=300.0, normalize=True, linewidth=0.7, arrow_length_ratio=0.45)

    # Wireframe box spanning the mountain's E/N/U extent.
    x0, x1 = mE.min(), mE.max()
    y0, y1 = mN.min(), mN.max()
    z0, z1 = mU.min(), mU.max()
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
        (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
        (5, 7), (6, 7),
    ]
    for k, (i, j) in enumerate(edges):
        ax1.plot(*zip(corners[i], corners[j]), c="k", lw=0.8, alpha=0.6,
                label="region of interest" if k == 0 else None)

    ax1.set_xlabel("East [m]", fontsize=fs_label)
    ax1.set_ylabel("North [m]", fontsize=fs_label)
    ax1.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax1, fs_tick=fs_tick, fs_label=fs_label)
    ax1.set_title(f"Tau travel directions (n={q.size})", fontsize=fs_title)
    ax1.view_init(elev=view[0], azim=view[1])

    norm = matplotlib.colors.Normalize(tri_up.min(), tri_up.max())
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="terrain"); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.3, pad=0.08)
    cbar.set_label("Up-component", fontsize=fs_label)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center",
              fontsize=fs_legend, ncol=1, bbox_to_anchor=(0.5, -0.02))
    return fig


def plot_showers_overlaid(rod, add_terrain, tE_e, tN_n, tU_u, inband,
                          mE, mN, mU, LOC_R, *,
                          figsize=(15, 6.8), fs_title=13, fs_label=11,
                          fs_tick=9, fs_legend=8, n_pick=40, seed=0):
    """malata_tau_geometry.ipynb section 9: real showers over the local terrain.

    `rod(i)` returns a schematic rod of points along shower `i`'s axis (ENU) and
    `add_terrain` draws the local mesh patch — both kept as parameters (not
    redefined here) because section 9b reuses the exact same `rod`/`add_terrain`
    closures the notebook cell defines, so they must stay notebook globals
    there."""
    rng = np.random.default_rng(seed)
    sel = np.where(inband & (np.hypot(tE_e, tN_n) < LOC_R))[0]
    pick = rng.choice(sel, size=min(n_pick, sel.size), replace=False)

    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    add_terrain(ax0, alpha=0.55)
    for i in pick:
        P = rod(i); ax0.plot(P[:, 0], P[:, 1], P[:, 2], lw=0.9, alpha=0.8)
    ax0.scatter(tE_e[pick], tN_n[pick], tU_u[pick], c="red", s=18,
               depthshade=False, label="decay vertex")
    ax0.scatter(mE, mN, mU, c="k", s=4, depthshade=False, label="obs region (1.4 km)")
    ax0.set_xlabel("East [m]", fontsize=fs_label)
    ax0.set_ylabel("North [m]", fontsize=fs_label)
    ax0.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax0, fs_tick=fs_tick, fs_label=fs_label)
    ax0.set_box_aspect((1, 1, 0.42))
    ax0.set_title(f"{pick.size} real showers (±{LOC_R/1e3:.0f} km)", fontsize=fs_title)
    ax0.view_init(elev=24, azim=-125)

    ax1 = fig.add_subplot(1, 2, 2)
    for i in pick:
        P = rod(i); ax1.plot(P[:, 0] / 1e3, P[:, 1] / 1e3, lw=0.9, alpha=0.75)
    ax1.scatter(tE_e[pick] / 1e3, tN_n[pick] / 1e3, c="red", s=18, zorder=5,
               label="decay vertex")
    ax1.plot(np.array([mE.min(), mE.max(), mE.max(), mE.min(), mE.min()]) / 1e3,
            np.array([mN.min(), mN.min(), mN.max(), mN.max(), mN.min()]) / 1e3,
            "k-", lw=2.2, zorder=6, label="obs footprint (1.4 km)")
    ax1.set_xlabel("East [km]", fontsize=fs_label)
    ax1.set_ylabel("North [km]", fontsize=fs_label)
    ax1.set_aspect("equal")
    ax1.set_title("Map view", fontsize=fs_title)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=fs_legend,
              ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    return fig

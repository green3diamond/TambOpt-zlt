"""Kernel frame issue, visualised. The kernel matches detectors to shower points in GLOBAL coordinates:
a fine Gaussian in North-Up and coarse 500 m East slabs (place_clouds_enu). C8 records each shower on
observation planes perpendicular to its own axis, 500 m apart, so the two agree only for a horizontal
shower travelling due West. Three held-out showers: run 17 plane ~due West, run 17 plane ~40 deg off
West, and Malata (run 15) travelling ~North into its South-facing slope.
Also shows a placement bug: C8 layer l lies at depth (l + 1) x 500 m along the axis (layer 0 at 500 m,
checked on the training showers), but place_clouds_enu builds Up = decay_Up + l x 500 m x d_Up, one
layer short. Per shower:
  a. top view: placed points (energy), labelled axis, the kernel's East slabs (grey) vs the shower's own
     500 m layers (blue, across the axis);
  b. deposit at every mesh face with the kernel as used (global frame, Up one layer short);
  c. the same, global frame with Up from (l + 1) x 500 m;
  d. the same kernel in the shower frame: detectors projected onto the axis (depth s, transverse offsets
     t1 horizontal / t2 ~vertical); points at s = (l + 1) x 500 m, t1 from their x/y, t2 = 0 (C8 dropped
     it); adaptive widths found in that frame;
  e. lateral profile: deposit vs the faces' horizontal-transverse offset t1 from the axis.
Uncapped adaptive widths (k = 8, floor 10 m), the three species summed.

    python kernel_frame_issue.py [--cap 50]   (adaptive widths capped at --cap metres; default uncapped)
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import showerdata
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"   # branch run17-bwcap50 (run 17 constants)
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.data import place_clouds_enu  # noqa: E402
from modules.geometry import load_tr_mountain  # noqa: E402
from modules.showers.kernel import adaptive_bandwidths, GetCounts_planeaware  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6"
EAST_ENTRY, DX, K, SIG = 1500.0, 500.0, 8, 10.0
INK, GREY = "#1f2328", "#52514e"
_ap = argparse.ArgumentParser()
_ap.add_argument("--cap", type=float, default=float("inf"))
CAP = _ap.parse_args().cap
SUFFIX = "" if math.isinf(CAP) else f"_cap{CAP:g}"
WIDTHS = "uncapped adaptive widths" if math.isinf(CAP) else f"adaptive widths capped at {CAP:g} m"
GEOM = {"run17": (C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY),
        "run15": (os.path.join(REPO, "data/malata.h5"), "malata", "detector1")}
CORPUS = {"run17": f"{V6}/17_plane45_interp_hurdle_100k/v6_run_00/cashed_showers_tau_dual_heldout.pt",
          "run15": f"{V6}/15_sigma10_interp_hurdle_10pct/v6_run_00/cashed_showers_tau_dual_heldout.pt"}
# (run, label, azimuth window as (centre, half-width) [deg], max core-to-mesh-centre distance [m])
CASES = [("run17", "run 17 plane, shower ~due West", (180, 8), 250),
         ("run17", "run 17 plane, shower ~40° off West", (180 - 42, 8), 700),
         ("run15", "Malata (run 15), shower ~North into the South-facing slope", (90, 20), 300)]


def load_run(run):
    m = load_tr_mountain(*GEOM[run], east_entry=EAST_ENTRY, layer_east_dx=DX, n_planes=24)
    mesh = np.asarray(m.centroids_ENU, dtype=float)
    path = CORPUS[run]
    pos = torch.load(os.path.splitext(path)[0] + "_positions.pt").double().numpy()
    n_pairs = pos.shape[0] // 3
    ip = showerdata.load_inc_particles(path, 0, n_pairs)
    d = np.asarray(ip.directions, dtype=float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return dict(mesh=mesh, path=path, n_pairs=n_pairs, pos=pos[:n_pairs], dirs=d,
                logE=np.log10(np.asarray(ip.energies, dtype=float).ravel()))


def axis_hit(p, d, mesh):
    """Nearest mesh face to the axis (in front of the decay): (face index, its distance to the axis)."""
    r = mesh - p
    t = r @ d
    perp = np.linalg.norm(r - t[:, None] * d, axis=1)
    perp[t <= 0] = np.inf
    j = int(np.argmin(perp))
    return j, float(perp[j])


def pick(R, az_win, max_core):
    az = np.degrees(np.arctan2(R["dirs"][:, 1], R["dirs"][:, 0]))
    dz = np.abs((az - az_win[0] + 180) % 360 - 180)
    ze = np.degrees(np.arccos(R["dirs"][:, 2]))
    centre = R["mesh"][:, :2].mean(0)
    best, best_score = None, np.inf
    for i in np.nonzero((dz < az_win[1]) & (ze > 80) & (ze < 100))[0]:
        j, perp = axis_hit(R["pos"][i], R["dirs"][i], R["mesh"])
        if perp > 40 or np.hypot(*(R["mesh"][j, :2] - centre)) > max_core:
            continue
        score = abs(R["logE"][i] - 6.3)
        if score < best_score:
            best, best_score = (int(i), j), score
    return best


def counts(cloud, xd, yd, zd, bw):
    E, _ = GetCounts_planeaware(cloud, torch.as_tensor(xd).float(), torch.as_tensor(yd).float(),
                                torch.as_tensor(zd).float(), None, torch.zeros(1), None, sigma=SIG, bandwidth=bw)
    return E[0].numpy()


def shower(R, i):
    p, d = R["pos"][i], R["dirs"][i]
    e1 = np.array([-d[1], d[0], 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    mesh = R["mesh"]
    r = mesh - p
    s_d, t1_d, t2_d = r @ d, r @ e1, r @ e2
    out = {k: np.zeros(len(mesh)) for k in ("used", "fixed", "frame")}
    pts = []
    P0, D0 = torch.as_tensor(p[None]).float(), torch.as_tensor(d[None]).float()
    for s in range(3):
        raw = torch.as_tensor(showerdata.load(R["path"], start=s * R["n_pairs"] + i,
                                              stop=s * R["n_pairs"] + i + 1).points, dtype=torch.float32)
        real = raw[..., 3] > 0
        for key, shift in (("used", 0.0), ("fixed", 1.0)):        # Up from l x 500 m (as used) or (l + 1) x 500 m
            placed = raw.clone()
            placed[..., 2] = torch.where(real, placed[..., 2] + shift, placed[..., 2])
            place_clouds_enu(placed, P0, D0, east_entry=EAST_ENTRY, layer_east_dx=DX)
            bw = adaptive_bandwidths(placed, K, SIG, sigma_max=CAP)
            out[key] += counts(placed, mesh[:, 1], mesh[:, 2], (EAST_ENTRY - mesh[:, 0]) / DX, bw)
            if key == "fixed":
                k = placed[0, :, 3] > 0
                pts.append(np.stack([EAST_ENTRY - DX * placed[0, k, 2].numpy(), placed[0, k, 0].numpy(),
                                     placed[0, k, 3].numpy()], 1))
        sf = raw.clone()                                     # shower frame: [t1, t2 = 0, depth / 500 m, e, t]
        sf[..., 0] = raw[..., 0] * float(e1[0]) + raw[..., 1] * float(e1[1])
        sf[..., 1] = 0.0
        sf[..., 2] = torch.where(real, raw[..., 2] + 1.0, raw[..., 2])
        bw_s = adaptive_bandwidths(sf, K, SIG, sigma_max=CAP)
        out["frame"] += counts(sf, t1_d, t2_d, s_d / DX, bw_s)
    out.update(pts=np.concatenate(pts), t1=t1_d)
    return out


COLS = (("used", "kernel as used\n(global frame, Up one layer short)"),
        ("fixed", "global frame, Up fixed\n(depth (l+1) × 500 m)"),
        ("frame", "shower frame, depth fixed\n(same kernel, axis coordinates)"))
fig = plt.figure(figsize=(27, 16))
gs = fig.add_gridspec(3, 7, width_ratios=[1.25, 1, 1, 1, 0.05, 0.22, 1.15], wspace=0.12, hspace=0.38)
runs = {}
summary = {}
for row, (run, label, az_win, max_core) in enumerate(CASES):
    R = runs.setdefault(run, load_run(run))
    sel = pick(R, az_win, max_core) or pick(R, (az_win[0], az_win[1] * 1.5), max_core * 2)
    assert sel, f"no shower for {label}"
    i, j = sel
    S = shower(R, i)
    p, d, mesh = R["pos"][i], R["dirs"][i], R["mesh"]
    core = p + ((mesh[j] - p) @ d) * d
    az = math.degrees(math.atan2(d[1], d[0]))
    ze = math.degrees(math.acos(d[2]))
    summary[label] = dict(event=i, log10E=float(R["logE"][i]), zenith=ze, azimuth=az,
                          up_shift_m=float(DX * d[2]),
                          **{k: dict(total=float(S[k].sum()), faces_lit=int((S[k] > 1).sum())) for k, _ in COLS})
    print(f"[{label}] event {i}: log10E {R['logE'][i]:.2f} zenith {ze:.0f} azimuth {az:.0f}  Up error {DX * d[2]:+.0f} m  " +
          "  ".join(f"{k} {S[k].sum():.0f} ({(S[k] > 1).sum()} lit)" for k, _ in COLS), flush=True)

    # a. top view, around the mesh
    a = fig.add_subplot(gs[row, 0])
    span = np.ptp(mesh[:, :2], 0).max()
    lo = mesh[:, :2].min(0) - 0.35 * span
    hi = mesh[:, :2].max(0) + 0.35 * span
    P = S["pts"]
    inb = (P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])
    for xE in np.arange(EAST_ENTRY, lo[0] - DX, -DX):
        if lo[0] < xE < hi[0]:
            a.axvline(xE, color="#c9c7bf", lw=0.9, zorder=0)
    a.scatter(mesh[:, 0], mesh[:, 1], s=4, color="#e2e0d8", lw=0, zorder=1)
    a.scatter(P[inb, 0], P[inb, 1], s=4, c=P[inb, 2], cmap="YlOrRd", norm=LogNorm(), lw=0, alpha=0.8, zorder=2)
    L = np.linalg.norm(core - p) * 2
    a.plot([p[0], p[0] + L * d[0]], [p[1], p[1] + L * d[1]], color="#2a78d6", lw=1.2, zorder=3)
    h = np.array([-d[1], d[0]]) / np.hypot(d[0], d[1])
    for sl in np.arange(DX, L, DX):
        c0 = p[:2] + sl * d[:2]
        a.plot(*np.stack([c0 - 0.12 * span * h, c0 + 0.12 * span * h]).T, color="#2a78d6", lw=1, zorder=3)
    a.plot(*core[:2], marker="*", color="#2a78d6", mec="white", ms=14, zorder=4)
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_aspect("equal")
    a.set_xlabel("East [m]"); a.set_ylabel("North [m]")
    a.set_title(label, fontsize=11, loc="left", color=INK, fontweight="bold")
    a.text(0.02, 0.02, f"event {i}: log10 E/GeV {R['logE'][i]:.2f}\nzenith {ze:.0f}°, heading azimuth {az:.0f}°\n"
           f"Up error from the layer bug: {DX * d[2]:+.0f} m", transform=a.transAxes, fontsize=8.5, color=INK,
           va="bottom", bbox=dict(fc="white", ec="#d8d6cf", alpha=0.9))
    if row == 0:
        a.text(0.0, 1.13, "grey: kernel's 500 m East slabs   blue: shower axis and its own 500 m layers   ★ core",
               transform=a.transAxes, fontsize=9, color=GREY)

    # b-d. footprints
    vmax = max(max(S[k].max() for k, _ in COLS), 10)
    norm = LogNorm(vmin=1, vmax=vmax)
    for c, (key, ttl) in enumerate(COLS):
        a = fig.add_subplot(gs[row, 1 + c])
        v = S[key]
        lit = v > 1
        a.scatter(mesh[~lit, 0], mesh[~lit, 1], s=7, color="#e2e0d8", lw=0)
        sc = a.scatter(mesh[lit, 0], mesh[lit, 1], s=12, c=v[lit], cmap="YlOrRd", norm=norm, lw=0)
        a.plot(*core[:2], marker="*", color="#2a78d6", mec="white", ms=14)
        a.set_aspect("equal"); a.set_yticks([]); a.set_xlabel("East [m]")
        a.text(0.03, 0.97, f"{lit.sum()} / {len(mesh)} faces ≥ 1\ntotal {v.sum():.3g}", transform=a.transAxes,
               fontsize=8.5, va="top", color=INK, bbox=dict(fc="white", ec="#d8d6cf", alpha=0.9))
        if row == 0:
            a.set_title(ttl, fontsize=10.5, color=INK)
    fig.colorbar(sc, cax=fig.add_subplot(gs[row, 4]), label="deposit per face [counts]")

    # e. lateral profile
    a = fig.add_subplot(gs[row, 6])
    w = 1.15 * np.abs(S["t1"]).max()
    bins = np.linspace(-w, w, 61)
    for key, col, lab in (("used", "#b9b7af", "as used"), ("fixed", "#eb6834", "global, Up fixed"),
                          ("frame", "#2a78d6", "shower frame")):
        hsum = np.histogram(S["t1"], bins=bins, weights=S[key])[0]
        a.step(0.5 * (bins[1:] + bins[:-1]), np.maximum(hsum, 0.1), where="mid", color=col, lw=1.5, label=lab)
    a.set_yscale("log"); a.set_ylim(bottom=0.5)
    a.axvline(0, color=GREY, lw=0.6)
    a.set_xlabel("face offset across the axis, horizontal t1 [m]")
    a.set_ylabel("deposit per bin [counts]")
    if row == 0:
        a.set_title("lateral profile across the axis", fontsize=10.5, color=INK)
    a.legend(fontsize=8)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

fig.suptitle("Kernel frame and layer placement. The kernel matches in global coordinates (fine North-Up Gaussian, 500 m East slabs); "
             "C8 records each shower on planes perpendicular to its own axis,\nso the two agree only for horizontal showers heading due "
             "West. Separately, C8 layer l sits at depth (l + 1) × 500 m, but placement uses l × 500 m for Up. "
             f"{WIDTHS[0].upper() + WIDTHS[1:]}, e + μ + γ.", fontsize=13, color=INK, y=0.995)
out = os.path.join(D, f"kernel_frame_issue{SUFFIX}.png")
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
json.dump(summary, open(os.path.join(D, f"kernel_frame_issue{SUFFIX}.json"), "w"), indent=1)
print(f"[write] {out}")

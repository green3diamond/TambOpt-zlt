"""Run 20 (Malata): where the shower axes hit the detector region. Every event of the run 20 corpus (run 15's Step-0
corpus with the >= 30 deg B-field cut of scripts/20_filter_corpus.py, training + held-out) is a ray from its decay
vertex along its direction; its core is the first of the 266 detector-region faces it crosses (Moller-Trumbore).
Two maps over the faces (East-North): cores per km^2 and the mean primary energy of those cores.
--src-run reads the corpus of another run (e.g. run 15's, before the copy is finished) and applies the same cut here
(a no-op on the filtered corpus). Writes <run 20>/diagnostics/core_heatmap.png and prints the coverage numbers.

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_core_heatmap.py [--src-run <run dir>]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import showerdata  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src-run", default=None, help="read <run>/v6_run_00 instead of this run's corpus")
ap.add_argument("--min-angle-b", type=float, default=30.0)
args = ap.parse_args()
N_SP = len(C.SPECIES_NAMES)
OUT = os.path.join(C.RUN_LOCATION, "diagnostics", "core_heatmap.png")
CORPUS = (os.path.join(args.src_run, "v6_run_00", os.path.basename(C.DUAL_SHOWER_CACHE_PATH)) if args.src_run
          else C.DUAL_SHOWER_CACHE_PATH)


def events(corpus):
    """Event-level (vertex (N, 3) ENU, unit direction (N, 3), energy (N,)) from species block 0 of a corpus."""
    inc = showerdata.load_inc_particles(corpus)
    per = len(inc.energies) // N_SP
    d = np.asarray(inc.directions[:per], float)
    pos = torch.load(os.path.splitext(corpus)[0] + "_positions.pt")[:per].numpy().astype(float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    keep = np.degrees(np.arccos(np.clip(np.abs(d[:, 0]), 0, 1))) >= args.min_angle_b          # the B-field cut
    return pos[keep], d[keep], np.asarray(inc.energies[:per], float).ravel()[keep]


verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)  # (F, 3, 3) ENU
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)          # m^2

parts = [events(CORPUS), events(os.path.splitext(CORPUS)[0] + "_heldout.pt")]
P, Dr, E = (np.concatenate(x) for x in zip(*parts))
print(f"{len(E)} events ({len(parts[0][2])} training + {len(parts[1][2])} held-out)")

# Moller-Trumbore, all faces per event chunk; first hit (smallest t > 0)
e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
first = np.full(len(E), -1)
for lo in range(0, len(E), 20000):
    o, d = P[lo:lo + 20000, None, :], Dr[lo:lo + 20000, None, :]
    pv = np.cross(d, e2[None])
    det = (e1[None] * pv).sum(-1)
    inv = 1.0 / np.where(np.abs(det) < 1e-9, np.nan, det)
    tv = o - tri[None, :, 0]
    u = (tv * pv).sum(-1) * inv
    qv = np.cross(tv, e1[None])
    v = (d * qv).sum(-1) * inv
    t = (e2[None] * qv).sum(-1) * inv
    hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    t = np.where(hit, t, np.inf)
    j = t.argmin(1)
    first[lo:lo + 20000] = np.where(np.isfinite(t[np.arange(len(j)), j]), j, -1)
ok = first >= 0
n = np.bincount(first[ok], minlength=len(tri)).astype(float)
esum = np.bincount(first[ok], weights=E[ok], minlength=len(tri))
dens = n / (area / 1e6)                                                       # cores per km^2
mean_e = np.where(n > 0, esum / np.maximum(n, 1), np.nan) / 1e6               # PeV
print(f"axes hitting the region: {ok.sum()} ({100 * ok.mean():.1f}%); faces hit: {(n > 0).sum()}/{len(tri)}")
print(f"cores per km^2: p10 {np.percentile(dens, 10):.0f}, median {np.median(dens):.0f}, p90 {np.percentile(dens, 90):.0f}, "
      f"max/median {dens.max() / np.median(dens):.1f}; mean E of all hits {E[ok].mean() / 1e6:.2f} PeV, "
      f"per face p10-p90 {np.nanpercentile(mean_e, 10):.2f}-{np.nanpercentile(mean_e, 90):.2f} PeV")

poly = tri[:, :, :2]
pad = 60
xl, yl = (poly[..., 0].min() - pad, poly[..., 0].max() + pad), (poly[..., 1].min() - pad, poly[..., 1].max() + pad)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 7.2), constrained_layout=True)
for ax, val, cmap, lab, title in (
        (axs[0], dens, "Blues", "shower cores per km²", f"Shower cores ({ok.sum():,} of {len(E):,} axes hit the region)"),
        (axs[1], mean_e, "Oranges", "mean primary energy [PeV]", "Mean energy of the showers hitting each face")):
    pc = PolyCollection(poly, array=val, cmap=cmap, edgecolors="white", linewidths=0.4)
    if np.isnan(val).any():
        pc.set_array(np.ma.masked_invalid(val)); pc.cmap.set_bad("#e3e1da")
    ax.add_collection(pc)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(lab); cb.outline.set_visible(False)
fig.suptitle("Run 20, Malata detector region (266 faces): run 15 corpus after the ≥ 30° B-field cut, "
             "training + held-out", fontsize=12)
fig.savefig(OUT, dpi=200, facecolor="white")
print("[write]", OUT)

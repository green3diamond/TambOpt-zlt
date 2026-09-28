"""Where do tau shower axes hit the detector mesh?

Intersects every tau's (decay vertex, direction) ray from the tau_wholesky h5
with the detector-region faces of the mountain mesh, and reports the hit
density per unit area. The question it answers: is the mesh illuminated evenly
(so no region is preferred and a uniform layout is the physical optimum of a
coverage-type objective), or are impacts concentrated (so a SWGO-like dense
core + sparse halo has a physical reason to appear)?

Also reports decay->mesh distance and the incidence angle to the local face
normal (footprint elongation ~ 1/|cos|).

Usage:
    python scripts/diagnostics/axis_impact_density.py [--h5 PATH] [--out DIR]
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, TAU_WHOLESKY_PATH,
    LOG_E_MIN, LOG_E_MAX, EDA_OUTPUTS_DIR,
)
from modules.geometry.mountain import (  # noqa: E402
    _read_detector_region, _resolve_enu_origin, _ecef_to_enu,
)


def mesh_triangles_enu():
    """(n_tri, 3 verts, 3 ENU) triangles of the detector region."""
    verts, faces, det_idx, h5_loc = _read_detector_region(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY)
    lon, lat = _resolve_enu_origin(None, None, h5_loc)
    tri_ecef = verts[:, faces[:, det_idx]]                    # (3 xyz, 3 verts, n_tri)
    tri = np.stack([_ecef_to_enu(tri_ecef[:, k, :], lon, lat) for k in range(3)], axis=0)
    return np.transpose(tri, (2, 0, 1))                       # (n_tri, 3 verts, 3 ENU)


def first_hits(orig, dirs, tri, chunk=20000, eps=1e-9):
    """Moller-Trumbore, nearest forward hit per ray. Returns (face, t); face=-1 if none."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]   # (T, 3)
    face = np.full(len(orig), -1, dtype=np.int64)
    tbest = np.full(len(orig), np.inf)
    for s in range(0, len(orig), chunk):
        o, d = orig[s:s + chunk, None, :], dirs[s:s + chunk, None, :]       # (C, 1, 3)
        p = np.cross(d, e2[None])                                          # (C, T, 3)
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[None]
        u = (tv * p).sum(-1) * inv
        q = np.cross(tv, e1[None])
        v = (d * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        t = np.where(hit, t, np.inf)
        j = t.argmin(1)
        tj = t[np.arange(len(j)), j]
        good = np.isfinite(tj)
        face[s:s + chunk][good] = j[good]
        tbest[s:s + chunk][good] = tj[good]
    return face, tbest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=TAU_WHOLESKY_PATH)
    ap.add_argument("--out", default=os.path.join(EDA_OUTPUTS_DIR, "axis_impact_density"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        pos = np.asarray(f["position"][...], dtype=np.float64).T   # (N, 3) E,N,U
        dirs = np.asarray(f["direction"][...], dtype=np.float64).T
        energy = np.asarray(f["energy"][...], dtype=np.float64)
    keep = (energy >= 10 ** LOG_E_MIN) & (energy <= 10 ** LOG_E_MAX)
    pos, dirs, energy = pos[keep], dirs[keep], energy[keep]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    print(f"[load] {args.h5}: {keep.sum()} of {len(keep)} events in the "
          f"1e{LOG_E_MIN:g}-1e{LOG_E_MAX:g} GeV band")

    tri = mesh_triangles_enu()
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= (2 * area)[:, None]
    cen = tri.mean(1)
    print(f"[mesh] {len(tri)} faces, total area {area.sum() / 1e6:.3f} km^2, "
          f"face area p10/p50/p90 = {np.percentile(area, [10, 50, 90]).round(0)} m^2")

    face, t = first_hits(pos, dirs, tri)
    hit = face >= 0
    print(f"[hits] {hit.mean():.2%} of rays hit the detector region")
    face, t, d, E = face[hit], t[hit], dirs[hit], energy[hit]
    hp = pos[hit] + t[:, None] * d

    counts = np.bincount(face, minlength=len(tri)).astype(float)
    dens = counts / area
    dens_n = dens / (counts.sum() / area.sum())            # 1.0 = perfectly uniform
    exp_cv = np.sqrt(np.mean(1.0 / np.maximum(counts, 1)))  # Poisson-only CV
    cv = dens_n.std()
    print(f"[density] per-face density / uniform: p5={np.percentile(dens_n, 5):.2f} "
          f"p25={np.percentile(dens_n, 25):.2f} p50={np.median(dens_n):.2f} "
          f"p75={np.percentile(dens_n, 75):.2f} p95={np.percentile(dens_n, 95):.2f}")
    print(f"[density] CV={cv:.3f} (Poisson-only expectation {exp_cv:.3f}); "
          f"max/min={dens_n.max() / max(dens_n.min(), 1e-9):.1f}")

    # Radial profile from the area-weighted mesh centre, in the mesh's own plane.
    c0 = (cen * area[:, None]).sum(0) / area.sum()
    rf = np.linalg.norm(cen - c0, axis=1)
    edges = np.quantile(rf, np.linspace(0, 1, 6))
    print("[radial] area-quintiles of distance from mesh centre -> density/uniform")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rf >= a) & (rf <= b)
        print(f"    r {a:6.0f}-{b:6.0f} m : {counts[m].sum() / area[m].sum() / (counts.sum() / area.sum()):.2f}")

    # Split by energy: does the high-E subset illuminate differently?
    le = np.log10(E)
    for lo, hi in [(5, 5.5), (5.5, 6), (6, 6.5), (6.5, 7.01)]:
        m = (le >= lo) & (le < hi)
        if m.sum() < 100:
            continue
        c = np.bincount(face[m], minlength=len(tri)) / area
        c = c / (m.sum() / area.sum())
        inner = rf <= np.median(rf)
        print(f"[energy] logE {lo}-{hi}: n={m.sum():7d}  CV={c.std():.3f}  "
              f"inner-half/outer-half density = {c[inner].mean() / c[~inner].mean():.2f}")

    cosi = np.abs((d * nrm[face]).sum(1))
    print(f"[geom] decay->mesh distance p10/p50/p90 = {np.percentile(t, [10, 50, 90]).round(0)} m")
    print(f"[geom] |cos incidence| p10/p50/p90 = {np.percentile(cosi, [10, 50, 90]).round(3)}  "
          f"-> footprint elongation 1/|cos| p50 = {np.median(1 / np.maximum(cosi, 1e-3)):.1f}x")

    np.savez(os.path.join(args.out, "axis_impact_density.npz"),
             tri=tri, area=area, counts=counts, hit_points=hp, t=t, cos_inc=cosi, logE=le)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    pc = PolyCollection(tri[:, :, :2], array=dens_n, cmap="viridis", edgecolor="none")
    ax[0, 0].add_collection(pc); ax[0, 0].autoscale(); ax[0, 0].set_aspect("equal")
    fig.colorbar(pc, ax=ax[0, 0], label="hit density / uniform")
    ax[0, 0].set_title("Axis impact density per face (East-North)")
    ax[0, 0].set_xlabel("East [m]"); ax[0, 0].set_ylabel("North [m]")
    ax[0, 1].hist(dens_n, bins=50); ax[0, 1].axvline(1, color="k", ls="--")
    ax[0, 1].set_title(f"Per-face density / uniform (CV={cv:.3f}, Poisson {exp_cv:.3f})")
    ax[1, 0].hist(t, bins=100); ax[1, 0].set_xlabel("decay -> mesh distance [m]")
    ax[1, 1].hist(cosi, bins=100); ax[1, 1].set_xlabel("|cos(axis, face normal)|")
    fig.tight_layout()
    png = os.path.join(args.out, "axis_impact_density.png")
    fig.savefig(png, dpi=120)
    print(f"[write] {png}")


if __name__ == "__main__":
    main()

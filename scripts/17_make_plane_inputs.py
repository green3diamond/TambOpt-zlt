"""Run 17 inputs: a synthetic 45-degree plane (geometry h5) and tau primaries aimed at it.

Geometry (site-local ENU, metres, origin = Malata's `location` so every frame matches):
    the plane spans North in [-1500, 1500] and 3000 m down its dip, rising 45 deg toward
    the West: East in [-2121, 0], Up = -East. It faces East (normal ~ (1, 0, 1)/sqrt2), the
    orientation the kernel's East-depth mapping assumes (z_cont = (EAST_ENTRY - E) / dx).
    Written in the malata h5 schema (ECEF vertices, 1-indexed faces, `detector1` = every
    face, `location`), so `load_tr_mountain` reads it unchanged.

Primaries (tau-loader schema, modules/showers/tau.py: position/direction (3,N), energy, pdg):
    one 6-D Latin hypercube per event:
        decay point in the cube East [500, 3500], North [-1500, 1500], Up [0, 3000] m
        target on the plane: along strike [0, 3000] m, down dip [0, 3000] m
        log10(E / GeV) in [5, 7]
    direction = unit(target - decay). Rejected: decay-target distance < 1 km, and zenith
    outside [60, 120] deg (the range of the production tau corpus the shower model has
    been run on). Oversampled, then the first --n-events kept.

    python scripts/17_make_plane_inputs.py --out-dir <run17>/inputs [--n-events 105300]
"""
import argparse
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

SIDE = 3000.0                       # plane side along strike and along dip [m]
TILT = math.radians(45.0)
D_E = SIDE * math.cos(TILT)         # horizontal (East) extent of the plane, 2121 m
CUBE = dict(E=(500.0, 3500.0), N=(-1500.0, 1500.0), U=(0.0, 3000.0))
MALATA = "/n/home05/zdimitrov/tambo/TambOpt/data/malata.h5"


def plane_point(s, d):
    """Along-strike s and down-dip d in [0, SIDE] -> ENU (East, North, Up)."""
    east = -D_E + d * math.cos(TILT)
    north = s - SIDE / 2
    up = -east * math.tan(TILT)
    return np.stack([east, north, up], axis=-1)


def enu_to_ecef(enu, lon_deg, lat_deg):
    """Exact inverse of modules/geometry/mountain.py `_ecef_to_enu` (sphere, R = 6 371 km)."""
    lon0, lat0, R_e = math.radians(lon_deg), math.radians(lat_deg), 6_371_000.0
    s = R_e * np.array([math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)])
    R = np.array([
        [-math.sin(lon0), math.cos(lon0), 0.0],
        [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)],
        [math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)],
    ])
    return R.T @ enu + s[:, None]              # (3, n)


def write_geometry(path, n_grid):
    with h5py.File(MALATA, "r") as f:
        loc = f["malata/location"][...]
        radii = f["malata/radii"][...]
    g = np.linspace(0.0, SIDE, n_grid)
    S, Dd = np.meshgrid(g, g, indexing="ij")
    verts_enu = plane_point(S.ravel(), Dd.ravel())               # (n_grid^2, 3)
    faces = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            a, b, c, d = i * n_grid + j, (i + 1) * n_grid + j, (i + 1) * n_grid + j + 1, i * n_grid + j + 1
            faces += [(a, b, c), (a, c, d)]
    faces = np.asarray(faces, dtype=np.int64).T + 1               # (3, Nf), 1-indexed (Julia)
    verts_ecef = enu_to_ecef(verts_enu.T, float(loc[0]), float(loc[1]))
    with h5py.File(path, "w") as f:
        grp = f.create_group("plane45")
        grp["vertices"] = verts_ecef
        grp["faces"] = faces
        grp["detector1"] = np.arange(1, faces.shape[1] + 1, dtype=np.int64)
        grp["location"] = loc
        grp["radii"] = radii
    print(f"[geometry] {path}: {verts_enu.shape[0]} vertices, {faces.shape[1]} faces, location {loc}")
    return verts_enu


def _cube(u):
    return np.stack([CUBE["E"][0] + u[:, 0] * (CUBE["E"][1] - CUBE["E"][0]),
                     CUBE["N"][0] + u[:, 1] * (CUBE["N"][1] - CUBE["N"][0]),
                     CUBE["U"][0] + u[:, 2] * (CUBE["U"][1] - CUBE["U"][0])], axis=1)


def _valid(decay, target, min_dist, zen_lo, zen_hi):
    v = target - decay
    dist = np.linalg.norm(v, axis=1)
    zen = np.degrees(np.arccos(np.clip(v[:, 2] / dist, -1, 1)))
    return (dist >= min_dist) & (zen >= zen_lo) & (zen <= zen_hi)


def sample_primaries(n_events, seed, min_dist, zen_lo, zen_hi, max_redraw=200):
    """One 6-D LHS draw per event. Targets and energies are kept as drawn (uniform over the
    plane and in log E); a decay point that fails a cut is redrawn uniformly in the cube until
    it passes, so the cuts reshape the decay cloud but never the illumination of the plane."""
    u = qmc.LatinHypercube(d=6, seed=seed).random(n_events)
    decay = _cube(u[:, :3])
    target = plane_point(u[:, 3] * SIDE, u[:, 4] * SIDE)
    log_e = 5.0 + 2.0 * u[:, 5]
    rng = np.random.default_rng(seed)
    bad = ~_valid(decay, target, min_dist, zen_lo, zen_hi)
    first_pass = 1.0 - bad.mean()
    for _ in range(max_redraw):
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        decay[idx] = _cube(rng.random((len(idx), 3)))
        bad[idx] = ~_valid(decay[idx], target[idx], min_dist, zen_lo, zen_hi)
    if bad.any():                       # targets no decay in the cube can reach within the cuts
        keep = ~bad
        decay, target, log_e = decay[keep], target[keep], log_e[keep]
    v = target - decay
    dist = np.linalg.norm(v, axis=1)
    return decay, v / dist[:, None], log_e, target, dist, first_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-events", type=int, default=105_300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-grid", type=int, default=23, help="vertices per plane side")
    ap.add_argument("--min-dist", type=float, default=1000.0)
    ap.add_argument("--zenith", type=float, nargs=2, default=[60.0, 120.0])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    verts = write_geometry(os.path.join(args.out_dir, "plane45.h5"), args.n_grid)
    decay, direc, log_e, target, dist, acc = sample_primaries(args.n_events, args.seed, args.min_dist,
                                                               *args.zenith)
    # checks
    assert np.allclose(target[:, 0] + target[:, 2], 0.0, atol=1e-6), "targets off the plane"
    assert (dist >= args.min_dist).all() and np.allclose(np.linalg.norm(direc, axis=1), 1.0)
    assert (decay[:, 0] + decay[:, 2] > 0).all(), "decay behind the plane"
    with h5py.File(os.path.join(args.out_dir, "plane_taus.h5"), "w") as f:
        f["position"] = decay.T.astype(np.float64)
        f["direction"] = direc.T.astype(np.float64)
        f["energy"] = (10.0 ** log_e).astype(np.float64)
        f["pdg"] = np.full(len(log_e), 15, dtype=np.int64)
        f["target"] = target.T.astype(np.float64)        # extra, for plots; the loader ignores it
    zen = np.degrees(np.arccos(direc[:, 2]))
    az = np.degrees(np.arctan2(direc[:, 1], direc[:, 0])) % 360
    print(f"[primaries] {len(log_e)} events (LHS pairs passing cuts first time {acc:.1%}, rest had the decay redrawn); distance p5/p50/p95 "
          f"{np.percentile(dist, [5, 50, 95]).round(0)} m; zenith p5/p50/p95 {np.percentile(zen, [5, 50, 95]).round(1)}; "
          f"azimuth p5/p50/p95 {np.percentile(az, [5, 50, 95]).round(0)}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    sub = np.random.default_rng(0).choice(len(decay), 4000, replace=False)
    ax[0, 0].scatter(verts[:, 0], verts[:, 1], s=2, color="#b9b7af", label="plane vertices")
    ax[0, 0].scatter(decay[sub, 0], decay[sub, 1], s=2, color="#2a78d6", alpha=0.4, label="decays")
    ax[0, 0].scatter(target[sub, 0], target[sub, 1], s=2, color="#eb6834", alpha=0.4, label="targets")
    ax[0, 0].set(xlabel="East [m]", ylabel="North [m]", title="top view"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].scatter(verts[:, 0], verts[:, 2], s=2, color="#b9b7af")
    ax[0, 1].scatter(decay[sub, 0], decay[sub, 2], s=2, color="#2a78d6", alpha=0.4)
    ax[0, 1].scatter(target[sub, 0], target[sub, 2], s=2, color="#eb6834", alpha=0.4)
    ax[0, 1].set(xlabel="East [m]", ylabel="Up [m]", title="side view (plane at 45°)")
    ax[0, 2].hist2d(target[:, 1], -target[:, 0] / math.cos(TILT), bins=40, cmap="viridis")
    ax[0, 2].set(xlabel="along strike (North) [m]", ylabel="down dip from bottom edge [m]",
                 title="targets on the plane (density)")
    ax[1, 0].hist(dist, bins=60, color="#2a78d6"); ax[1, 0].set(xlabel="decay-target distance [m]")
    ax[1, 1].hist(zen, bins=60, color="#2a78d6"); ax[1, 1].set(xlabel="zenith [deg]")
    ax[1, 2].hist(log_e, bins=40, color="#2a78d6"); ax[1, 2].set(xlabel="log10(E / GeV)")
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Run 17 primaries: {len(log_e)} events, LHS, min distance {args.min_dist:.0f} m, "
                 f"zenith {args.zenith[0]:.0f}-{args.zenith[1]:.0f} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plane_primaries.png"), dpi=110)
    print(f"[write] {args.out_dir}/plane45.h5, plane_taus.h5, plane_primaries.png")


if __name__ == "__main__":
    main()

"""Synthetic-plane inputs (runs 17, 18): a 45-degree plane (geometry h5) and tau primaries aimed at it.

Geometry (site-local ENU, metres, origin = Malata's `location` so every frame matches):
    a square plane of side --side, tilted 45 deg, facing --facing: f is its horizontal outward
    normal (east = (1, 0), south = (0, -1), north = (0, 1), west = (-1, 0)) and g = (-f_N, f_E)
    its strike. A point at strike s and down-dip d (both in [0, side]) is
        horizontal = (s - side/2) g + (-D + d cos 45) f,   Up = -(horizontal . f) tan 45,
    with D = side cos 45: the top edge lies D behind the origin, the bottom edge at the origin.
    Run 17: --side 3000 --facing east (North in [-1500, 1500], East in [-2121, 0], Up = -East).
    Run 18: --side 6000 --facing south (East in [-3000, 3000], North in [0, 4243], Up = North),
    like the Malata slope, so the taus travel North into it.
    Written in the malata h5 schema (ECEF vertices, 1-indexed faces, `detector1` = every face,
    `location`) as <out-dir>/<group>.h5, so `load_tr_mountain` reads it unchanged.

Primaries (tau-loader schema, modules/showers/tau.py: position/direction (3,N), energy, pdg):
    one 6-D Latin hypercube per event:
        decay point in the box in front of the plane: along f [500, 500 + depth] (--box-depth, default side), along g
        +-side/2, Up [0, side] m
    Run 19: --side 12000 --box-depth 6000 --max-dist 11500: a 12 km deep box put half the decays farther than
    the 12 km (24 planes x 500 m) the C8 showers are recorded over; with the 6 km box 24% still were.
        target on the plane: along strike [0, side] m, down dip [0, side] m
        log10(E / GeV) in [5, 7]
    direction = unit(target - decay). Rejected (decay redrawn): decay-target distance < --min-dist,
    zenith outside --zenith, and (--min-angle-b) a direction within that 3-D angle of the +-x
    (East) axis, along which the C8 training showers' magnetic field points: showers within ~30 deg
    of it come out of C8 shifted sideways. Further LHS draws top the sample up to exactly --n-events.

    python scripts/17_make_plane_inputs.py --out-dir <run17>/inputs [--n-events 105300]
    python scripts/17_make_plane_inputs.py --out-dir <run18>/inputs --side 6000 --facing south \
        --n-events 210600 --zenith 60 100 --min-angle-b 30 --group plane45s
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

TILT = math.radians(45.0)
MALATA = "/n/home05/zdimitrov/tambo/TambOpt/data/malata.h5"
FACINGS = {"east": (1.0, 0.0), "south": (0.0, -1.0), "north": (0.0, 1.0), "west": (-1.0, 0.0)}


class Plane:
    """The tilted square plane and the decay box in front of it."""

    def __init__(self, side, facing, box_depth=None):
        self.side = float(side)
        self.depth = float(box_depth) if box_depth else self.side   # decay box extent along f (run 19: 6 km)
        self.f = np.array(FACINGS[facing])                      # horizontal outward normal (E, N)
        self.g = np.array([-self.f[1], self.f[0]])              # strike (E, N)
        self.D = self.side * math.cos(TILT)                     # horizontal extent of the dip

    def point(self, s, d):
        """Along-strike s and down-dip d in [0, side] -> ENU (East, North, Up)."""
        s, d = np.asarray(s, float), np.asarray(d, float)
        h = (s - self.side / 2)[..., None] * self.g + (-self.D + d * math.cos(TILT))[..., None] * self.f
        up = -(h @ self.f) * math.tan(TILT)
        return np.concatenate([h, up[..., None]], axis=-1)

    def box(self, u):
        """(n, 3) uniform u -> decay points: along f [500, 500 + depth], along g +-side/2, Up [0, side]."""
        a = 500.0 + u[:, 0] * self.depth
        b = (u[:, 1] - 0.5) * self.side
        h = a[:, None] * self.f + b[:, None] * self.g
        return np.concatenate([h, (u[:, 2] * self.side)[:, None]], axis=1)

    def height_above(self, q):
        """Signed distance-like test: q . f + Up (0 on the plane, > 0 in front of it)."""
        return q[:, :2] @ self.f + q[:, 2]


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


def write_geometry(path, n_grid, plane, group):
    with h5py.File(MALATA, "r") as f:
        loc = f["malata/location"][...]
        radii = f["malata/radii"][...]
    g = np.linspace(0.0, plane.side, n_grid)
    S, Dd = np.meshgrid(g, g, indexing="ij")
    verts_enu = plane.point(S.ravel(), Dd.ravel())               # (n_grid^2, 3)
    faces = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            a, b, c, d = i * n_grid + j, (i + 1) * n_grid + j, (i + 1) * n_grid + j + 1, i * n_grid + j + 1
            faces += [(a, b, c), (a, c, d)]
    faces = np.asarray(faces, dtype=np.int64).T + 1               # (3, Nf), 1-indexed (Julia)
    verts_ecef = enu_to_ecef(verts_enu.T, float(loc[0]), float(loc[1]))
    with h5py.File(path, "w") as f:
        grp = f.create_group(group)
        grp["vertices"] = verts_ecef
        grp["faces"] = faces
        grp["detector1"] = np.arange(1, faces.shape[1] + 1, dtype=np.int64)
        grp["location"] = loc
        grp["radii"] = radii
    print(f"[geometry] {path}: {verts_enu.shape[0]} vertices, {faces.shape[1]} faces, location {loc}")
    return verts_enu


def _valid(decay, target, min_dist, zen_lo, zen_hi, min_angle_b, max_dist=np.inf):
    v = target - decay
    dist = np.linalg.norm(v, axis=1)
    u = v / np.maximum(dist, 1e-9)[:, None]
    zen = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
    ang_b = np.degrees(np.arccos(np.clip(np.abs(u[:, 0]), 0, 1)))     # 3-D angle to the +-x axis
    return (dist >= min_dist) & (dist <= max_dist) & (zen >= zen_lo) & (zen <= zen_hi) & (ang_b >= min_angle_b)


def _draw(plane, n, seed, cuts, max_redraw=200):
    """One 6-D LHS draw of n events. Targets and energies are kept as drawn (uniform over the
    plane and in log E); a decay point that fails a cut is redrawn uniformly in the box until it
    passes, so the cuts reshape the decay cloud but never the illumination of the plane.
    Events whose target no decay in the box can reach within the cuts are dropped."""
    u = qmc.LatinHypercube(d=6, seed=seed).random(n)
    decay = plane.box(u[:, :3])
    target = plane.point(u[:, 3] * plane.side, u[:, 4] * plane.side)
    log_e = 5.0 + 2.0 * u[:, 5]
    rng = np.random.default_rng(seed)
    bad = ~_valid(decay, target, *cuts)
    first_pass = 1.0 - bad.mean()
    for _ in range(max_redraw):
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        decay[idx] = plane.box(rng.random((len(idx), 3)))
        bad[idx] = ~_valid(decay[idx], target[idx], *cuts)
    keep = ~bad
    return decay[keep], target[keep], log_e[keep], first_pass


def sample_primaries(plane, n_events, seed, cuts):
    """Exactly n_events: the main LHS draw, topped up by further LHS draws (seed + k) when the
    cuts drop events whose target cannot be reached."""
    decay, target, log_e, first_pass = _draw(plane, n_events, seed, cuts)
    k = 0
    while len(log_e) < n_events:
        k += 1
        need = n_events - len(log_e)
        d2, t2, l2, _ = _draw(plane, max(need * 2, 1000), seed + 1000 * k, cuts)
        decay, target, log_e = (np.concatenate([a, b[:need]]) for a, b in
                                ((decay, d2), (target, t2), (log_e, l2)))
        if k > 50:
            raise RuntimeError("could not reach --n-events within the cuts")
    decay, target, log_e = decay[:n_events], target[:n_events], log_e[:n_events]
    v = target - decay
    dist = np.linalg.norm(v, axis=1)
    return decay, v / dist[:, None], log_e, target, dist, first_pass, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-events", type=int, default=105_300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--side", type=float, default=3000.0, help="plane side along strike and dip [m]")
    ap.add_argument("--facing", choices=sorted(FACINGS), default="east")
    ap.add_argument("--group", default="plane45", help="h5 group = constants.GEOMETRY_GROUP; file <group>.h5")
    ap.add_argument("--n-grid", type=int, default=None,
                    help="vertices per plane side (default: ~136 m spacing, 23 at 3 km, 45 at 6 km)")
    ap.add_argument("--min-dist", type=float, default=1000.0)
    ap.add_argument("--zenith", type=float, nargs=2, default=[60.0, 120.0])
    ap.add_argument("--max-dist", type=float, default=float("inf"),
                    help="redraw decays farther than this from their target [m] (run 19: 11500; C8 covers 12 km)")
    ap.add_argument("--box-depth", type=float, default=None,
                    help="decay box extent in front of the plane [m] (default: --side). Run 19: 6000, as run 18, "
                         "so the decay-target distance stays within the 12 km the C8 showers cover")
    ap.add_argument("--min-angle-b", type=float, default=0.0,
                    help="reject directions within this 3-D angle [deg] of the +-x axis (C8 field)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plane = Plane(args.side, args.facing, args.box_depth)
    n_grid = args.n_grid or int(round(args.side / 136.4)) + 1
    cuts = (args.min_dist, args.zenith[0], args.zenith[1], args.min_angle_b, args.max_dist)

    verts = write_geometry(os.path.join(args.out_dir, f"{args.group}.h5"), n_grid, plane, args.group)
    decay, direc, log_e, target, dist, acc, n_top = sample_primaries(plane, args.n_events, args.seed, cuts)
    # checks
    assert len(log_e) == args.n_events
    assert np.allclose(plane.height_above(target), 0.0, atol=1e-6), "targets off the plane"
    assert (dist >= args.min_dist).all() and (dist <= args.max_dist).all() and np.allclose(np.linalg.norm(direc, axis=1), 1.0)
    assert (plane.height_above(decay) > 0).all(), "decay behind the plane"
    zen = np.degrees(np.arccos(direc[:, 2]))
    ang_b = np.degrees(np.arccos(np.abs(direc[:, 0])))
    assert (zen >= args.zenith[0] - 1e-9).all() and (zen <= args.zenith[1] + 1e-9).all()
    assert (ang_b >= args.min_angle_b - 1e-9).all()
    with h5py.File(os.path.join(args.out_dir, "plane_taus.h5"), "w") as f:
        f["position"] = decay.T.astype(np.float64)
        f["direction"] = direc.T.astype(np.float64)
        f["energy"] = (10.0 ** log_e).astype(np.float64)
        f["pdg"] = np.full(len(log_e), 15, dtype=np.int64)
        f["target"] = target.T.astype(np.float64)        # extra, for plots; the loader ignores it
    az = np.degrees(np.arctan2(direc[:, 1], direc[:, 0])) % 360
    print(f"[primaries] {len(log_e)} events (LHS pairs passing the cuts first time {acc:.1%}, rest had the "
          f"decay redrawn; {n_top} top-up draws); distance p5/p50/p95 {np.percentile(dist, [5, 50, 95]).round(0)} m; "
          f"zenith p5/p50/p95 {np.percentile(zen, [5, 50, 95]).round(1)}; azimuth p5/p50/p95 "
          f"{np.percentile(az, [5, 50, 95]).round(0)}; angle to the x axis min {ang_b.min():.1f} deg")

    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    sub = np.random.default_rng(0).choice(len(decay), min(4000, len(decay)), replace=False)
    ax[0, 0].scatter(verts[:, 0], verts[:, 1], s=2, color="#b9b7af", label="plane vertices")
    ax[0, 0].scatter(decay[sub, 0], decay[sub, 1], s=2, color="#2a78d6", alpha=0.4, label="decays")
    ax[0, 0].scatter(target[sub, 0], target[sub, 1], s=2, color="#eb6834", alpha=0.4, label="targets")
    ax[0, 0].set(xlabel="East [m]", ylabel="North [m]", title="top view", aspect="equal")
    ax[0, 0].legend(fontsize=8)
    along = lambda q: q[:, :2] @ plane.f                                 # noqa: E731
    ax[0, 1].scatter(along(verts), verts[:, 2], s=2, color="#b9b7af")
    ax[0, 1].scatter(along(decay[sub]), decay[sub, 2], s=2, color="#2a78d6", alpha=0.4)
    ax[0, 1].scatter(along(target[sub]), target[sub, 2], s=2, color="#eb6834", alpha=0.4)
    ax[0, 1].set(xlabel=f"along the plane's normal ({args.facing}) [m]", ylabel="Up [m]",
                 title="side view (plane at 45°)", aspect="equal")
    ax[0, 2].hist2d(target[:, :2] @ plane.g, (-along(target)) / math.cos(TILT), bins=40, cmap="viridis")
    ax[0, 2].set(xlabel="along strike [m]", ylabel="up the dip from the bottom edge [m]",
                 title="targets on the plane (density)")
    ax[0, 3].hist(az, bins=72, color="#2a78d6")
    ax[0, 3].set(xlabel="azimuth of travel [deg] (0 = East, 90 = North)")
    ax[1, 0].hist(dist, bins=60, color="#2a78d6"); ax[1, 0].set(xlabel="decay-target distance [m]")
    ax[1, 1].hist(zen, bins=60, color="#2a78d6"); ax[1, 1].set(xlabel="zenith of travel [deg]")
    ax[1, 2].hist(log_e, bins=40, color="#2a78d6"); ax[1, 2].set(xlabel="log10(E / GeV)")
    ax[1, 3].hist(ang_b, bins=60, color="#2a78d6")
    ax[1, 3].set(xlabel="3-D angle to the x axis (C8 field) [deg]")
    for a in ax.ravel():
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Plane primaries: {args.side / 1000:g} km plane facing {args.facing}, {len(log_e)} events, LHS, "
                 f"distance {args.min_dist:.0f}-{args.max_dist:.0f} m, zenith {args.zenith[0]:.0f}-{args.zenith[1]:.0f} deg, "
                 f">= {args.min_angle_b:g} deg from the x axis")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "plane_primaries.png"), dpi=110)
    print(f"[write] {args.out_dir}/{args.group}.h5, plane_taus.h5, plane_primaries.png")


if __name__ == "__main__":
    main()

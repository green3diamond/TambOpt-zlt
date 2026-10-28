"""The preprocessing chain as a forward model: particles in each plane's own frame -> the stored ground cells.

Mirrors hhanif/preprocess_showers.py per species: plane frame -> ground (x, y) (height dropped); per plane, the
particles beyond the 90th percentile of ground radius from the injection column are dropped (by count); particles
are summed into cell x cell ground cells (energy summed, earliest time kept, cell centre as position); the brightest
cells of the whole shower are kept up to the cap; cells are ordered by plane. The key overflow of the original
clustering is not reproduced, so the output matches the repaired data (unwrap).
"""
import numpy as np

from symbolic import atmosphere, lateral, sources

CELL = 10.0                  # m
RADIUS_PCT = 90.0            # per plane, by particle count


def to_ground(k, across, along, d):
    """Ground (x, y) of points at in-plane offsets (across, along) from the axis point of plane k (0-based)."""
    h, perp, c = lateral.frame(d)
    d = np.asarray(d, np.float64) / np.linalg.norm(d)
    return (k + 1)[:, None] * atmosphere.PLANE_SPACING * d[None, :2] + across[:, None] * perp + (along * c)[:, None] * h


def radius_cut(k, xy):
    """Keep mask: per plane, particles within the 90th percentile of ground radius from the injection column."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    keep = np.zeros(len(r), bool)
    for p in np.unique(k):
        m = k == p
        keep[m] = r[m] <= np.percentile(r[m], RADIUS_PCT)
    return keep


def cluster(k, xy, e, t, cell=CELL):
    """Per plane, sum particles into cell x cell ground cells -> (n, 5) [x, y, plane, energy, earliest time]."""
    idx = np.floor(xy.astype(np.float32) / np.float32(cell)).astype(np.int64)
    keys, inv = np.unique(np.column_stack([k, idx]), axis=0, return_inverse=True)
    inv = inv.ravel()
    energy = np.bincount(inv, e, len(keys))
    time = np.full(len(keys), np.inf)
    np.minimum.at(time, inv, t)
    centre = (keys[:, 1:] + 0.5) * cell
    return np.column_stack([centre, keys[:, 0], energy, time])


def cap(cells, n_max):
    """The n_max brightest cells of the shower, ordered by plane."""
    if len(cells) > n_max:
        cells = cells[np.argpartition(-cells[:, 3], n_max - 1)[:n_max]]
    return cells[np.argsort(cells[:, 2], kind="stable")]


def render(particles, d, n_max):
    """particles (n, 5) [plane k, across, along, energy, time] in each plane's frame about the axis -> stored cells."""
    k = particles[:, 0].astype(int)
    e = np.maximum(particles[:, 3], 0.0)
    xy = to_ground(k, particles[:, 1], particles[:, 2], d)
    keep = radius_cut(k, xy)
    return cap(cluster(k[keep], xy[keep], e[keep], particles[keep, 4]), n_max)


def render_species(particles, d, species):
    return render(particles, d, sources.CAP[species])

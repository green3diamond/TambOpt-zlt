"""Undo the cell-key overflow in the CORSIKA preprocessing (preprocess_showers.py, _cluster_shower_part).

The preprocessing keys a plane's cells as (ix - x_min) * stride + (iy - y_min) with stride = x range + 2, so a cell with
iy - y_min >= stride decodes q = floor((iy - y_min) / stride) cells East and q * stride cells South of where it was
(and merges with any cell already there). The plane's energy is unchanged; its positions are not.

Repair, per plane: a plane is moved when < NEAR_FRACTION of its energy lies within NEAR m of the axis point and one of
its TOP brightest cells is EAST_RANGE East and more than SOUTH_MIN South. Their shifts give q and stride up to a cell;
every stride in that window is tried, each cell then takes the q in 0..Q_MAX that puts it nearest the axis in the plane
(lateral.to_plane), and the stride with the most energy under a SHARP Gaussian about the axis wins (strides read off
neighbouring q move the core alike and differ by one cell East; ranking by the energy within NEAR first did worse for
every species). Exact for cells within stride / 2 cells of the axis; farther tail cells may take the wrong q. Planes
whose core did not move keep any wrapped tail cells (their stride is not observable). Merged cells stay merged.
Checked on clean planes corrupted with the preprocessing's keying: e/gamma restored; muons approximate (broad planes
whose brightest cell is often not the core: median plane restored, about a third take a wrong stride), which moves the
detector response by under 0.3%.
"""
import numpy as np

from symbolic import atmosphere, lateral

CELL = lateral.CELL
NEAR = 50.0                       # m, ground radius about the axis point
NEAR_FRACTION = 0.5
EAST_RANGE = (-15.0, 200.0)       # m; q = 1 moves 10 m East, and the brightest cell may sit a cell West of the axis
SOUTH_MIN = 100.0                 # m; below it South shifts and their mirror occur about equally often (muons)
Q_MAX = 40
STRIDE_SLACK = 15.0               # m, the core's own offset from the axis when reading the stride off its shift
TOP = 1                           # brightest cells read for the signature (5 raised the muon mirror null 0.25% -> 1.9%)
SHARP = 10.0                      # m, Gaussian weight about the axis that picks the stride (resolves one cell East)


def axis_point(k, direction):
    d = np.asarray(direction, np.float64)
    return (k + 1) * atmosphere.PLANE_SPACING * d[:2] / np.linalg.norm(d)


def near_share(xy, e, axis_xy):
    g = np.asarray(xy, np.float64) - axis_xy
    return e[np.hypot(g[:, 0], g[:, 1]) < NEAR].sum() / e.sum()


def _top(xy, e, axis_xy):
    """Ground offsets (East, North) of the TOP brightest cells from the axis point."""
    i = np.argsort(e)[::-1][:TOP]
    return np.asarray(xy, np.float64)[i] - axis_xy


def _signature(off, sign):
    east, north = sign * off[:, 0], sign * off[:, 1]
    return (EAST_RANGE[0] <= east) & (east <= EAST_RANGE[1]) & (north < -SOUTH_MIN)


def moved(xy, e, axis_xy):
    """True when one of the plane's brightest cells carries the overflow's signature and the core is off the axis."""
    if not len(e) or e.sum() <= 0:
        return False
    return bool(near_share(xy, e, axis_xy) < NEAR_FRACTION and _signature(_top(xy, e, axis_xy), 1).any())


def mirror(xy, e, axis_xy):
    """The same signature turned by 180 deg: no mechanism produces it (a null for the flag)."""
    if not len(e) or e.sum() <= 0:
        return False
    return bool(near_share(xy, e, axis_xy) < NEAR_FRACTION and _signature(_top(xy, e, axis_xy), -1).any())


def apply_stride(xy, axis_xy, direction, stride, q_max=Q_MAX):
    """Each cell moved back by the q in 0..q_max that puts it nearest the axis in the plane; (positions, q)."""
    h, perp, c = lateral.frame(direction)
    q = np.arange(q_max + 1)
    shift = np.stack([-q * CELL, q * stride * CELL], -1)                         # (Q, 2)
    cand = np.asarray(xy, np.float64)[None] + shift[:, None]                     # (Q, N, 2)
    ab = lateral.to_plane(cand - axis_xy, h, perp, c)
    best = np.argmin(np.hypot(ab[..., 0], ab[..., 1]), 0)
    return cand[best, np.arange(cand.shape[1])], best


def strides(xy, e, axis_xy):
    """Candidate strides from the shifts of the brightest cells that carry the signature."""
    out = set()
    for east, north in _top(xy, e, axis_xy)[_signature(_top(xy, e, axis_xy), 1)]:
        q0 = max(1, int(round(east / CELL)))
        for qc in range(max(1, q0 - 2), q0 + 3):
            lo = int(np.floor((-north - STRIDE_SLACK) / (CELL * qc)))
            hi = int(np.ceil((-north + STRIDE_SLACK) / (CELL * qc)))
            out.update(range(max(2, lo), hi + 1))
    return sorted(out)


def unwrap_plane(xy, e, axis_xy, direction):
    """(positions, stride): the plane repaired, stride 0 when it was not moved."""
    xy = np.asarray(xy, np.float64)
    if not moved(xy, e, axis_xy):
        return xy, 0
    best, best_s, best_score = xy, 0, centred(xy, e, axis_xy)
    for s in strides(xy, e, axis_xy):
        new, _ = apply_stride(xy, axis_xy, direction, s)
        sc = centred(new, e, axis_xy)
        if sc > best_score:
            best, best_s, best_score = new, s, sc
    return best, best_s


def centred(xy, e, axis_xy):
    """Energy under a SHARP Gaussian about the axis point."""
    g = np.asarray(xy, np.float64) - axis_xy
    return float(np.sum(e * np.exp(-0.5 * (g[:, 0] ** 2 + g[:, 1] ** 2) / SHARP ** 2)))


def unwrap_shower(points, direction, n_planes):
    """(P, 5) native cloud [x, y, plane, energy, time] -> (repaired copy, stride per plane, 0 = untouched)."""
    out = np.array(points, np.float64, copy=True)
    s_k = np.zeros(n_planes, np.int64)
    k_all = np.rint(out[:, 2]).astype(int)
    live = out[:, 3] > 0
    for k in range(n_planes):
        m = live & (k_all == k)
        if m.any():
            out[m, :2], s_k[k] = unwrap_plane(out[m, :2], out[m, 3], axis_point(k, direction), direction)
    return out, s_k


def restore(points, direction, s_k):
    """Re-apply stored strides (from unwrap_shower) without the search."""
    out = np.array(points, np.float64, copy=True)
    k_all = np.rint(out[:, 2]).astype(int)
    for k in np.nonzero(s_k)[0]:
        m = (out[:, 3] > 0) & (k_all == k)
        out[m, :2], _ = apply_stride(out[m, :2], axis_point(k, direction), direction, int(s_k[k]))
    return out


def encode(xy, e, stride, y_min=None):
    """The preprocessing's keying with a given stride on cell centres (the overflow, for tests): (positions, energies)."""
    idx = np.floor(np.asarray(xy, np.float64) / CELL).astype(np.int64)
    x_min = idx[:, 0].min()
    y_min = idx[:, 1].min() if y_min is None else y_min
    if idx[:, 0].max() - x_min + 2 > stride:
        raise ValueError("stride below the x range + 2; the preprocessing never uses one")
    keys = (idx[:, 0] - x_min) * stride + (idx[:, 1] - y_min)
    u, inv = np.unique(keys, return_inverse=True)
    pos = np.stack([u // stride + x_min, u % stride + y_min], -1) * CELL + CELL / 2
    return pos, np.bincount(inv, e)

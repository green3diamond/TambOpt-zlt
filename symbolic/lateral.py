"""Geometry within a plane: stored ground cells -> in-plane coordinates about the shower axis."""
import numpy as np

from symbolic import atmosphere

CELL = 10.0                       # m, ground cell size
X_M = 9.6                         # g/cm2, Moliere unit; r_M,ref = X_M / air density at the plane


def frame(direction):
    """Horizontal travel direction h, across-track perp and |cos zenith| of one shower."""
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    h = d[:2] / np.linalg.norm(d[:2])
    return h, np.array([-h[1], h[0]]), abs(d[2])


def to_plane(g, h, perp, c):
    """Ground offsets from the axis point -> in-plane (across track, along track); the plane is squashed by |cos z|."""
    return np.stack([g @ perp, g @ h / c], -1)


def r_m_ref(k, cos_zenith):
    """Reference Moliere radius [m] at plane k (0-based)."""
    alt = atmosphere.altitude((k + 1) * atmosphere.PLANE_SPACING, cos_zenith)
    return X_M / atmosphere.density(alt) / 100.0

"""Geometry within a plane (stored ground cells -> in-plane coordinates about the axis) and the NKG radial profile."""
import numpy as np
from scipy import special

from symbolic import atmosphere

CELL = 10.0                       # m, ground cell size
X_M = 9.6                         # g/cm2, Moliere unit; r_M,ref = X_M / air density at the plane
S_BOUNDS = (0.05, 2.2)            # NKG normalisable for 0 < s < 2.25
LN_R_BOUNDS = (np.log(1e-3), np.log(1e2))     # ln(r50 / r_M,ref)
S_STARTS = (1.0, 1.8)


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


def q(r, r_s, s):
    """Share of the NKG profile beyond radius r."""
    return special.betainc(4.5 - 2 * s, s, r_s / (r_s + r))


def x_median(s):
    """r / r_s holding half the energy."""
    t = special.betaincinv(s, 4.5 - 2 * s, 0.5)
    return t / (1 - t)

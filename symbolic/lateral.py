"""Geometry within a plane and the NKG radial law.

Frame: stored cells are CELL x CELL squares in ground (East, North) from the injection point; plane k's axis point is
(k + 1) x 500 m x direction (horizontal part). A ground offset g has in-plane coordinates a = g . perp (across track)
and b = g . h / |cos zenith| (along track), with h the horizontal travel direction and perp = h turned by 90 deg.

Radial law (NKG): density f(r) = C(s) / r_s^2 (r / r_s)^(s - 2) (1 + r / r_s)^(s - 4.5), normalised to 1 in the plane,
containment H(r) = I_{x / (1 + x)}(s, 4.5 - 2 s), x = r / r_s, and r_s = r50 / x_median(s) (r50 holds half the
energy). The Moliere radius at the plane, r_M,ref = X_M / air density, sets the scale.
"""
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
    """(..., 2) ground offsets from the axis point -> (..., 2) in-plane (a, b)."""
    return np.stack([g @ perp, g @ h / c], -1)


def r_m_ref(k, cos_zenith):
    """Reference Moliere radius [m] at plane k (0-based)."""
    alt = atmosphere.altitude((k + 1) * atmosphere.PLANE_SPACING, cos_zenith)
    return X_M / atmosphere.density(alt) / 100.0


def q(r, r_s, s):
    """1 - containment within radius r: I_{1 / (1 + x)}(4.5 - 2 s, s), x = r / r_s."""
    return special.betainc(4.5 - 2 * s, s, r_s / (r_s + r))


def x_median(s):
    """r / r_s holding half the energy."""
    t = special.betaincinv(s, 4.5 - 2 * s, 0.5)
    return t / (1 - t)

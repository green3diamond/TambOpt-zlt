"""Geometry within a plane: the shower's in-plane frame and the Moliere radius at each plane."""
import numpy as np

from symbolic import atmosphere

X_M = 9.6                         # g/cm2, Moliere unit; r_M,ref = X_M / air density at the plane


def frame(direction):
    """Horizontal travel direction h, across-track perp and |cos zenith| of one shower."""
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    h = d[:2] / np.linalg.norm(d[:2])
    return h, np.array([-h[1], h[0]]), abs(d[2])


def r_m_ref(k, cos_zenith):
    """Reference Moliere radius [m] at plane k (0-based)."""
    alt = atmosphere.altitude((k + 1) * atmosphere.PLANE_SPACING, cos_zenith)
    return X_M / atmosphere.density(alt) / 100.0

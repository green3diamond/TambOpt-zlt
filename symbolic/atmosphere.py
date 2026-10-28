"""Slant depth of the observation planes in the 5-layer USStdBK atmosphere used by the CORSIKA 8 training showers.

"zenith" is the direction of travel, so an axis with zenith below 90 deg climbs.
"""
import numpy as np

EARTH_RADIUS = 6371000.0          # m, CORSIKA 8 EarthRadius::Mean
INJECTION_HEIGHT = 2500.0         # m above EARTH_RADIUS
PLANE_SPACING = 500.0             # m; plane k sits at (k + 1) * PLANE_SPACING along the axis
STEP = 1.0                        # m, midpoint-rule step for the depth integral
# (upper boundary [m], b [g/cm2], c [m]) for the four exponential layers, then the constant top layer.
LAYERS = ((7.0e3, 1183.6071, 9542.4834), (11.4e3, 1143.0425, 8000.0534),
          (37.0e3, 1322.9748, 6295.6893), (100.0e3, 655.67307, 7375.2177))
TOP, TOP_DENSITY = 112.8e3, 1.0 / 1.0e9     # m, g/cm3


def density(h):
    """Air density [g/cm3] at altitude h [m] above the mean Earth radius."""
    h = np.asarray(h, dtype=np.float64)
    rho = np.where(h < TOP, TOP_DENSITY, 0.0)
    for top, b, c in reversed(LAYERS):
        rho = np.where(h < top, b / (c * 100.0) * np.exp(-h / c), rho)
    return rho


def altitude(s, cos_zenith, h0=INJECTION_HEIGHT):
    """Altitude [m] after travelling s [m] along a straight axis from height h0 at the given zenith (travel direction)."""
    r0 = EARTH_RADIUS + h0
    return np.sqrt(r0 * r0 + s * s + 2.0 * r0 * s * cos_zenith) - EARTH_RADIUS


def slant_depth(zenith_deg, distances, h0=INJECTION_HEIGHT, chunk=512):
    """(N, K) slant depth [g/cm2] from the injection point to each distance [m, multiples of STEP] along each axis."""
    zenith_deg, distances = np.asarray(zenith_deg, np.float64), np.asarray(distances, np.float64)
    idx = np.rint(distances / STEP).astype(int)
    if np.any(np.abs(idx * STEP - distances) > 1e-9) or np.any(idx < 1) or np.any(np.diff(idx) <= 0):
        raise ValueError("distances must be increasing positive multiples of STEP")
    s = (np.arange(idx[-1]) + 0.5) * STEP
    out = np.empty((len(zenith_deg), len(idx)))
    for lo in range(0, len(zenith_deg), chunk):
        cz = np.cos(np.radians(zenith_deg[lo:lo + chunk]))[:, None]
        h = altitude(s[None, :], cz, h0)
        if np.any(h < 0):
            raise ValueError("an axis goes below the mean Earth radius; depth undefined in this atmosphere")
        out[lo:lo + chunk] = np.cumsum(density(h), 1)[:, idx - 1] * STEP * 100.0
    return out


def plane_depths(zenith_deg, n_planes):
    """(N, n_planes) slant depth [g/cm2] of each observation plane."""
    return slant_depth(zenith_deg, PLANE_SPACING * np.arange(1, n_planes + 1))

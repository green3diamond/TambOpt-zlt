"""Arrival times per ring: log10(1 + t / ns) of the ring's earliest particle (lead) and its mean delay behind light on
the axis, each a power law t = exp(A) r^p k^-q in ring radius r and plane number k, constants per species and class.
The mean is plane_stats' ring_tmean, energy-weighted."""
import numpy as np
from scipy.optimize import least_squares

from symbolic import data, rows

TIMES = ("lead", "delay")
EDGE_IN = np.r_[0.5, data.EDGES[1:-1]]              # m, ring inner edges, the first floored
EDGE_OUT = np.r_[data.EDGES[1:-1], 12600.0]         # m, ring outer edges, the open last ring closed
LN_R = 0.5 * (np.log(EDGE_IN) + np.log(EDGE_OUT))   # ln of each ring's centre
GEOMETRIC = np.log(1 / (2 * 500.0 * 0.299792458))   # ln ns / m^2, t = r^2 / (2 k x 500 m x c)
STARTS = [(GEOMETRIC, p, q) for p in (1.0, 2.0) for q in (0.5, 1.0)]


def expand(v):
    """Rows (plane, ring) of rows.plane_inputs rows v: ln r first."""
    return np.column_stack([np.tile(LN_R, len(v)), np.repeat(v, data.N_RING, axis=0)])


def law(th, X):
    """log10(1 + t / ns) at expanded rows X (ln r, age, ln k, ...)."""
    return np.log10(1 + np.exp(np.minimum(th[0] + th[1] * X[:, 0] - th[2] * X[:, 2], 50.0)))


def fit(r, what):
    """Constants of one time from one species and class's plane rows r, least squares over rings with particles, the
    best of STARTS."""
    m = (r["ring_count"] > 0) & np.isfinite(r[what])
    X, y = expand(rows.plane_inputs(r["x"], r["k"]))[m.ravel()], r[what][m]
    fits = [least_squares(lambda th: law(th, X) - y, s, max_nfev=200) for s in STARTS]
    return min(fits, key=lambda q: q.cost).x


def ring_times(theta, x, k):
    """log10(1 + t / ns) per ring of one plane; x: its feature row, k: its 0-based number."""
    return law(theta, expand(rows.plane_inputs(x[None], np.array([k]))))

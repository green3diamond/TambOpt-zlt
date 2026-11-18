"""The width of a plane's ring profile along the shower: log10 of the stretch of the rings law that best matches the
plane (rings.plane_widths), as A tanh(B (ln k - C)) + D + E l + F age, with a spread ln sd = a + b ln k and AR(1)
memory rho from plane to plane. Law per species and class: [B, C, A, D, E, F, a, b, rho]."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from modules.constants import N_PLANES
from symbolic import rings

TANH_B = (0.5, 1.0, 2.0, 4.0)
TANH_C = (0.2, 0.4, 0.6, 0.8)                         # quantiles of ln k the grid of C starts from


def targets(r, B):
    """One species and class: per plane ln k, the linear inputs (l, age) and the target log10 stretch, and the shower
    and plane of each row; r: its plane rows, B: its rings law."""
    true = r["ring_energy"] / r["E_total"][:, None]
    y = rings.plane_widths(true, softmax(r["x"] @ B, axis=1))
    p, L = np.log(r["k"] + 1.0), r["x"][:, [3, 1]]
    if not (np.isfinite(p).all() and np.isfinite(L).all() and np.isfinite(y).all()):
        raise ValueError("width inputs or target not finite")
    return p, L, y


def basis(p, L, B, C):
    return np.column_stack([np.tanh(B * (p - C)), np.ones(len(p)), L])


def predict(theta, p, L):
    return basis(p, L, theta[0], theta[1]) @ theta[2:]


def fit_tanh(p, L, y, c_range=(0.0, np.log(N_PLANES))):
    """A tanh(B (p - C)) + D + L e by least squares: A, D, e solved exactly for each (B, C), which are searched on a
    grid and then refined, C kept inside c_range -> [B, C, A, D, e...]."""
    def design(theta):
        return basis(p, L, theta[0], np.clip(theta[1], *c_range))

    def loss(theta):
        b = design(theta)
        return ((y - b @ np.linalg.lstsq(b, y, rcond=None)[0]) ** 2).sum()

    start = min(((B, C) for B in TANH_B for C in np.quantile(p, TANH_C)), key=loss)
    best = minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000}).x
    return np.r_[best[0], np.clip(best[1], *c_range), np.linalg.lstsq(design(best), y, rcond=None)[0]]


def fit_spread(p, e):
    """ln sd = a + b p of residuals e by maximum likelihood -> (a, b)."""
    def nll(t):
        s = t[0] + t[1] * p
        return (s + 0.5 * e ** 2 * np.exp(-2 * s)).sum()

    return minimize(nll, [np.log(e.std()), 0.0], method="BFGS").x


def lag_one(v, unit, k):
    """Correlation of v on adjacent planes of one shower."""
    by = dict(zip(zip(unit.tolist(), k.tolist()), v))
    return float(np.corrcoef(np.array([(a, by[(u, q + 1)]) for (u, q), a in by.items() if (u, q + 1) in by]).T)[0, 1])


def fit(r, B):
    """The width law of one species and class from its plane rows r and rings law B."""
    p, L, y = targets(r, B)
    theta = fit_tanh(p, L, y)
    e = y - predict(theta, p, L)
    a, b = fit_spread(p, e)
    return np.r_[theta, a, b, lag_one(e * np.exp(-(a + b * p)), r["i"], r["k"])]


def mean(w, k, l, age):
    """Mean log10 width of planes k (0-based)."""
    return predict(w[:6], np.log(k + 1.0), np.column_stack([l, age]))


def sd(w, k):
    return np.exp(w[6] + w[7] * np.log(k + 1.0))

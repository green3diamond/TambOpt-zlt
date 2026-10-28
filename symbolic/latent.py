"""The four numbers per shower and species from a profile fit, z = (log10 Emax/E, Xmax, ln L, R), and summaries of
their distribution."""
import numpy as np
from scipy import stats

from symbolic import families as fam, structure

NAMES = {name: ("log10 Emax/E", "Xmax", "ln L", "R") for name in ("gh4lr", "gh4lr_signed")}
L_BOUNDS = (5.0, 5000.0)              # bounds on L [g/cm2]
R_ZERO = 1e-6                         # |R| below this: the Gaussian limit
E_EMPTY = 1.0                         # GeV: a plane with less is empty (below the detector's lit threshold)


def to_latent(f, log10_e):
    """(n, 4) latent from the fit arrays (theta, ln_nmax) and the primary log10 energy."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.column_stack([f["ln_nmax"] / np.log(10) - log10_e, f["theta"][:, 0], np.log(f["theta"][:, 1]),
                                f["theta"][:, 2]])


def from_latent(z, log10_e):
    """Inverse of to_latent: (xmax, L, R, ln_nmax)."""
    return z[:, 1], np.exp(z[:, 2]), z[:, 3], (z[:, 0] + log10_e) * np.log(10)


def start_depth(E, X, rng):
    """Depth where the profile starts: uniform between the last leading empty plane and the first lit one."""
    lit = E >= E_EMPTY
    n0, rows = np.argmax(lit, 1), np.arange(len(E))
    lo = np.where(n0 > 0, X[rows, np.maximum(n0 - 1, 0)], 0.0)
    hi = X[rows, n0]
    return np.where(lit.any(1), lo + rng.uniform(size=len(E)) * (hi - lo), np.nan)


def flags(f, X, Y):
    """Per-shower masks: unfit, on a parameter bound, peak before / after / outside the planes, R near 0."""
    xmax, L, R = f["theta"][:, 0], f["theta"][:, 1], f["theta"][:, 2]
    unfit = ~np.isfinite(f["ln_n"])
    first = np.where(np.isfinite(Y), X, np.inf).min(1)
    last = np.where(np.isfinite(Y), X, -np.inf).max(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos, neg = fam.r_caps(xmax, L, first, last)
        at_cap = (R >= 0.999 * pos * (1 - 1e-6)) | (R <= -0.999 * neg * (1 - 1e-6))
        bound = ((L <= L_BOUNDS[0] * 1.001) | (L >= L_BOUNDS[1] * 0.999) | at_cap
                 | (xmax - first <= -1000.0 + 1e-3) | (xmax - first >= fam.MAX_DEPTH - 1e-3))
        before, after = xmax < X[:, 0], xmax > X[:, -1]
    return {"unfit": unfit, "bound": bound & ~unfit, "before": before & ~unfit, "after": after & ~unfit,
            "outside": (before | after) & ~unfit, "r_zero": (np.abs(R) < R_ZERO) & ~unfit}


def fit_trend(Z, log10_e, cos_z):
    """Least-squares quadratic trend in (log10 E, cos zenith) per latent component; rows must be finite."""
    beta, *_ = np.linalg.lstsq(structure.design(log10_e, cos_z), Z, rcond=None)
    return beta


def residuals(Z, log10_e, cos_z, beta):
    return Z - structure.design(log10_e, cos_z) @ beta


def cell_index(log10_e, cos_z, e_edges, c_edges):
    ie = np.clip(np.digitize(log10_e, e_edges) - 1, 0, len(e_edges) - 2)
    ic = np.clip(np.digitize(cos_z, c_edges) - 1, 0, len(c_edges) - 2)
    return ie * (len(c_edges) - 1) + ic


def binned_means(Z, cells, n_cells, min_n=20):
    """(n_cells, d) mean latent per (energy, zenith) cell; NaN for cells with fewer than min_n showers."""
    out = np.full((n_cells, Z.shape[1]), np.nan)
    for k in range(n_cells):
        m = cells == k
        if m.sum() >= min_n:
            out[k] = Z[m].mean(0)
    return out


def r2(Z, pred):
    """Per component, 1 - residual variance / total variance, over rows where pred is finite."""
    ok = np.all(np.isfinite(pred), 1)
    return 1.0 - np.var(Z[ok] - pred[ok], 0) / np.var(Z[ok], 0)


def tail_stats(R):
    """Per component: skewness, excess kurtosis, fraction beyond 3 standard deviations (0.0027 for a Gaussian)."""
    s = (R - R.mean(0)) / R.std(0)
    return {"skew": stats.skew(s, 0).tolist(), "excess_kurtosis": stats.kurtosis(s, 0).tolist(),
            "frac_beyond_3sd": np.mean(np.abs(s) > 3, 0).tolist()}


def gaussian_loglik(R, cov):
    """Mean per-row log density of zero-mean Gaussian residuals R (n, d) under covariance cov (d, d)."""
    L = np.linalg.cholesky(cov)
    u = np.linalg.solve(L, R.T)
    return float(-0.5 * np.mean(np.sum(u * u, 0)) - np.sum(np.log(np.diag(L))) - 0.5 * R.shape[1] * np.log(2 * np.pi))


def block_diagonal(cov, sizes):
    """Keep only the diagonal blocks of the given sizes, in order (species independent)."""
    if sum(sizes) != len(cov):
        raise ValueError(f"blocks {sizes} do not tile a {len(cov)}x{len(cov)} matrix")
    out = np.zeros_like(cov)
    for i, k in zip(np.cumsum([0, *sizes[:-1]]), sizes):
        out[i:i + k, i:i + k] = cov[i:i + k, i:i + k]
    return out

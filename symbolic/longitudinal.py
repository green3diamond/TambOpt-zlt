"""The longitudinal profile: a Gaisser-Hillas curve in width L and asymmetry R (R may be negative), its per-shower fit
to the log energy per plane, and the four numbers per shower the sampler draws."""
import numpy as np
from scipy.optimize import least_squares

MAX_DEPTH = 1.0e4      # g/cm2
L_BOUNDS = (5.0, 5000.0)
R_MAX = 5.0
R_SERIES = 1e-4        # below this |R z| the R -> 0 series replaces log1p
E_EMPTY = 1.0          # GeV; a plane with less is empty
NAMES = ("log10 Emax/E", "Xmax", "ln L", "R")
LO = np.array([-1000.0, np.log(L_BOUNDS[0]), -0.999])
HI = np.array([MAX_DEPTH, np.log(L_BOUNDS[1]), 0.999])


def shape(X, xmax, L, R):
    """log E(X) - log E(Xmax); R = 0 is the Gaussian limit."""
    z = (X - xmax) / L
    t = R * z
    small = np.abs(t) < R_SERIES
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exact = np.log1p(t) / (R * R) - z / R
    series = -0.5 * z * z + R * z ** 3 / 3.0 - R * R * z ** 4 / 4.0
    return np.where(t <= -1.0, -np.inf, np.where(small, series, exact))


def shape_torch(X, xmax, L, R):
    """shape() on torch tensors with finite gradients everywhere (the unused branch gets safe inputs)."""
    import torch
    z = (X - xmax) / L
    t = R * z
    small = t.abs() < R_SERIES
    Rs = torch.where(small, torch.ones_like(R * z), R * torch.ones_like(z))
    ts = torch.clamp(torch.where(small, torch.zeros_like(t), t), min=-1.0 + 1e-12)
    exact = torch.log1p(ts) / (Rs * Rs) - z / Rs
    series = -0.5 * z * z + R * z ** 3 / 3.0 - R * R * z ** 4 / 4.0
    return torch.where(t <= -1.0, torch.full_like(z, -torch.inf), torch.where(small, series, exact))


def r_caps(xmax, L, xfirst, xlast):
    """Largest |R| for each sign that keeps the curve's start before the first plane and its end after the last."""
    ahead, behind = np.asarray(xmax - xfirst, np.float64), np.asarray(xlast - xmax, np.float64)
    with np.errstate(divide="ignore"):
        pos = np.where(ahead > 0, np.minimum(R_MAX, L / np.where(ahead > 0, ahead, 1.0)), R_MAX)
        neg = np.where(behind > 0, np.minimum(R_MAX, L / np.where(behind > 0, behind, 1.0)), R_MAX)
    return pos, neg


def _theta(u, xfirst, xlast):
    xmax, L = xfirst + u[0], np.exp(u[1])
    pos, neg = r_caps(xmax, L, xfirst, xlast)
    return xmax, L, u[2] * float(pos if u[2] >= 0 else neg)


def _starts(X, y):
    i = int(np.argmax(y))
    xm = X[i] if i < len(X) - 1 else X[-1] + 150.0
    return [np.array([x - X[0], np.log(L), v]) for x in (xm, xm + 150.0) for L in (150.0, 400.0) for v in (-0.3, 0.05, 0.5)]


def fit_one(X, y):
    """Fit one shower's log energies y (NaN = empty plane) at depths X -> (ln_n, (xmax, L, R), mse, n_fit), or None."""
    ok = np.isfinite(y)
    Xf, yf = X[ok], y[ok]
    if len(yf) < 6:
        return None

    def resid(u):
        r = yf - shape(Xf, *_theta(u, Xf[0], Xf[-1]))
        return r - r.mean()            # ln_n is profiled out

    best = None
    for u0 in _starts(Xf, yf):
        u0 = np.clip(u0, LO + 1e-6, HI - 1e-6)
        sol = least_squares(resid, u0, bounds=(LO, HI), method="trf", x_scale="jac", max_nfev=100 * len(u0))
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        return None
    theta = _theta(best.x, Xf[0], Xf[-1])
    ln_n = float(np.mean(yf - shape(Xf, *theta)))
    return ln_n, np.asarray(theta, np.float64), 2.0 * best.cost / len(yf), len(yf)


def fit_block(args):
    """Pool worker: fit every row of Y (log energy per plane) at depths X."""
    X, Y = args
    n = len(Y)
    out = {"ln_n": np.full(n, np.nan), "theta": np.full((n, 3), np.nan), "mse": np.full(n, np.nan),
           "n_fit": np.zeros(n, np.int64), "xmax": np.full(n, np.nan)}
    for i in range(n):
        r = fit_one(X[i], Y[i])
        if r is not None:
            out["ln_n"][i], out["theta"][i], out["mse"][i], out["n_fit"][i] = r
            out["xmax"][i] = r[1][0]
    return out


def log_energy(E):
    with np.errstate(divide="ignore"):
        return np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan)


def to_latent(f, log10_e):
    """(n, 4) numbers per shower from the fit arrays: log10 Emax/E, Xmax, ln L, R (ln_n is the log peak energy)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.column_stack([f["ln_n"] / np.log(10) - log10_e, f["theta"][:, 0], np.log(f["theta"][:, 1]),
                                f["theta"][:, 2]])


def from_latent(z, log10_e):
    """Inverse of to_latent: (xmax, L, R, ln_n)."""
    return z[:, 1], np.exp(z[:, 2]), z[:, 3], (z[:, 0] + log10_e) * np.log(10)


def start_depth(E, X, rng):
    """Depth where the profile starts: uniform between the last leading empty plane and the first lit one."""
    lit = E >= E_EMPTY
    n0, rows = np.argmax(lit, 1), np.arange(len(E))
    lo = np.where(n0 > 0, X[rows, np.maximum(n0 - 1, 0)], 0.0)
    hi = X[rows, n0]
    return np.where(lit.any(1), lo + rng.uniform(size=len(E)) * (hi - lo), np.nan)


def flags(f, X, E):
    """Per-shower masks: unfit, on a parameter bound, peak outside the planes (Xmax then undetermined)."""
    xmax, L, R = f["theta"][:, 0], f["theta"][:, 1], f["theta"][:, 2]
    unfit = ~np.isfinite(f["ln_n"])
    first = np.where(E > 0, X, np.inf).min(1)
    last = np.where(E > 0, X, -np.inf).max(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos, neg = r_caps(xmax, L, first, last)
        at_cap = (R >= 0.999 * pos * (1 - 1e-6)) | (R <= -0.999 * neg * (1 - 1e-6))
        bound = ((L <= L_BOUNDS[0] * 1.001) | (L >= L_BOUNDS[1] * 0.999) | at_cap
                 | (xmax - first <= -1000.0 + 1e-3) | (xmax - first >= MAX_DEPTH - 1e-3))
        outside = (xmax < X[:, 0]) | (xmax > X[:, -1])
    return {"unfit": unfit, "bound": bound & ~unfit, "outside": outside & ~unfit}

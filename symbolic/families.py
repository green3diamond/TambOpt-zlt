"""Longitudinal profile shapes log E(X) = ln_n + shape(X; theta) in slant depth X, and their per-shower fits."""
import numpy as np
from scipy.optimize import least_squares

X_RAD = 36.62          # g/cm2, radiation length of air in Greisen's t = X / X_RAD
X0_MIN = -500.0        # g/cm2, lowest Gaisser-Hillas start depth allowed
MAX_DEPTH = 1.0e4      # g/cm2, cap on depth-like parameters


def _gh_shape(X, xmax, x0, lam):
    d, z = xmax - x0, X - x0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = d / lam * np.log(z / d) + (xmax - X) / lam
    return np.where(z > 0, out, -np.inf)


def _peak_starts(X, y):
    i = int(np.argmax(y))
    xm = X[i] if i < len(X) - 1 else X[-1] + 150.0
    return (xm, xm + 150.0)


class Family:
    """A profile shape with its parameter transform, bounds and starting points."""

    def __init__(self, name, theta_names, shape, to_theta, bounds, starts, xmax, uses_last=False):
        self.name, self.theta_names = name, theta_names
        self.shape, self.to_theta, self.bounds, self.starts, self.xmax = shape, to_theta, bounds, starts, xmax
        self.uses_last = uses_last          # `extra` is the depth of the last fitted plane, set by fit_one

    @property
    def n_par(self):
        """Free parameters per shower, ln_n included."""
        return len(self.bounds(1.0, None)[0]) + 1


def gaisser_hillas(fixed_lambda=False):
    """theta = (xmax, x0, lam). u = (ln(xmax - x0), ln(xref - x0), ln lam); with fixed_lambda, lam comes from `extra`."""
    def to_theta(u, xref, extra):
        x0 = xref - np.exp(u[1])
        return x0 + np.exp(u[0]), x0, (extra if fixed_lambda else np.exp(u[2]))

    def bounds(xref, extra):
        lo, hi = [np.log(10.0), np.log(1.0)], [np.log(MAX_DEPTH), np.log(xref - X0_MIN)]
        if not fixed_lambda:
            lo, hi = lo + [np.log(5.0)], hi + [np.log(3000.0)]
        return lo, hi

    def starts(X, y, extra):
        x0 = min(0.0, X[0] - 1.0)
        lams = (None,) if fixed_lambda else (70.0, 200.0)
        return [np.array([np.log(xm - x0), np.log(X[0] - x0)] + ([] if lam is None else [np.log(lam)]))
                for xm in _peak_starts(X, y) for lam in lams]

    return Family("gh3" if fixed_lambda else "gh4", ("xmax", "x0", "lam"), _gh_shape, to_theta, bounds, starts,
                  lambda xmax, x0, lam: xmax)


R_MAX = 5.0            # cap on the Gaisser-Hillas asymmetry R
R_SERIES = 1e-4        # below this |R z|, the R -> 0 series replaces log1p (cancellation)


def _gh_lr_shape(X, xmax, L, R):
    """Gaisser-Hillas in width L and asymmetry R; R = 0 is the Gaussian limit."""
    z = (X - xmax) / L
    t = R * z
    small = np.abs(t) < R_SERIES
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exact = np.log1p(t) / (R * R) - z / R
    series = -0.5 * z * z + R * z ** 3 / 3.0 - R * R * z ** 4 / 4.0
    return np.where(t <= -1.0, -np.inf, np.where(small, series, exact))


def r_caps(xmax, L, xfirst, xlast):
    """Largest |R| for each sign that keeps the curve's start before the first plane and its end after the last."""
    ahead, behind = np.asarray(xmax - xfirst, np.float64), np.asarray(xlast - xmax, np.float64)
    with np.errstate(divide="ignore"):
        pos = np.where(ahead > 0, np.minimum(R_MAX, L / np.where(ahead > 0, ahead, 1.0)), R_MAX)
        neg = np.where(behind > 0, np.minimum(R_MAX, L / np.where(behind > 0, behind, 1.0)), R_MAX)
    return pos, neg


def gaisser_hillas_lr(signed=False):
    """theta = (xmax, L, R). gh4lr keeps R >= 0; gh4lr_signed (the one in use) also allows R < 0."""
    def to_theta(u, xref, extra):
        xmax, L = xref + u[0], np.exp(u[1])
        pos, neg = r_caps(xmax, L, xref, extra)
        return xmax, L, u[2] * float(pos if u[2] >= 0 else neg)

    def bounds(xref, extra):
        return [-1000.0, np.log(5.0), -0.999 if signed else 0.0], [MAX_DEPTH, np.log(5000.0), 0.999]

    def starts(X, y, extra):
        return [np.array([xm - X[0], np.log(L), v]) for xm in _peak_starts(X, y) for L in (150.0, 400.0)
                for v in ((-0.3, 0.05, 0.5) if signed else (0.05, 0.5))]

    return Family("gh4lr_signed" if signed else "gh4lr", ("xmax", "L", "R"), _gh_lr_shape, to_theta, bounds, starts,
                  lambda xmax, L, R: xmax, uses_last=True)


def _age_shape(X, xmax, w):
    s = 3.0 * X / (X + 2.0 * xmax)
    return -(s - 1.0) ** 2 / (2.0 * w * w)


GAUSS_AGE = Family(
    "age3", ("xmax", "w"), _age_shape,
    lambda u, xref, extra: (np.exp(u[0]), np.exp(u[1])),
    lambda xref, extra: ([np.log(10.0), np.log(0.02)], [np.log(MAX_DEPTH), np.log(5.0)]),
    lambda X, y, extra: [np.log([xm, w]) for xm in _peak_starts(X, y) for w in (0.3, 0.6)],
    lambda xmax, w: xmax)


def _greisen_shape(X, beta0):
    t = X / X_RAD
    s = 3.0 * t / (t + 2.0 * beta0)
    return t * (1.0 - 1.5 * np.log(s)) - 0.5 * np.log(beta0)


GREISEN = Family(
    "greisen2", ("beta0",), _greisen_shape,
    lambda u, xref, extra: (np.exp(u[0]),),
    lambda xref, extra: ([np.log(0.3)], [np.log(MAX_DEPTH / X_RAD)]),
    lambda X, y, extra: [np.log([xm / X_RAD]) for xm in _peak_starts(X, y)],
    lambda beta0: beta0 * X_RAD)


def _rise_shape(X, a, b):
    return np.log1p(-np.exp(-X / a)) - X / b


RISE = Family(
    "rise3", ("a", "b"), _rise_shape,
    lambda u, xref, extra: (np.exp(u[0]), np.exp(u[1])),
    lambda xref, extra: ([np.log(1.0), np.log(10.0)], [np.log(MAX_DEPTH), np.log(1.0e5)]),
    lambda X, y, extra: [np.log([a, b]) for a in (X[int(np.argmax(y))] / 2 + 1.0, 100.0) for b in (X[-1], 3 * X[-1])],
    lambda a, b: a * np.log((a + b) / a))

FAMILIES = {f.name: f for f in (gaisser_hillas(False), gaisser_hillas(True), gaisser_hillas_lr(), gaisser_hillas_lr(True),
                                 GAUSS_AGE, GREISEN, RISE)}
FOR_SPECIES = {"electron": ("gh4", "gh4lr", "gh4lr_signed", "gh3", "age3", "greisen2"),
               "photon": ("gh4", "gh4lr", "gh4lr_signed", "gh3", "age3", "greisen2"),
               "muon": ("gh4", "gh4lr", "gh4lr_signed", "gh3", "age3", "rise3")}
N_THETA = 3


def fit_one(fam, X, y, extra=None):
    """Fit one shower's log energies y (NaN = empty plane) at depths X -> (ln_n, theta, mse, n_fit), or None."""
    ok = np.isfinite(y)
    Xf, yf = X[ok], y[ok]
    if len(yf) < fam.n_par + 2:
        return None
    xref = Xf[0]
    if fam.uses_last:
        extra = Xf[-1]
    lo, hi = (np.asarray(b, np.float64) for b in fam.bounds(xref, extra))

    def resid(u):
        r = yf - fam.shape(Xf, *fam.to_theta(u, xref, extra))
        return r - r.mean()

    best = None
    for u0 in fam.starts(Xf, yf, extra):
        u0 = np.clip(u0, lo + 1e-6, hi - 1e-6)
        sol = least_squares(resid, u0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=100 * len(u0))
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        return None
    theta = fam.to_theta(best.x, xref, extra)
    ln_n = float(np.mean(yf - fam.shape(Xf, *theta)))
    return ln_n, np.asarray(theta, np.float64), 2.0 * best.cost / len(yf), len(yf)


def fit_block(args):
    """Pool worker: fit family `name` to every row of Y."""
    name, X, Y, extra = args
    fam = FAMILIES[name]
    n = len(Y)
    out = {"ln_n": np.full(n, np.nan), "theta": np.full((n, N_THETA), np.nan), "mse": np.full(n, np.nan),
           "n_fit": np.zeros(n, np.int64), "xmax": np.full(n, np.nan), "ln_nmax": np.full(n, np.nan)}
    for i in range(n):
        r = fit_one(fam, X[i], Y[i], None if extra is None else extra[i])
        if r is None:
            continue
        ln_n, theta, mse, n_fit = r
        xmax = fam.xmax(*theta)
        out["ln_n"][i], out["mse"][i], out["n_fit"][i], out["xmax"][i] = ln_n, mse, n_fit, xmax
        out["theta"][i, :len(theta)] = theta
        out["ln_nmax"][i] = ln_n + fam.shape(np.array([xmax]), *theta)[0]
    return out


def evaluate(name, fit, X):
    """Fitted log energy at depths X (-inf where the curve is zero, NaN where unfit)."""
    fam = FAMILIES[name]
    k = len(fam.theta_names)
    out = np.full(X.shape, np.nan)
    for i in np.nonzero(np.isfinite(fit["ln_n"]))[0]:
        out[i] = fit["ln_n"][i] + fam.shape(X[i], *fit["theta"][i, :k])
    return out

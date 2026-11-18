"""Particle energies within a ring: E = E_MIN + (ring mean - E_MIN) v above CORSIKA's tracking threshold E_MIN, with
v ~ Burr XII of mean 1 (CDF 1 - (1 + (v / lam)^c)^-k, a power-law tail). (c, k) per species, class and ring, by binned
likelihood over the raw tables' histograms of log10(E / ring mean); rings with fewer than MIN_PARTICLES take the
nearest fitted ring."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

from symbolic import data

E_MIN = {"electron": 0.01 * 0.999, "muon": 0.5 * 0.999, "photon": 0.01 * 0.999}   # GeV, a hair below the v2 runs' cuts
MIN_PARTICLES = 1000


def lam(c, k):
    """Burr XII scale that gives mean 1."""
    return np.exp(-np.log(k) - betaln(k - 1 / c, 1 + 1 / c))


def cdf(v, c, k):
    v = np.asarray(v, np.float64)
    with np.errstate(divide="ignore"):
        z = c * (np.log(np.where(v > 0, v, 1.0)) - np.log(lam(c, k)))
    return np.where(v > 0, -np.expm1(-k * np.logaddexp(0.0, z)), 0.0)


def params(a):
    """Unconstrained (a0, a1) -> (c, k) with c k > 1 (finite mean)."""
    c = np.exp(a[0])
    return c, (1 + np.exp(a[1])) / c


def sample(rng, c, k, n):
    """n draws of v (mean 1)."""
    return lam(c, k) * np.expm1(-np.log1p(-rng.uniform(size=n)) / k) ** (1 / c)


def fit_ring(h, mean, e_min):
    """(c, k) of one ring from its plane-ring histograms h over data.HIST and ring mean energies."""
    e = mean[:, None] * 10.0 ** data.HIST[None, :]
    v = (e - e_min) / (mean - e_min)[:, None]
    v[:, 0], v[:, -1] = -np.inf, np.inf
    w = h / h.sum()

    def nll(a):
        F = cdf(np.where(np.isfinite(v), v, 0.0), *params(a))
        F[:, -1] = 1.0
        return -np.sum(w * np.log(np.maximum(np.diff(F, axis=1), 1e-300)))

    best = min((minimize(nll, a0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
                for a0 in ([np.log(3.0), 0.0], [np.log(1.5), 1.0], [np.log(8.0), -1.0])), key=lambda s: s.fun)
    return params(best.x)


def fit(r, s):
    """(rings, 2) array of (c, k) for species s from one class's plane rows r."""
    mean = r["ring_energy"] / np.maximum(r["ring_count"], 1)
    ck = np.full((data.N_RING, 2), np.nan)
    for j in range(data.N_RING):
        h, m = r["hist"][:, j].astype(np.float64), mean[:, j]
        use = (h.sum(1) > 0) & (m > E_MIN[s] * (1 + 1e-6))
        if h[use].sum() >= MIN_PARTICLES:
            ck[j] = fit_ring(h[use], m[use], E_MIN[s])
    fitted = np.flatnonzero(np.isfinite(ck[:, 0]))
    for j in np.flatnonzero(~np.isfinite(ck[:, 0])):
        ck[j] = ck[fitted[np.argmin(np.abs(fitted - j))]]
    return ck

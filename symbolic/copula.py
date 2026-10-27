"""Distributional regression of the profile numbers: one margin per number, joined by a Gaussian copula.

Margins. Each number gets a normal (location, log scale) or Johnson SU law (location, log scale, skew gamma, log tail
delta) whose parameters are linear in (a, b) plus a b, with a, b the standardized log10 E and cos zenith (Design).
Johnson SU: x = xi + lambda sinh((z - gamma) / delta) with z standard normal, so its normal score is explicit.

Copula. z = Phi^-1(u), u the margin CDF; the scores of all numbers are joined by one correlation matrix. A shower's
log-likelihood is the sum of the margin log densities plus log c(z) = log N(z; 0, C) - sum log phi(z).
"""
import numpy as np
import torch
from scipy import stats

torch.set_default_dtype(torch.float64)
LOG_2PI = np.log(2 * np.pi)
U_EPS = 1e-12
LOG_DELTA_BOUNDS = (-3.0, 4.0)        # Johnson SU tail parameter; large delta is the normal limit


class Design:
    """Terms 1, a, b, a b in standardized (log10 E, cos zenith) = (a, b); scaling set on the training rows."""

    def __init__(self, log10_e, cos_z):
        self.m = np.array([np.mean(log10_e), np.mean(cos_z)])
        self.s = np.array([np.std(log10_e), np.std(cos_z)])

    def __call__(self, log10_e, cos_z):
        a, b = (np.asarray(log10_e) - self.m[0]) / self.s[0], (np.asarray(cos_z) - self.m[1]) / self.s[1]
        return np.column_stack([np.ones_like(a), a, b, a * b])


def _t(x):
    return torch.as_tensor(np.asarray(x, np.float64))


class Margin:
    """One number: family 'normal' or 'jsu', parameters D @ W. Values are standardized (loc, scale) before fitting."""
    N_PAR = {"normal": 2, "jsu": 4}

    def __init__(self, family):
        self.family, self.W = family, None

    def _params(self, D, W):
        P = D @ W
        if self.family == "normal":
            return P[:, 0], P[:, 1]
        return P[:, 0], P[:, 1], P[:, 2], P[:, 3].clamp(*LOG_DELTA_BOUNDS)

    def _g(self, x, p):
        """Standard-normal variate of standardized x, and log |dg/dx|."""
        if self.family == "normal":
            mu, ls = p
            return (x - mu) / ls.exp(), -ls
        xi, ll, gam, ld = p
        y = (x - xi) / ll.exp()
        return gam + ld.exp() * torch.asinh(y), ld - ll - 0.5 * torch.log1p(y * y)

    def _g_inv(self, z, p):
        if self.family == "normal":
            mu, ls = p
            return mu + ls.exp() * z
        xi, ll, gam, ld = p
        return xi + ll.exp() * torch.sinh((z - gam) / ld.exp())

    def _logpdf(self, xs, p):
        g, log_dg = self._g(xs, p)
        return log_dg - 0.5 * g * g - 0.5 * LOG_2PI

    def fit(self, D, x, max_iter=500):
        """Maximum likelihood (L-BFGS) on rows D (n, k), values x (n,); Johnson SU starts at delta 2, gamma 0."""
        D, x = _t(D), _t(x)
        self.loc, self.scale = float(x.mean()), float(x.std())
        xs = (x - self.loc) / self.scale
        beta = torch.linalg.lstsq(D, xs[:, None]).solution[:, 0]
        ls = torch.log(torch.std(xs - D @ beta))
        W = torch.zeros(D.shape[1], self.N_PAR[self.family])
        W[:, 0] = beta
        if self.family == "normal":
            W[0, 1] = ls
        else:
            W[0, 1], W[0, 3] = ls + np.log(1.76), np.log(2.0)
        W.requires_grad_(True)
        opt = torch.optim.LBFGS([W], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = -self._logpdf(xs, self._params(D, W)).mean()
            loss.backward()
            return loss
        opt.step(closure)
        self.W = W.detach()
        return self

    def loglik(self, D, x):
        D, x = _t(D), _t(x)
        return (self._logpdf((x - self.loc) / self.scale, self._params(D, self.W)) - np.log(self.scale)).numpy()

    def cdf(self, D, x):
        D, x = _t(D), _t(x)
        return torch.special.ndtr(self._g((x - self.loc) / self.scale, self._params(D, self.W))[0]).numpy()

    def ppf(self, D, u):
        D, u = _t(D), _t(u)
        z = torch.special.ndtri(u.clamp(U_EPS, 1 - U_EPS))
        return (self.loc + self.scale * self._g_inv(z, self._params(D, self.W))).numpy()

    def score(self, D, x):
        """Normal score of x, computed without passing through the CDF (exact in the tails)."""
        D, x = _t(D), _t(x)
        return self._g((x - self.loc) / self.scale, self._params(D, self.W))[0].numpy()


def normal_scores(margins, D, X):
    """(n, d) normal scores of the latent X (n, d)."""
    Z = np.column_stack([m.score(D, X[:, j]) for j, m in enumerate(margins)])
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"non-finite normal scores in columns {np.nonzero(~np.isfinite(Z).all(0))[0].tolist()}")
    return Z


def tail_rate(margin, D, x, q=stats.norm.sf(3)):
    """Fraction of rows beyond the margin's q and 1 - q quantiles (2 q expected; 0.0027 for 3 sd)."""
    u = margin.cdf(D, x)
    return float(np.mean((u < q) | (u > 1 - q)))


def corr(Z):
    C = np.cov(Z.T)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def gaussian_logc(Z, C):
    """Per-row log Gaussian copula density at normal scores Z under correlation C."""
    L = np.linalg.cholesky(C)
    w = np.linalg.solve(L, Z.T)
    return -0.5 * np.sum(w * w, 0) + 0.5 * np.sum(Z * Z, 1) - np.sum(np.log(np.diag(L)))


def block_corr(C, sizes):
    """Keep only the diagonal blocks of the given sizes (numbers of different species independent)."""
    out = np.zeros_like(C)
    for i, n in zip(np.cumsum([0, *sizes[:-1]]), sizes):
        out[i:i + n, i:i + n] = C[i:i + n, i:i + n]
    return out

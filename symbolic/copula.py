"""Conditional law of the profile numbers: a Johnson SU margin per number with parameters linear in log10 E,
cos zenith and their product, joined by one Gaussian copula."""
import numpy as np
import torch

LOG_2PI = np.log(2 * np.pi)
U_EPS = 1e-12
LOG_DELTA_BOUNDS = (-3.0, 4.0)        # Johnson SU tail parameter; large delta is the normal limit
LOG_SCALE_BOUNDS = (-7.0, 3.0)        # standardized scale; keeps small-sample fits off the zero-scale singularity


class Design:
    """Terms 1, a, b, a b in standardized (log10 E, cos zenith) = (a, b); scaling set on the training rows."""

    def __init__(self, m, s):
        self.m, self.s = np.asarray(m, np.float64), np.asarray(s, np.float64)

    @classmethod
    def fit(cls, log10_e, cos_z):
        return cls([np.mean(log10_e), np.mean(cos_z)], [np.std(log10_e), np.std(cos_z)])

    def __call__(self, log10_e, cos_z):
        a, b = (np.asarray(log10_e) - self.m[0]) / self.s[0], (np.asarray(cos_z) - self.m[1]) / self.s[1]
        return np.column_stack([np.ones_like(a), a, b, a * b])


def _t(x):
    return torch.as_tensor(np.asarray(x, np.float64))


class Margin:
    """The Johnson SU margin of one number: location, log scale, skew and log tail, each linear in the design."""

    def __init__(self, loc=None, scale=None, W=None):
        self.loc, self.scale = loc, scale
        self.W = None if W is None else _t(W)

    def _params(self, D, W):
        P = D @ W
        return P[:, 0], P[:, 1].clamp(*LOG_SCALE_BOUNDS), P[:, 2], P[:, 3].clamp(*LOG_DELTA_BOUNDS)

    def _g(self, x, p):
        """Standard-normal variate of standardized x, and log |dg/dx|."""
        xi, ll, gam, ld = p
        y = (x - xi) / ll.exp()
        return gam + ld.exp() * torch.asinh(y), ld - ll - 0.5 * torch.log1p(y * y)

    def _g_inv(self, z, p):
        xi, ll, gam, ld = p
        return xi + ll.exp() * torch.sinh((z - gam) / ld.exp())

    def fit(self, D, x, max_iter=500):
        """Maximum likelihood with L-BFGS."""
        D, x = _t(D), _t(x)
        self.loc, self.scale = float(x.mean()), float(x.std())
        xs = (x - self.loc) / self.scale
        beta = torch.linalg.lstsq(D, xs[:, None]).solution[:, 0]
        ls = torch.log(torch.std(xs - D @ beta))
        W = torch.zeros(D.shape[1], 4, dtype=torch.float64)
        W[:, 0] = beta
        W[0, 1], W[0, 3] = ls + np.log(1.76), np.log(2.0)
        W.requires_grad_(True)
        opt = torch.optim.LBFGS([W], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            g, log_dg = self._g(xs, self._params(D, W))
            loss = -(log_dg - 0.5 * g * g - 0.5 * LOG_2PI).mean()
            loss.backward()
            return loss
        opt.step(closure)
        self.W = W.detach()
        return self

    def ppf(self, D, u):
        D, u = _t(D), _t(u)
        z = torch.special.ndtri(u.clamp(U_EPS, 1 - U_EPS))
        return (self.loc + self.scale * self._g_inv(z, self._params(D, self.W))).numpy()

    def score(self, D, x):
        """Normal score of x, computed without passing through the CDF (exact in the tails)."""
        D, x = _t(D), _t(x)
        return self._g((x - self.loc) / self.scale, self._params(D, self.W))[0].numpy()


def normal_scores(margins, D, X):
    """(n, d) normal scores of the numbers X (n, d)."""
    Z = np.column_stack([m.score(D, X[:, j]) for j, m in enumerate(margins)])
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"non-finite normal scores in columns {np.nonzero(~np.isfinite(Z).all(0))[0].tolist()}")
    return Z


def corr(Z):
    C = np.cov(Z.T)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)

"""How a plane's energy splits over the rings about the axis: a multinomial logit of the ring shares in the plane's
features (a table of coefficients per ring), and the radial stretch that moves a profile in or out (width.py)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from symbolic import data

RIDGE = 1.0
CUM_CLIP = 1e-6
LOG_LAM = np.arange(-3.0, 3.0001, 0.01)                         # log10 stretch grid of plane_widths
BAND = (data.EDGES[1:-1] >= 10.0) & (data.EDGES[1:-1] <= 400.0)  # interior edges the width is matched on


def fit(x, shares, alpha=RIDGE):
    """Multinomial logit of shares (rows on the simplex), first ring the reference, ridge on all but the intercept."""
    w = np.clip(shares, 0.0, 1.0)
    n_f, n_b = x.shape[1], w.shape[1]
    pen = np.full((n_f, 1), alpha)
    pen[0] = 0.0

    def f(b):
        b = b.reshape(n_f, n_b - 1)
        eta = x @ np.column_stack([np.zeros(n_f), b])
        lp = eta - logsumexp(eta, axis=1, keepdims=True)
        g = (x.T @ (np.exp(lp) - w))[:, 1:] + pen * b
        return (-np.sum(w * lp) + 0.5 * np.sum(pen * b ** 2)) / len(x), g.ravel() / len(x)

    res = minimize(f, np.zeros(n_f * (n_b - 1)), jac=True, method="L-BFGS-B", options={"maxiter": 10000})
    return np.column_stack([np.zeros(n_f), res.x.reshape(n_f, n_b - 1)])


def cum_logodds(q):
    """Log-odds of the cumulative shares at the interior ring edges (last axis: rings -> rings - 1 edges)."""
    c = np.clip(np.cumsum(q, -1)[..., :-1], CUM_CLIP, 1 - CUM_CLIP)
    return np.log(c / (1 - c))


def stretch_logodds(y, log_lam):
    """Cumulative log-odds y (edges) of one profile stretched in radius by each 10^log_lam -> (len(log_lam), edges):
    linear in log r between the edges, extended linearly past both ends."""
    lr = np.log10(data.EDGES[1:-1])
    at = lr[None, :] - np.atleast_1d(log_lam)[:, None]
    v = np.interp(at, lr, y)
    v = np.where(at < lr[0], y[0] + (y[1] - y[0]) / (lr[1] - lr[0]) * (at - lr[0]), v)
    return np.where(at > lr[-1], y[-1] + (y[-1] - y[-2]) / (lr[-1] - lr[-2]) * (at - lr[-1]), v)


def stretch(q, log_lam):
    """Ring shares of profile q stretched in radius by 10^log_lam."""
    c = np.maximum.accumulate(1 / (1 + np.exp(-stretch_logodds(cum_logodds(q), log_lam)[0])))
    return np.diff(np.r_[0.0, c, 1.0])


def plane_widths(q_true, q_law, edges=BAND):
    """Per plane, the log10 stretch (grid LOG_LAM) of its law profile closest to the true profile on the cumulative
    log-odds at `edges`."""
    lam = np.empty(len(q_true))
    for p, (yl, yt) in enumerate(zip(cum_logodds(q_law), cum_logodds(q_true))):
        Y = stretch_logodds(yl, LOG_LAM)
        lam[p] = LOG_LAM[((yt[None, :] - Y)[:, edges] ** 2).sum(1).argmin()]
    return lam

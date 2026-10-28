"""Conditional law of the profile numbers: a Johnson SU (or normal) margin per number with parameters linear in
log10 E, cos zenith and their product, joined by one Gaussian copula."""
import numpy as np
import torch
from scipy import stats

torch.set_default_dtype(torch.float64)
LOG_2PI = np.log(2 * np.pi)
U_EPS = 1e-12
LOG_DELTA_BOUNDS = (-3.0, 4.0)        # Johnson SU tail parameter; large delta is the normal limit
LOG_SCALE_BOUNDS = (-7.0, 3.0)        # standardized scale; keeps small-sample fits off the zero-scale singularity


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
    """The margin of one number, family 'normal' or 'jsu'."""
    N_PAR = {"normal": 2, "jsu": 4}

    def __init__(self, family):
        self.family, self.W = family, None

    def _params(self, D, W):
        P = D @ W
        if self.family == "normal":
            return P[:, 0], P[:, 1].clamp(*LOG_SCALE_BOUNDS)
        return P[:, 0], P[:, 1].clamp(*LOG_SCALE_BOUNDS), P[:, 2], P[:, 3].clamp(*LOG_DELTA_BOUNDS)

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
        """Maximum likelihood with L-BFGS."""
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


class MLPMargin(Margin):
    """Johnson SU margin whose four parameters are a small MLP of the standardized (log10 E, cos zenith) instead of
    linear laws: the flexible-function control. Maximum likelihood with Adam, early stopping on a held-out tenth."""

    def __init__(self, seed=0, hidden=32, lr=1e-2, max_steps=20000, check=100, patience=20, val_frac=0.1):
        super().__init__("jsu")
        self.seed, self.hidden, self.lr, self.max_steps = seed, hidden, lr, max_steps
        self.check, self.patience, self.val_frac = check, patience, val_frac

    def _params(self, D, net):
        P = net(D[:, 1:3])
        return P[:, 0], P[:, 1], P[:, 2], P[:, 3].clamp(*LOG_DELTA_BOUNDS)

    def fit(self, D, x):
        D, x = _t(D), _t(x)
        self.loc, self.scale = float(x.mean()), float(x.std())
        xs = (x - self.loc) / self.scale
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(xs), generator=g)
        n_val = max(1, int(self.val_frac * len(xs)))
        va, tr = perm[:n_val], perm[n_val:]
        torch.manual_seed(self.seed)
        net = torch.nn.Sequential(torch.nn.Linear(2, self.hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(self.hidden, 4))
        with torch.no_grad():                                   # start at the constant law of the linear fit
            ls = torch.log(torch.std(xs))
            net[-1].weight.mul_(0.01)
            net[-1].bias.copy_(torch.stack([xs.mean(), ls + np.log(1.76), torch.tensor(0.0), torch.tensor(np.log(2.0))]))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        nll = lambda idx: -self._logpdf(xs[idx], self._params(D[idx], net)).mean()
        best, best_state, bad = np.inf, None, 0
        for step in range(self.max_steps):
            opt.zero_grad()
            nll(tr).backward()
            opt.step()
            if step % self.check == 0:
                with torch.no_grad():
                    v = float(nll(va))
                if v < best - 1e-5:
                    best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                elif (bad := bad + 1) >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.requires_grad_(False)
        self.W, self.steps, self.val_nll = net, step + 1, best
        return self


def normal_scores(margins, D, X):
    """(n, d) normal scores of the latent X (n, d)."""
    Z = np.column_stack([m.score(D, X[:, j]) for j, m in enumerate(margins)])
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"non-finite normal scores in columns {np.nonzero(~np.isfinite(Z).all(0))[0].tolist()}")
    return Z


def tail_rate(margin, D, x, q=stats.norm.sf(3)):
    """Fraction of rows beyond the margin's 3-sigma quantiles (0.0027 if the margin is right)."""
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
    """Zero the correlations between species."""
    out = np.zeros_like(C)
    for i, n in zip(np.cumsum([0, *sizes[:-1]]), sizes):
        out[i:i + n, i:i + n] = C[i:i + n, i:i + n]
    return out

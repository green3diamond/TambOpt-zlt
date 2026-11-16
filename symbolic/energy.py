"""Mean particle energy per ring: ln(ring energy / particle count) = c + sum_u a_u tanh(w_u . x + b_u), a sum of
TERMS tanh terms in ln r and the plane inputs, per species and class. Fitted as a tiny network (full-batch Adam, early
stopping on a fifth of the showers, best of SEEDS), then written out as its 9 TERMS + 1 constants."""
import numpy as np

from symbolic import data, rows, times

TERMS = 3
LR = 3e-3
EPOCHS = 3000
PATIENCE = 200
SEEDS = 3
VAL_SHARE = 0.2


def tanh_sum(theta, X):
    """c + sum_u a_u tanh(w_u . x + b_u) on raw inputs; theta = [w (U rows of len(x)), b (U), a (U), c]."""
    u = (len(theta) - 1) // (X.shape[1] + 2)
    w, b, a, c = theta[:u * X.shape[1]].reshape(u, -1), theta[-1 - 2 * u:-1 - u], theta[-1 - u:-1], theta[-1]
    return np.tanh(X @ w.T + b) @ a + c


def ring_means(theta, x, k):
    """ln mean particle energy per ring of one plane; x: its feature row, k: its 0-based number."""
    return tanh_sum(theta, times.expand(rows.plane_inputs(x[None], np.array([k]))))


def train(X, y, val, seed, terms=TERMS, epochs=EPOCHS):
    """One tanh sum on the rows not in val -> (validation error, constants on raw inputs and target)."""
    import torch
    torch.manual_seed(seed)
    mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-9
    y_mu, y_sd = y[~val].mean(), y[~val].std() + 1e-9
    tx = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
    ty = torch.as_tensor((y - y_mu) / y_sd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], terms), torch.nn.Tanh(), torch.nn.Linear(terms, 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    fit, hold = np.flatnonzero(~val), np.flatnonzero(val)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        opt.zero_grad()
        ((net(tx[fit]).squeeze(1) - ty[fit]) ** 2).mean().backward()
        opt.step()
        with torch.no_grad():
            v = float(((net(tx[hold]).squeeze(1) - ty[hold]) ** 2).mean())
        if v < best - 1e-6:
            best, state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(state)
    w, b, a, c = (t.detach().numpy().astype(np.float64) for t in net.parameters())
    return best, np.concatenate([(w / sd).ravel(), b - (w / sd) @ mu, a.ravel() * y_sd, c * y_sd + y_mu])


def fit(r, seeds=SEEDS, epochs=EPOCHS):
    """The law of one species and class from its plane rows r -> (validation error, constants)."""
    n = r["ring_count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(r["ring_energy"] / n)
    m = ((n > 0) & np.isfinite(y)).ravel()
    X, y = times.expand(rows.plane_inputs(r["x"], r["k"]))[m], y.ravel()[m]
    showers = np.repeat(r["i"], data.N_RING)[m]
    ids = np.unique(showers)
    fits = []
    for s in range(seeds):
        rng = np.random.default_rng(200 + s)
        val = np.isin(showers, rng.choice(ids, max(1, int(round(VAL_SHARE * len(ids)))), replace=False))
        fits.append(train(X, y, val, s, epochs=epochs))
    return min(fits, key=lambda q: q[0])

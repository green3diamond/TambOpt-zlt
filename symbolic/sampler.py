"""Energy per plane of a primary (part 1). Per species a profile curve (longitudinal.py) and its start depth, the 15
numbers of the three species drawn together from their joint law (copula.py), then plane-to-plane noise: the real
leftovers of the training showers around their curves, as quantile tables per position relative to the peak
t = (X - Xmax) / L, read through AR(1) normal scores. Fitted per class on clean showers (fitted, off the bounds, peaking
within the planes)."""
import numpy as np
from scipy import special

from modules.constants import SPECIES_NAMES
from symbolic import copula as cp, longitudinal

BLOCK = len(longitudinal.NAMES) + 1     # numbers per species: the profile, then the start depth
MAX_REDRAWS = 100
MIN_PAIR = 1000            # plane pairs a lag-1 correlation needs (at most a tenth of the showers)
MIN_BIN = 50               # leftovers a bin of t needs; emptier bins share the table of all leftovers
PEAK_EDGES = np.array([-np.inf, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6, np.inf])
Z_GRID = np.round(np.arange(-4.0, 4.0001, 0.05), 10)     # normal scores of the leftover tables


def load(path):
    """Conditions, plane energies and profile numbers of every species from a fits file (fit.py profiles)."""
    z = np.load(path)
    d = {"meta": {k: z[f"meta.{k}"] for k in ("log10_E", "cos_zenith", "labels")}, "X": z["X"]}
    rng = np.random.default_rng(0)
    for s in SPECIES_NAMES:
        E = z[f"{s}.E"]
        f = {k: z[f"{s}.{k}"] for k in ("ln_n", "theta")}
        fl = longitudinal.flags(f, d["X"], E)
        d[s] = {"E": E, **f, "Z": longitudinal.to_latent(f, d["meta"]["log10_E"]),
                "start": longitudinal.start_depth(E, d["X"], rng), "clean": ~(fl["unfit"] | fl["bound"] | fl["outside"])}
    return d


def peak_position(X, xmax, L):
    """t = (X - Xmax) / L of every plane, (n, planes)."""
    with np.errstate(all="ignore"):
        return (X - xmax[:, None]) / L[:, None]


def table_bins(t, r):
    """Per bin of t: its median t (bins with fewer than MIN_BIN leftovers: their middle, open bins one unit beyond
    their edge) and the leftovers' quantiles on the normal-score grid (emptier bins share those of all leftovers)."""
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    levels = special.ndtr(Z_GRID)
    pooled = np.quantile(r[ok], levels)
    at, q = [], []
    for k in range(len(PEAK_EDGES) - 1):
        m = ok & (b == k)
        if m.sum() >= MIN_BIN:
            at.append(np.median(t[m]))
            q.append(np.quantile(r[m], levels))
            continue
        lo, hi = PEAK_EDGES[k], PEAK_EDGES[k + 1]
        at.append(hi - 1.0 if not np.isfinite(lo) else lo + 1.0 if not np.isfinite(hi) else 0.5 * (lo + hi))
        q.append(pooled)
    return np.array(at), np.array(q)


def table_scores(at, q, t, r):
    """Normal scores of leftovers r: their position in their own bin's table (NaN where r or t is not finite)."""
    z = np.full(r.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    for k, qk in enumerate(q):
        sel = ok & (b == k)
        z[sel] = np.interp(r[sel], qk, Z_GRID)
    return z


def table_noise(p, t, z):
    """Leftovers at positions t from normal numbers z (clipped to the grid): each bin's table, linear in t between
    the two neighbouring bins' median t, flat beyond the outer bins."""
    z = np.clip(z, Z_GRID[0], Z_GRID[-1])
    Q = np.stack([np.interp(z, Z_GRID, qk) for qk in p["q"]])
    at = p["at"]
    i1 = np.clip(np.searchsorted(at, t), 1, len(at) - 1)
    i0 = i1 - 1
    w = np.clip((t - at[i0]) / (at[i1] - at[i0]), 0.0, 1.0)
    return (1 - w) * np.take_along_axis(Q, i0[None], 0)[0] + w * np.take_along_axis(Q, i1[None], 0)[0]


def plane_noise(d, rows):
    """Per species: the leftover tables (log E_k minus the curve) and the lag-1 correlation of their normal scores."""
    X = d["X"][rows]
    out = {}
    for s in SPECIES_NAMES:
        th = d[s]["theta"][rows]
        with np.errstate(all="ignore"):
            curve = d[s]["ln_n"][rows][:, None] + longitudinal.shape(X, *[th[:, i, None] for i in range(3)])
            r = longitudinal.log_energy(d[s]["E"][rows]) - curve
        r[~np.isfinite(r)] = np.nan
        t = peak_position(X, th[:, 0], th[:, 1])
        at, q = table_bins(t, r)
        u = table_scores(at, q, t, r)
        min_pair = min(MIN_PAIR, len(rows) // 10)
        rhos = [np.corrcoef(u[ok, k], u[ok, k + 1])[0, 1] for k in range(r.shape[1] - 1)
                for ok in [np.isfinite(u[:, k]) & np.isfinite(u[:, k + 1])] if ok.sum() >= min_pair]
        out[s] = {"at": at, "q": q, "rho": float(np.mean(rhos))}
    return out


def fit(d, c):
    """The sampler of class c from the output of load(), on the showers clean in every species."""
    ok = [d[s]["clean"] & np.isfinite(d[s]["start"]) for s in SPECIES_NAMES]
    rows = np.nonzero((d["meta"]["labels"] == c) & np.all(ok, 0))[0]
    X = np.hstack([np.column_stack([d[s]["Z"][rows], d[s]["start"][rows]]) for s in SPECIES_NAMES])
    l, u = d["meta"]["log10_E"][rows], d["meta"]["cos_zenith"][rows]
    design = cp.Design.fit(l, u)
    D = design(l, u)
    margins = [cp.Margin().fit(D, X[:, j]) for j in range(X.shape[1])]
    return {"design": design, "margins": margins, "C": cp.corr(cp.normal_scores(margins, D, X)),
            "planes": plane_noise(d, rows), "n_train": len(rows)}


def to_arrays(model, cn):
    """A class's sampler as flat arrays (for the model folder)."""
    out = {f"{cn}.design": np.r_[model["design"].m, model["design"].s], f"{cn}.C": model["C"],
           f"{cn}.loc": np.array([m.loc for m in model["margins"]]), f"{cn}.scale": np.array([m.scale for m in model["margins"]]),
           f"{cn}.W": np.stack([m.W.numpy() for m in model["margins"]]), f"{cn}.n_train": np.array(model["n_train"])}
    for s, p in model["planes"].items():
        out.update({f"{cn}.{s}.{k}": np.asarray(v) for k, v in p.items()})
    return out


def from_arrays(z, cn):
    """Inverse of to_arrays."""
    margins = [cp.Margin(float(a), float(b), W) for a, b, W in zip(z[f"{cn}.loc"], z[f"{cn}.scale"], z[f"{cn}.W"])]
    planes = {s: {"at": z[f"{cn}.{s}.at"], "q": z[f"{cn}.{s}.q"], "rho": float(z[f"{cn}.{s}.rho"])} for s in SPECIES_NAMES}
    return {"design": cp.Design(z[f"{cn}.design"][:2], z[f"{cn}.design"][2:]), "margins": margins, "C": z[f"{cn}.C"],
            "planes": planes, "n_train": int(z[f"{cn}.n_train"])}


def species_block(Z, i):
    """Species i's profile numbers and start depth."""
    b = Z[:, BLOCK * i:BLOCK * (i + 1)]
    return b[:, :-1], b[:, -1]


def ar1(n, k, rho, rng):
    a = np.empty((n, k))
    a[:, 0] = rng.normal(size=n)
    for j in range(1, k):
        a[:, j] = rho * a[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return a


def ends_inside(Z, l, X):
    """Rows where any species' curve ends (R < 0) before the last plane."""
    bad = np.zeros(len(l), bool)
    for i in range(len(SPECIES_NAMES)):
        xmax, L, R, _ = longitudinal.from_latent(species_block(Z, i)[0], l)
        with np.errstate(divide="ignore", invalid="ignore"):
            bad |= (R < 0) & (xmax - L / R < X[:, -1])
    return bad


def sample(model, l, u, X, rng):
    """One draw per primary -> (energy per plane per species, the drawn numbers, redraws). Curves that end before
    the last plane are drawn again."""
    def draw(idx):
        D = model["design"](l[idx], u[idx])
        Z = rng.multivariate_normal(np.zeros(len(model["C"])), model["C"], len(idx))
        return np.column_stack([m.ppf(D, special.ndtr(Z[:, j])) for j, m in enumerate(model["margins"])])

    Z = draw(np.arange(len(l)))
    bad = ends_inside(Z, l, X)
    redrawn = int(bad.sum())
    for _ in range(MAX_REDRAWS):
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        Z[idx] = draw(idx)
        bad[idx] = ends_inside(Z[idx], l[idx], X[idx])
    if bad.any():
        raise ValueError(f"{bad.sum()} draws still end inside the planes after {MAX_REDRAWS} redraws")
    out = {}
    for i, s in enumerate(SPECIES_NAMES):
        z, start = species_block(Z, i)
        xmax, L, R, ln_n = longitudinal.from_latent(z, l)
        with np.errstate(all="ignore"):
            y = ln_n[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
        y = np.where(np.isfinite(y) & (X >= start[:, None]), y, np.nan)
        p = model["planes"][s]
        t = peak_position(X, xmax, L)
        t = np.where(np.isfinite(t), t, 0.0)
        y = y + table_noise(p, t, ar1(len(l), y.shape[1], p["rho"], rng))
        out[s] = np.where(np.isfinite(y), np.exp(np.where(np.isfinite(y), y, 0.0)), 0.0)
    return out, Z, redrawn

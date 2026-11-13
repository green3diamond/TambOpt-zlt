"""Draws the energy per plane for a primary: 5 numbers per species (profile + start depth) from the copula, the
profile curve at the plane depths, then a per-plane correction and AR(1) noise. Fitted per class.
The correction and noise size are set per plane number (noise "plane", the adopted model) or per position relative
to the shower's own peak, t = (X - Xmax) / L (noise "peak"): with "plane", a plane number that is early in some
showers and at the peak in others carries the early planes' large scatter to the peak. Noise "skew": per bin of t a
Johnson SU for the leftovers instead of a normal (they are skewed low near the peak, and a symmetric noise on ln E
inflates the summed energy). Noise "table": per bin of t the real leftovers' quantiles, read through normal scores,
never beyond the leftovers seen (the Johnson SU's tails reach past them and blow up the summed energy). Noise "bright":
per bin of t a linear term in the curve's brightness v = ln curve - ln E (a curve bright for its energy has real planes
below it), then the "table" form on what that leaves."""
import numpy as np
from scipy import special

from modules.constants import SPECIES_NAMES
from symbolic import copula as cp, longitudinal

LOG10_E_MAX = 6.25         # the training showers end here
TOP = 6.125                # log10 E; the training file's top bin is under-filled and left out
MIN_PAIR = 1000
MAX_REDRAWS = 100
BLOCK = len(longitudinal.NAMES) + 1     # numbers per species: the profile, then the start depth
PEAK_EDGES = np.array([-np.inf, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6, np.inf])
MIN_BIN = 50               # leftovers a bin of t needs; emptier bins take the pooled bias and sd
MAX_SKEW_FIT = 20000       # leftovers per bin the Johnson SU is fitted on (seeded subsample)
Z_GRID = np.round(np.arange(-4.0, 4.0001, 0.05), 10)     # normal scores of the leftover tables (noise "table")


def load(path, e_max=LOG10_E_MAX):
    """Conditions, plane data and profile numbers of every species from a fits file (fit_profiles.py), showers with
    log10 E <= e_max. Clean = fitted, off the bounds and peaking within the planes."""
    z = np.load(path)
    keep = z["meta.log10_E"] <= e_max
    d = {"meta": {k: z[f"meta.{k}"][keep] for k in ("log10_E", "cos_zenith", "labels")}, "X": z["X"][keep]}
    rng = np.random.default_rng(0)
    for s in SPECIES_NAMES:
        E = z[f"{s}.E"][keep]
        f = {k: z[f"{s}.{k}"][keep] for k in ("ln_n", "theta")}
        fl = longitudinal.flags(f, d["X"], E)
        d[s] = {"E": E, **f, "Z": longitudinal.to_latent(f, d["meta"]["log10_E"]),
                "start": longitudinal.start_depth(E, d["X"], rng), "clean": ~(fl["unfit"] | fl["bound"] | fl["outside"])}
    return d


def class_rows(d, c, top=TOP, keep=None, rows="clean"):
    """Showers of class c (within keep, if given): joint numbers and conditions. rows "clean": a clean fit in every
    species; "fitted": every shower fitted in every species (also peaks outside the planes and numbers on a bound)."""
    if rows == "clean":
        ok = [d[s]["clean"] & np.isfinite(d[s]["start"]) for s in SPECIES_NAMES]
    elif rows == "fitted":
        ok = [np.all(np.isfinite(d[s]["Z"]), 1) & np.isfinite(d[s]["start"]) for s in SPECIES_NAMES]
    else:
        raise ValueError(f"rows {rows!r}: clean or fitted")
    m = ((d["meta"]["labels"] == c) & (d["meta"]["log10_E"] < top) & (True if keep is None else keep) & np.all(ok, 0))
    return {"rows": np.nonzero(m)[0],
            "X": np.hstack([np.column_stack([d[s]["Z"][m], d[s]["start"][m]]) for s in SPECIES_NAMES]),
            "l": d["meta"]["log10_E"][m], "u": d["meta"]["cos_zenith"][m]}


def peak_position(X, xmax, L):
    """t = (X - Xmax) / L of every plane, (n, planes)."""
    with np.errstate(all="ignore"):
        return (X - xmax[:, None]) / L[:, None]


def peak_bins(t, r):
    """Per bin of t (PEAK_EDGES): the median t, mean and sd of the leftovers r. A bin with fewer than MIN_BIN takes
    the pooled mean and sd, placed at its middle (open outer bins: one unit beyond their edge)."""
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    pooled = (np.mean(r[ok]), np.std(r[ok]))
    at, bias, sd = [], [], []
    for k in range(len(PEAK_EDGES) - 1):
        m = ok & (b == k)
        if m.sum() >= MIN_BIN:
            at.append(np.median(t[m])), bias.append(np.mean(r[m])), sd.append(np.std(r[m]))
            continue
        lo, hi = PEAK_EDGES[k], PEAK_EDGES[k + 1]
        at.append(hi - 1.0 if not np.isfinite(lo) else lo + 1.0 if not np.isfinite(hi) else 0.5 * (lo + hi))
        bias.append(pooled[0]), sd.append(pooled[1])
    return np.array(at), np.array(bias), np.array(sd)


def at_peak(p, t):
    """Bias and sd of the peak noise model p read off at positions t (linear between bins, flat beyond)."""
    return np.interp(t, p["at"], p["bias_t"]), np.interp(t, p["at"], p["sd_t"])


def _ones(n):
    return np.ones((n, 1))


def skew_bins(t, r, rng):
    """Per bin of t: the median t (as peak_bins) and a Johnson SU of the leftovers r (at most MAX_SKEW_FIT of them,
    seeded subsample); bins with fewer than MIN_BIN share one Johnson SU of all leftovers."""
    at, _, _ = peak_bins(t, r)
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1

    def fitted(x):
        x = x if len(x) <= MAX_SKEW_FIT else rng.choice(x, MAX_SKEW_FIT, replace=False)
        return cp.Margin("jsu").fit(_ones(len(x)), x)
    pooled = None
    margins = []
    for k in range(len(PEAK_EDGES) - 1):
        x = r[ok & (b == k)]
        if len(x) >= MIN_BIN:
            margins.append(fitted(x))
        else:
            pooled = pooled or fitted(r[ok])
            margins.append(pooled)
    return at, margins


def skew_scores(p, t, r):
    """Normal scores of leftovers r through the Johnson SU of their own bin of t (NaN where r or t is not finite)."""
    z = np.full(r.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    for k, m in enumerate(p["margins"]):
        sel = ok & (b == k)
        if sel.any():
            z[sel] = m.score(_ones(sel.sum()), r[sel])
    return z


def skew_noise(p, t, z):
    """Leftovers at positions t from normal numbers z: each bin's Johnson SU quantile, linear in t between the two
    neighbouring bins' median t, flat beyond the outer bins."""
    u = special.ndtr(z).ravel()
    return blend(p["at"], np.stack([m.ppf(_ones(len(u)), u) for m in p["margins"]]).reshape(len(p["at"]), *z.shape), t)


def blend(at, Q, t):
    """Q[k] = the value of bin k at every point; the value at positions t, linear between the two neighbouring bins'
    median t (at), flat beyond the outer bins."""
    i1 = np.clip(np.searchsorted(at, t), 1, len(at) - 1)
    i0 = i1 - 1
    w = np.clip((t - at[i0]) / (at[i1] - at[i0]), 0.0, 1.0)
    return (1 - w) * np.take_along_axis(Q, i0[None], 0)[0] + w * np.take_along_axis(Q, i1[None], 0)[0]


def table_bins(t, r):
    """Per bin of t: the median t (as peak_bins) and the leftovers' quantiles on the normal-score grid Z_GRID; bins
    with fewer than MIN_BIN share the quantiles of all leftovers."""
    at, _, _ = peak_bins(t, r)
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    levels = special.ndtr(Z_GRID)
    pooled = np.quantile(r[ok], levels)
    q = [np.quantile(r[ok & (b == k)], levels) if (ok & (b == k)).sum() >= MIN_BIN else pooled
         for k in range(len(PEAK_EDGES) - 1)]
    return at, np.array(q)


def table_scores(p, t, r):
    """Normal scores of leftovers r: their position in their own bin's table (NaN where r or t is not finite)."""
    z = np.full(r.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(r)
    b = np.digitize(t, PEAK_EDGES) - 1
    for k, q in enumerate(p["q"]):
        sel = ok & (b == k)
        z[sel] = np.interp(r[sel], q, Z_GRID)
    return z


def table_noise(p, t, z):
    """Leftovers at positions t from normal numbers z (clipped to the grid): each bin's table, blended in t."""
    z = np.clip(z, Z_GRID[0], Z_GRID[-1])
    return blend(p["at"], np.stack([np.interp(z, Z_GRID, q) for q in p["q"]]), t)


def bright_bins(t, v, r):
    """Per bin of t: least squares r = a + b v (bins with fewer than MIN_BIN leftovers take the fit on all) -> (a, b)."""
    ok = np.isfinite(t) & np.isfinite(v) & np.isfinite(r)
    bins = np.digitize(t, PEAK_EDGES) - 1
    line = lambda m: np.polyfit(v[m], r[m], 1)[::-1]
    pooled = line(ok)
    ab = [line(ok & (bins == k)) if (ok & (bins == k)).sum() >= MIN_BIN else pooled for k in range(len(PEAK_EDGES) - 1)]
    return np.array(ab)


def bright_mean(p, t, v):
    """The brightness term a + b v at positions t, blended in t as the tables."""
    return blend(p["at"], p["ab"][:, 0, None, None] + p["ab"][:, 1, None, None] * v[None], t)


def plane_residuals(d, rows, noise="plane"):
    """Per species: mean and sd of log E_k minus the curve, per plane number (noise "plane") or per position
    relative to the shower's own peak (noise "peak"), and the plane-to-plane correlation of the standardized
    leftovers (noise "skew": of their normal scores through each bin's Johnson SU)."""
    X = d["X"][rows]
    out = {}
    rng = np.random.default_rng(0)
    for s in SPECIES_NAMES:
        th = d[s]["theta"][rows]
        with np.errstate(all="ignore"):
            curve = d[s]["ln_n"][rows][:, None] + longitudinal.shape(X, *[th[:, i, None] for i in range(3)])
            r = longitudinal.log_energy(d[s]["E"][rows]) - curve
        r[~np.isfinite(r)] = np.nan
        if noise == "peak":
            t = peak_position(X, th[:, 0], th[:, 1])
            at, bias_t, sd_t = peak_bins(t, r)
            p = {"noise": "peak", "at": at, "bias_t": bias_t, "sd_t": sd_t}
            bias, sd = at_peak(p, np.where(np.isfinite(t), t, 0.0))
        elif noise == "skew":
            t = peak_position(X, th[:, 0], th[:, 1])
            at, margins = skew_bins(t, r, rng)
            p = {"noise": "skew", "at": at, "margins": margins}
            bias, sd = 0.0, 1.0
            r = skew_scores(p, t, r)
        elif noise == "table":
            t = peak_position(X, th[:, 0], th[:, 1])
            at, q = table_bins(t, r)
            p = {"noise": "table", "at": at, "q": q}
            bias, sd = 0.0, 1.0
            r = table_scores(p, t, r)
        elif noise == "bright":
            t = peak_position(X, th[:, 0], th[:, 1])
            v = curve - np.log(10.0) * d["meta"]["log10_E"][rows][:, None]
            ab = bright_bins(t, v, r)
            bins = np.clip(np.digitize(t, PEAK_EDGES) - 1, 0, len(ab) - 1)
            e = r - (ab[bins, 0] + ab[bins, 1] * v)
            at, q = table_bins(t, e)
            p = {"noise": "bright", "at": at, "q": q, "ab": ab}
            bias, sd = 0.0, 1.0
            r = table_scores(p, t, e)
        elif noise == "plane":
            bias, sd = np.nanmean(r, 0), np.nanstd(r, 0)
            p = {"bias": np.nan_to_num(bias), "sd": np.nan_to_num(sd)}
        else:
            raise ValueError(f"noise {noise!r}: plane, peak, skew, table or bright")
        u = (r - bias) / sd
        min_pair = min(MIN_PAIR, len(rows) // 10)
        rhos = [np.corrcoef(u[ok, k], u[ok, k + 1])[0, 1] for k in range(r.shape[1] - 1)
                for ok in [np.isfinite(u[:, k]) & np.isfinite(u[:, k + 1])] if ok.sum() >= min_pair]
        out[s] = {**p, "rho": float(np.mean(rhos))}
    return out


def fit(d, c, top=TOP, keep=None, margin=lambda: cp.Margin("jsu"), noise="plane", rows="clean"):
    """Fit the sampler for class c on the output of load(), on showers below log10 E = top (and within keep).
    margin() makes each number's margin: linear laws by default. noise: the plane residual model (module docstring).
    rows: which showers it learns from (class_rows)."""
    t = class_rows(d, c, top, keep, rows)
    design = cp.Design(t["l"], t["u"])
    D = design(t["l"], t["u"])
    margins = [margin().fit(D, t["X"][:, j]) for j in range(t["X"].shape[1])]
    C = cp.corr(cp.normal_scores(margins, D, t["X"]))
    return {"design": design, "margins": margins, "C": C, "planes": plane_residuals(d, t["rows"], noise),
            "n_train": len(t["rows"]), "rows": rows}


def species_block(Zlat, i):
    """Species i's profile numbers and start depth."""
    b = Zlat[:, BLOCK * i:BLOCK * (i + 1)]
    return b[:, :-1], b[:, -1]


def to_profiles(Zlat, l, X):
    """Per-plane log energies (NaN where the curve vanishes or before the start depth) of each species."""
    out = {}
    for i, s in enumerate(SPECIES_NAMES):
        z, start = species_block(Zlat, i)
        xmax, L, R, ln_n = longitudinal.from_latent(z, l)
        with np.errstate(all="ignore"):
            Y = ln_n[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
        out[s] = np.where(np.isfinite(Y) & (X >= start[:, None]), Y, np.nan)
    return out


def ar1(n, k, rho, rng):
    a = np.empty((n, k))
    a[:, 0] = rng.normal(size=n)
    for j in range(1, k):
        a[:, j] = rho * a[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return a


def ends_inside(Zlat, l, X):
    """Rows where any species' curve ends (R < 0) before the last plane."""
    bad = np.zeros(len(l), bool)
    for i in range(len(SPECIES_NAMES)):
        xmax, L, R, _ = longitudinal.from_latent(species_block(Zlat, i)[0], l)
        with np.errstate(divide="ignore", invalid="ignore"):
            bad |= (R < 0) & (xmax - L / R < X[:, -1])
    return bad


def sample(model, l, u, X, rng):
    """One draw per primary -> (energy per plane per species, the drawn numbers). Curves that end before the last
    plane are drawn again."""
    def draw(idx):
        D = model["design"](l[idx], u[idx])
        Z = rng.multivariate_normal(np.zeros(len(model["C"])), model["C"], len(idx))
        return np.column_stack([m.ppf(D, special.ndtr(Z[:, j])) for j, m in enumerate(model["margins"])])

    lat = draw(np.arange(len(l)))
    bad = ends_inside(lat, l, X)
    model["n_drawn"] = model.get("n_drawn", 0) + len(l)
    model["n_redrawn"] = model.get("n_redrawn", 0) + int(bad.sum())
    for _ in range(MAX_REDRAWS):
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        lat[idx] = draw(idx)
        bad[idx] = ends_inside(lat[idx], l[idx], X[idx])
    if bad.any():
        raise ValueError(f"{bad.sum()} draws still end inside the planes after {MAX_REDRAWS} redraws")
    out = {}
    for i, (s, y) in enumerate(to_profiles(lat, l, X).items()):
        p = model["planes"][s]
        if p.get("noise") in ("peak", "skew", "table", "bright"):
            xmax, L = longitudinal.from_latent(species_block(lat, i)[0], l)[:2]
            t = peak_position(X, xmax, L)
            t = np.where(np.isfinite(t), t, 0.0)
        if p.get("noise") == "bright":
            v = np.where(np.isfinite(y), y - np.log(10.0) * l[:, None], 0.0)
            y = y + bright_mean(p, t, v) + table_noise(p, t, ar1(len(l), y.shape[1], p["rho"], rng))
            out[s] = np.where(np.isfinite(y), np.exp(np.where(np.isfinite(y), y, 0.0)), 0.0)
            continue
        if p.get("noise") in ("skew", "table"):
            y = y + {"skew": skew_noise, "table": table_noise}[p["noise"]](p, t, ar1(len(l), y.shape[1], p["rho"], rng))
            out[s] = np.where(np.isfinite(y), np.exp(np.where(np.isfinite(y), y, 0.0)), 0.0)
            continue
        if p.get("noise") == "peak":
            bias, sd = at_peak(p, t)
        else:
            bias, sd = p["bias"], p["sd"]
        y = y + bias + sd * ar1(len(l), y.shape[1], p["rho"], rng)
        out[s] = np.where(np.isfinite(y), np.exp(np.where(np.isfinite(y), y, 0.0)), 0.0)
    return out, lat


def check(model, d, c):
    """On held-out showers: tail rate of each margin and the log-likelihood the correlations add."""
    t = class_rows(d, c)
    D = model["design"](t["l"], t["u"])
    Z = cp.normal_scores(model["margins"], D, t["X"])
    logc = {"all correlations": cp.gaussian_logc(Z, model["C"]),
            "within species only": cp.gaussian_logc(Z, cp.block_corr(model["C"], [BLOCK] * len(SPECIES_NAMES)))}
    return {"n_test": len(t["rows"]),
            "tail_rate": [round(cp.tail_rate(m, D, t["X"][:, j]), 4) for j, m in enumerate(model["margins"])],
            "logc_per_shower": {k: [float(v.mean()), float(v.std() / np.sqrt(len(v)))] for k, v in logc.items()}}

"""Draws the energy per plane for a primary: 5 numbers per species (profile + start depth) from the copula, the
profile curve at the plane depths, then a per-plane correction and AR(1) noise. Fitted per class."""
import numpy as np
from scipy import special

from modules.constants import SPECIES_NAMES
from symbolic import copula as cp, longitudinal

LOG10_E_MAX = 6.25         # the training showers end here
TOP = 6.125                # log10 E; the training file's top bin is under-filled and left out
MIN_PAIR = 1000
MAX_REDRAWS = 100
BLOCK = len(longitudinal.NAMES) + 1     # numbers per species: the profile, then the start depth


def load(path):
    """Conditions, plane data and profile numbers of every species from a fits file (fit_profiles.py). Clean =
    fitted, off the bounds and peaking within the planes."""
    z = np.load(path)
    keep = z["meta.log10_E"] <= LOG10_E_MAX
    d = {"meta": {k: z[f"meta.{k}"][keep] for k in ("log10_E", "cos_zenith", "labels")}, "X": z["X"][keep]}
    rng = np.random.default_rng(0)
    for s in SPECIES_NAMES:
        E = z[f"{s}.E"][keep]
        f = {k: z[f"{s}.{k}"][keep] for k in ("ln_n", "theta")}
        fl = longitudinal.flags(f, d["X"], E)
        d[s] = {"E": E, **f, "Z": longitudinal.to_latent(f, d["meta"]["log10_E"]),
                "start": longitudinal.start_depth(E, d["X"], rng), "clean": ~(fl["unfit"] | fl["bound"] | fl["outside"])}
    return d


def class_rows(d, c, top=TOP):
    """Showers of class c with a clean fit in every species: their joint numbers and conditions."""
    m = ((d["meta"]["labels"] == c) & (d["meta"]["log10_E"] < top)
         & np.all([d[s]["clean"] & np.isfinite(d[s]["start"]) for s in SPECIES_NAMES], 0))
    return {"rows": np.nonzero(m)[0],
            "X": np.hstack([np.column_stack([d[s]["Z"][m], d[s]["start"][m]]) for s in SPECIES_NAMES]),
            "l": d["meta"]["log10_E"][m], "u": d["meta"]["cos_zenith"][m]}


def plane_residuals(d, rows):
    """Per species: mean and sd per plane of log E_k minus the curve, and their plane-to-plane correlation."""
    X = d["X"][rows]
    out = {}
    for s in SPECIES_NAMES:
        th = d[s]["theta"][rows]
        with np.errstate(all="ignore"):
            curve = d[s]["ln_n"][rows][:, None] + longitudinal.shape(X, *[th[:, i, None] for i in range(3)])
            r = longitudinal.log_energy(d[s]["E"][rows]) - curve
        r[~np.isfinite(r)] = np.nan
        bias, sd = np.nanmean(r, 0), np.nanstd(r, 0)
        u = (r - bias) / sd
        rhos = [np.corrcoef(u[ok, k], u[ok, k + 1])[0, 1] for k in range(r.shape[1] - 1)
                for ok in [np.isfinite(u[:, k]) & np.isfinite(u[:, k + 1])] if ok.sum() >= MIN_PAIR]
        out[s] = {"bias": np.nan_to_num(bias), "sd": np.nan_to_num(sd), "rho": float(np.mean(rhos))}
    return out


def fit(d, c, top=TOP):
    """Fit the sampler for class c on the output of load(), on showers below log10 E = top."""
    t = class_rows(d, c, top)
    design = cp.Design(t["l"], t["u"])
    D = design(t["l"], t["u"])
    margins = [cp.Margin("jsu").fit(D, t["X"][:, j]) for j in range(t["X"].shape[1])]
    C = cp.corr(cp.normal_scores(margins, D, t["X"]))
    return {"design": design, "margins": margins, "C": C, "planes": plane_residuals(d, t["rows"]),
            "n_train": len(t["rows"])}


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
    for s, y in to_profiles(lat, l, X).items():
        p = model["planes"][s]
        y = y + p["bias"] + p["sd"] * ar1(len(l), y.shape[1], p["rho"], rng)
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

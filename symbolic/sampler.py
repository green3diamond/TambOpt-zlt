"""The sampler of per-plane energies: the profile numbers' conditional law, then the curve, then the planes.

Fitted per class on training showers clean in all three species with log10 E < TOP (the training file's top energy
bin is a depleted sub-sample and is left out):
  numbers  the four numbers of each species (m2.NAME latent) and its start depth (latent.start_depth: where the
           profile begins; planes before it are empty), a Johnson SU margin each, parameters linear in (log10 E,
           cos zenith) plus their product; one Gaussian correlation matrix of all 15 normal scores.
  planes   log E_k about the curve, per species and class from the training fits: a fixed correction per plane (the
           mean residual, what the curve misses on average) and noise with an sd per plane and one lag-1
           correlation along the planes (AR(1)).
A draw: z ~ N(0, C) -> margins -> each species' curve at the primary's plane depths -> + correction + noise; planes
before the species' start depth empty.
The numbers are drawn from their law truncated to physical profiles: a draw whose curve ends (R < 0, at xmax + L/|R|)
before the last plane is drawn again, as the fits never end inside the planes.
"""
import numpy as np
from scipy import special

from modules.constants import SPECIES_NAMES
from symbolic import copula as cp, families as fam, latent, m2

TOP = 6.125
MIN_PAIR = 1000            # showers with both planes for a lag-1 correlation to count
MAX_REDRAWS = 100
BLOCK = len(latent.NAMES[m2.NAME]) + 1     # numbers per species in the joint latent: the profile, then the start


def class_rows(d, c, top=TOP):
    """Rows (of m2.load_latent output) of class c, clean in every species, below `top`: joint latent and conditions."""
    m = ((d["meta"]["labels"] == c) & (d["meta"]["log10_E"] < top)
         & np.all([d[s]["clean"] & np.isfinite(d[s]["start"]) for s in SPECIES_NAMES], 0))
    return {"rows": np.nonzero(m)[0],
            "X": np.hstack([np.column_stack([d[s]["Z"][m], d[s]["start"][m]]) for s in SPECIES_NAMES]),
            "l": d["meta"]["log10_E"][m], "u": d["meta"]["cos_zenith"][m]}


def plane_residuals(fits, rows):
    """Per species: mean (correction) and sd per plane of log E_k minus the curve, and the lag-1 correlation of the
    standardized residuals averaged over neighbouring plane pairs. `fits` holds X and the species' E and fit arrays."""
    X = fits["X"][rows]
    out = {}
    for s in SPECIES_NAMES:
        E = fits[f"{s}.E"][rows]
        th = fits[f"{s}.{m2.NAME}.theta"][rows]
        with np.errstate(all="ignore"):
            curve = fits[f"{s}.{m2.NAME}.ln_n"][rows][:, None] + fam.FAMILIES[m2.NAME].shape(X, *[th[:, i, None] for i in range(3)])
            r = np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan) - curve
        r[~np.isfinite(r)] = np.nan
        bias, sd = np.nanmean(r, 0), np.nanstd(r, 0)
        u = (r - bias) / sd
        rhos = [np.corrcoef(u[ok, k], u[ok, k + 1])[0, 1] for k in range(r.shape[1] - 1)
                for ok in [np.isfinite(u[:, k]) & np.isfinite(u[:, k + 1])] if ok.sum() >= MIN_PAIR]
        out[s] = {"bias": np.nan_to_num(bias), "sd": np.nan_to_num(sd), "rho": float(np.mean(rhos))}
    return out


def fit(tr, fits, c):
    """Sampler for class c from m2.load_latent output `tr` and the matching fit arrays `fits` (same rows)."""
    t = class_rows(tr, c)
    design = cp.Design(t["l"], t["u"])
    D = design(t["l"], t["u"])
    margins = [cp.Margin("jsu").fit(D, t["X"][:, j]) for j in range(t["X"].shape[1])]
    C = cp.corr(cp.normal_scores(margins, D, t["X"]))
    return {"design": design, "margins": margins, "C": C, "planes": plane_residuals(fits, t["rows"]),
            "n_train": len(t["rows"])}


def species_block(Zlat, i):
    """Species i's profile numbers (n, 4) and start depth (n,) from a joint latent (n, BLOCK x species)."""
    b = Zlat[:, BLOCK * i:BLOCK * (i + 1)]
    return b[:, :-1], b[:, -1]


def to_profiles(Zlat, l, X):
    """Per-plane log energies (NaN where the curve vanishes or before the start depth) of each species."""
    out = {}
    for i, s in enumerate(SPECIES_NAMES):
        z, start = species_block(Zlat, i)
        xmax, L, R, ln_nmax = latent.from_latent(z, l)
        with np.errstate(all="ignore"):
            Y = ln_nmax[:, None] + fam.FAMILIES[m2.NAME].shape(X, xmax[:, None], L[:, None], R[:, None])
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
        xmax, L, R, _ = latent.from_latent(species_block(Zlat, i)[0], l)
        with np.errstate(divide="ignore", invalid="ignore"):
            bad |= (R < 0) & (xmax - L / R < X[:, -1])
    return bad


def sample(model, l, u, X, rng, planes=True, independent=False, return_latent=False):
    """Per-plane energies (n, K) per species, one draw per primary (l, u, plane depths X). planes=False: the curve
    only; independent=True: species drawn independently (block-diagonal C), as the pipeline does; return_latent=True:
    also the drawn joint latent (n, 15) (species_block, then latent.from_latent, gives xmax, L, R, ln_nmax)."""
    C = cp.block_corr(model["C"], [BLOCK] * len(SPECIES_NAMES)) if independent else model["C"]

    def draw(idx):
        D = model["design"](l[idx], u[idx])
        Z = rng.multivariate_normal(np.zeros(len(C)), C, len(idx))
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
        if planes:
            p = model["planes"][s]
            y = y + p["bias"] + p["sd"] * ar1(len(l), y.shape[1], p["rho"], rng)
        out[s] = np.where(np.isfinite(y), np.exp(np.where(np.isfinite(y), y, 0.0)), 0.0)
    return (out, lat) if return_latent else out


def check(model, te, c):
    """Held-out check on the test file: Johnson SU tail rate per number (0.0027 expected) and the log-likelihood per
    shower the correlations add (Gaussian copula against independent numbers and against independent species)."""
    t = class_rows(te, c)
    D = model["design"](t["l"], t["u"])
    Z = cp.normal_scores(model["margins"], D, t["X"])
    logc = {"all correlations": cp.gaussian_logc(Z, model["C"]),
            "within species only": cp.gaussian_logc(Z, cp.block_corr(model["C"], [BLOCK] * len(SPECIES_NAMES)))}
    return {"n_test": len(t["rows"]),
            "tail_rate": [round(cp.tail_rate(m, D, t["X"][:, j]), 4) for j, m in enumerate(model["margins"])],
            "logc_per_shower": {k: [float(v.mean()), float(v.std() / np.sqrt(len(v)))] for k, v in logc.items()}}

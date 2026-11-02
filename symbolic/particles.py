"""Particle laws (study_plan.md 2.4 step 2): what turns a plane's ring energies into particles, from Hamza's raw v2
showers (every particle of every plane, 1, 2, 5 m inner rings).

Stage A, tables (Hamza's env python, which has pyarrow; one shower per call): the raw particles in our frame about
the axis, rendered with the OLD data's caps (sources.CAP), so each shower gets the cells it would have had in
h5_files_v3. Saved per shower and species, per plane: energy of all particles, of those kept by the radius cut, and
stored after the cap; per plane and ring (Hamza's 23 rings of in-plane radius; our 16 rings are his first 16):
energy and particle count before and after the cut, Hamza's ring_tmin and ring_tmean, and the histogram of
log10(E / ring mean energy) over the ring's particles before the cut. Check (gate): our ring energies and counts
before the cut equal Hamza's plane_stats (counts exactly, energy to float32 summing).

Stage B, fit (tambo env): one row per plane with >= annuli.MIN_CELLS particles and stored energy; inputs
annuli.features with the longitudinal age from the plane's own species (Xmax fitted to its stored plane energies, as
fit_profiles). Laws per species and class, one form for every species:
  kept, stored     share of the plane's energy kept by the radius cut / stored after the cap (fractional logit)
  rings            shares of the plane's energy in all 23 rings before the cut (multinomial logit): the ring law
                   the draw generates particles from (the 2.3 law describes cut and capped planes, C0 below)
  width, shape     the plane-to-plane variety: each plane's ring profile is the rings law stretched in radius by lam
                   (width), or stretched and tilted by tau about its half-energy radius (shape: steeper or flatter),
                   both fitted per plane on the cumulative log-odds; each number ridge-linear in the features plus an
                   AR(1) residual along the shower's planes: [beta..., residual sd, lag-1 rho] per number
  mean_energy      ln mean particle energy per ring, before the cut (ridge-linear, one law per ring)
  lead, delay      log10(1 + delay / ns) of the ring's earliest and mean time, delay = time minus the axis light time
Check C0 (descriptive): the ring law of 2.3 (radial_bins) against the true ring shares inside OUTER, before and after
the cut. Energy spread within a ring: E = E_MIN + (ring mean - E_MIN) v above CORSIKA's tracking threshold E_MIN,
v ~ Burr XII (CDF 1 - (1 + (v / lam)^c)^-k, lam set by mean 1: a power-law tail, the ring mean kept exactly), c and k
per species, class and ring by binned likelihood over the plane-ring histograms (rings with fewer than MIN_PARTICLES
take the nearest fitted ring); reported, not gated: the suite judges it. Multinomial laws carry the same
ridge penalty as the linear ones.

Rule (fixed before the run), per law, species and class, on held-out showers (every second shower in log10 E order
within the class): mean residual within 2 se (clustered by shower) and R2 > 0 against the training mean; per-ring laws
are judged ring by ring, rings with fewer than MIN_SHOWERS held-out showers are not judged. Fitted on the other half;
the laws saved for the draw are refitted on every shower.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np

from symbolic import lateral, render, sources

RAW = "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_test_raw_v2"
FOLDER = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim/particles"
N_PLANE = 24
EDGES = np.array([0.0, 1, 2, 5, 10, 20, 30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600, 2300, 3200,
                  4500, 6400, 9000, np.inf])                # m, in-plane radius (Hamza's ring_edges, lower edges)
N_RING = len(EDGES) - 1
HIST = np.linspace(-6.0, 4.0, 101)                           # log10(E / ring mean); outside values go to the end bins
PDG = {"electron": (11, -11), "muon": (13, -13), "photon": (22,)}
CLASS = {"pdg_11": "EM", "pdg_211": "hadronic"}
E_TOL = 1e-4                                                 # relative, float32 sums in the preprocessing
OUTER = 16                                                   # rings inside 1100 m: the rings of the 2.3 ring law
RING_LAWS = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim/radial_bins"
MIN_SHOWERS = 5
RIDGE = 1.0
SHARE_EPS = 1e-12
SPREAD_RINGS = (4, 9, 14, 16)                                # 10-20, 100-140, 560-800, 1100-1600 m
MIN_PARTICLES = 1000
E_MIN = {"electron": 0.01 * 0.999, "muon": 0.5 * 0.999, "photon": 0.01 * 0.999}   # GeV kinetic, CORSIKA
LOG_LAM = np.arange(-3.0, 3.0001, 0.01)                        # log10 width grid of plane_widths
CUM_CLIP = 1e-6
# --emcut 0.01 --mucut 0.5 of the v2 runs (observed minima 0.0099990, 0.49995: float32), a hair below so no particle
# falls under it


def plane_index(name):
    return 23 if name == "particles" else int(name[len("particles"):]) - 1


def raw_planes(sim, pdg):
    """Per plane with data: k, local x, local y, kinetic energy, time of the species' particles."""
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(os.path.join(sim, "particles*"))):
        f = os.path.join(path, "particles.parquet")
        if not os.path.exists(f):
            continue
        t = pq.read_table(f, columns=["x", "y", "pdg", "time", "kinetic_energy"]).to_pandas()
        t = t[t["pdg"].isin(pdg)]
        if len(t):
            yield (plane_index(os.path.basename(path)), t["x"].to_numpy(np.float64), t["y"].to_numpy(np.float64),
                   np.maximum(t["kinetic_energy"].to_numpy(np.float32), 0.0), t["time"].to_numpy(np.float32))


def read_shower(sim, species):
    """Direction, energy, Hamza's plane_stats, and the particles: (n, 5) [plane k, across, along, energy, time] in
    our frame about the axis, plus their in-plane radius from the raw local coordinates."""
    h5 = glob.glob(os.path.join(sim, f"*_{sources.CORSIKA_NAME[species]}.h5"))[0]
    with h5py.File(h5, "r") as f:
        d, axes = f["directions"][0].astype(np.float64), f["plane_axes"][:]
        energy = float(f["energies"][0])
        stats = {k: f["plane_stats"][k][:] for k in f["plane_stats"]}
    d = d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    parts, radius = [], []
    for k, x, y, e, t in raw_planes(sim, PDG[species]):
        g = np.column_stack([x, y, np.zeros_like(x)]) @ axes[k][:, :2]     # ground offset from the axis point
        parts.append(np.column_stack([np.full(len(x), k), g @ perp, g @ h / c, e, t]))
        radius.append(np.hypot(x, y))
    if not parts:
        return d, energy, stats, np.zeros((0, 5)), np.zeros(0)
    return d, energy, stats, np.concatenate(parts), np.concatenate(radius)


def tables(p, r, d, n_max):
    """Per-plane and per-ring tables of one shower and species (see the module docstring)."""
    k = p[:, 0].astype(int)
    e = np.maximum(p[:, 3], 0.0)
    xy = render.to_ground(k, p[:, 1], p[:, 2], d)
    keep = render.radius_cut(k, xy)
    cells = render.cluster(k[keep], xy[keep], e[keep], p[keep, 4])          # = render.render before the cap
    stored = render.cap(cells, n_max)
    flat = k * N_RING + np.searchsorted(EDGES, r, side="right") - 1
    size = N_PLANE * N_RING

    def per_ring(m, w=None):
        return np.bincount(flat[m], None if w is None else w[m], size).reshape(N_PLANE, N_RING)

    every = np.ones(len(k), bool)
    out = {"E_total": np.bincount(k, e, N_PLANE), "E_cut": np.bincount(k[keep], e[keep], N_PLANE),
           "E_stored": np.bincount(stored[:, 2].astype(int), stored[:, 3], N_PLANE), "n_cells": np.array(len(cells)),
           "ring_energy": per_ring(every, e), "ring_count": per_ring(every), "ring_energy_cut": per_ring(keep, e),
           "ring_count_cut": per_ring(keep)}
    mean = out["ring_energy"].ravel() / np.maximum(out["ring_count"].ravel(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log10(e / mean[flat])
    u[np.isnan(u)] = -np.inf                                                # 0 / 0: a ring of zero-energy particles
    b = np.clip(np.searchsorted(HIST, u, side="right") - 1, 0, len(HIST) - 2)
    out["hist"] = np.bincount(flat * (len(HIST) - 1) + b, None, size * (len(HIST) - 1)).reshape(
        N_PLANE, N_RING, len(HIST) - 1).astype(np.int32)
    return out


def check(tab, stats):
    """Our ring energies and counts before the cut against Hamza's plane_stats."""
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30), initial=0.0))
    res = {"ring count diff": int(np.abs(tab["ring_count"] - stats["ring_count"]).max()),
           "ring energy rel": rel(tab["ring_energy"], stats["ring_energy"]),
           "plane energy rel": rel(tab["E_total"], stats["energy"]),
           "plane count diff": int(np.abs(tab["ring_count"].sum(1) - stats["count"]).max())}
    res["pass"] = bool(res["ring count diff"] == 0 == res["plane count diff"] and res["ring energy rel"] <= E_TOL
                       and res["plane energy rel"] <= E_TOL)
    return res


def stage_tables(sim, out):
    """Tables of one shower (all species) -> <out>/tables/<class>_<sim>.npz; one JSON line per species."""
    cls = CLASS[os.path.relpath(sim, RAW).split(os.sep)[0]]
    name = os.path.basename(os.path.normpath(sim))
    save = {"git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "sim": sim, "class": cls, "edges": EDGES, "hist_edges": HIST}
    for s in PDG:
        d, energy, stats, p, r = read_shower(sim, s)
        tab = tables(p, r, d, sources.CAP[s])
        tab["ring_tmin"], tab["ring_tmean"] = stats["ring_tmin"], stats["ring_tmean"]
        save.update({"direction": d, "log10E": np.log10(energy)}, **{f"{s}.{k}": v for k, v in tab.items()})
        res = check(tab, stats)
        print(json.dumps({"sim": name, "class": cls, "species": s, "log10E": round(float(np.log10(energy)), 3),
                          "cos z": round(float(d[2]), 4), "particles": len(p), "cells before cap": int(tab["n_cells"]),
                          "kept": round(float(tab["E_cut"].sum() / tab["E_total"].sum()), 4),
                          "stored": round(float(tab["E_stored"].sum() / tab["E_total"].sum()), 4), **res}), flush=True)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "tables", f"{cls}_{name}.npz"), **save)


def plane_rows(tabs):
    """Per species: one row per usable plane (shower i, class, plane k, features x, its tables)."""
    from symbolic import annuli, atmosphere, generator, longitudinal
    rows = {s: [] for s in PDG}
    k = np.arange(N_PLANE)
    for i, t in enumerate(tabs):
        cz = float(t["direction"][2])
        X = atmosphere.plane_depths(np.degrees(np.arccos([cz])), N_PLANE)[0]
        r_ref = np.array([lateral.r_m_ref(p, cz) for p in k], np.float64)
        t_light = (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT
        for s in PDG:
            E = t[f"{s}.E_stored"]
            fit = longitudinal.fit_one(X, longitudinal.log_energy(E))
            age = generator.age(X, fit[1][0] if fit else np.nan)
            ok = (t[f"{s}.ring_count"].sum(1) >= annuli.MIN_CELLS) & (E > 0)
            x = annuli.features(age, np.full(N_PLANE, float(t["log10E"])), np.full(N_PLANE, cz),
                                np.log(np.where(E > 0, E, 1.0)), r_ref)
            lead, mean = ((np.maximum(t[f"{s}.ring_{w}"] - t_light[:, None], 0.0) * 1e9) for w in ("tmin", "tmean"))
            r = {"i": np.full(N_PLANE, i), "cls": np.full(N_PLANE, str(t["class"])), "k": k, "x": x,
                 "lead": np.log10(1 + lead), "delay": np.log10(1 + mean),
                 **{w: t[f"{s}.{w}"] for w in ("E_total", "E_cut", "E_stored", "ring_energy", "ring_count",
                                               "ring_energy_cut", "hist")}}
            rows[s].append({w: v[ok] for w, v in r.items()})
    return {s: {w: np.concatenate([r[w] for r in rs]) for w in rs[0]} for s, rs in rows.items()}


def targets(r):
    """The law targets of a species' rows: name -> (kind, y, usable mask); y per plane, or per plane and ring."""
    e, n = r["ring_energy"], r["ring_count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_e = np.log(e / n)
    every = np.ones(len(e), bool)
    return {"kept": ("share", r["E_cut"] / r["E_total"], every),
            "stored": ("share", r["E_stored"] / r["E_total"], every),
            "rings": ("split", e / r["E_total"][:, None], every),
            "mean_energy": ("ring", mean_e, (n > 0) & np.isfinite(mean_e)),
            "lead": ("ring", r["lead"], np.isfinite(r["lead"]) & (n > 0)),
            "delay": ("ring", r["delay"], np.isfinite(r["delay"]) & (n > 0))}


def ridge(x, y, alpha=RIDGE):
    """Ridge-linear law (intercept not penalised) -> coefficients."""
    pen = alpha * np.eye(x.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + pen, x.T @ y)


def fit_multinomial(x, w, alpha=RIDGE):
    """Multinomial logit of shares w (rows on the simplex), first column the reference; generator.fit_shares plus the
    ridge penalty (intercept free), so a ring that is empty in most planes cannot run off."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
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


def fit_law(kind, x, y):
    """Coefficients of one law: share / split by (fractional) multinomial logit, ring targets by ridge."""
    if kind == "ring":
        return ridge(x, y)
    w = np.column_stack([y, 1 - y]) if kind == "share" else y
    return fit_multinomial(x, np.clip(w, 0.0, 1.0))


def predict(kind, x, B):
    from scipy.special import softmax
    if kind == "ring":
        return x @ B
    p = softmax(x @ B, axis=1)
    return p[:, 0] if kind == "share" else p


def judge(y, p, y_train, unit):
    """Held-out check of one law: mean residual against its shower-clustered se, and R2 against the training mean."""
    r = y - p
    groups = np.unique(unit)
    out = {"planes": int(len(y)), "showers": int(len(groups))}
    if len(groups) < MIN_SHOWERS:
        return {**out, "pass": None}
    bias = r.mean()
    se = np.sqrt(sum((r[unit == g] - bias).sum() ** 2 for g in groups)) / len(r)
    r2 = 1 - np.sum(r ** 2) / max(np.sum((y - y_train) ** 2), 1e-300)
    return {**out, "bias": float(bias), "se": float(se), "R2": float(r2),
            "pass": bool(abs(bias) <= 2 * se and r2 > 0)}


def split_showers(i, cls, log10e):
    """Held-out mask: every second shower in log10 E order within its class."""
    test = np.zeros(len(i), bool)
    for c in np.unique(cls):
        ids = np.unique(i[cls == c])
        order = ids[np.argsort(log10e[ids], kind="stable")]
        test |= np.isin(i, order[1::2])
    return test


def run_law(kind, x, y, ok, test, unit):
    """Fit on the training showers, judge on the held-out ones, refit on all -> (results, coefficients)."""
    if kind != "ring":
        tr, te = ok & ~test, ok & test
        B = fit_law(kind, x[tr], y[tr])
        p = predict(kind, x[te], B)
        if kind == "share":
            res = judge(y[te], p, y[tr].mean(), unit[te])
        else:
            res = {f"ring {j}": judge(y[te, j], p[:, j], y[tr, j].mean(), unit[te]) for j in range(y.shape[1])}
        return res, fit_law(kind, x[ok], y[ok])
    res, B = {}, np.full((y.shape[1], x.shape[1]), np.nan)
    for j in range(y.shape[1]):
        tr, te, m = ok[:, j] & ~test, ok[:, j] & test, ok[:, j]
        if len(np.unique(unit[tr])) < MIN_SHOWERS or len(np.unique(unit[te])) < MIN_SHOWERS:
            res[f"ring {j}"] = {"showers": int(len(np.unique(unit[te]))), "pass": None}
            continue
        Bj = ridge(x[tr], y[tr, j])
        res[f"ring {j}"] = judge(y[te, j], x[te] @ Bj, y[tr, j].mean(), unit[te])
        B[j] = ridge(x[m], y[m, j])
    return res, B


def cum_logodds(q):
    """Log-odds of the cumulative shares at the interior ring edges (last axis: rings -> N_RING - 1 edges)."""
    c = np.clip(np.cumsum(q, -1)[..., :-1], CUM_CLIP, 1 - CUM_CLIP)
    return np.log(c / (1 - c))


def stretch_logodds(y, log_lam):
    """Cumulative log-odds y (edges) of one profile stretched in radius by each 10^log_lam -> (len(log_lam), edges):
    linear in log r between the edges, extended linearly past both ends."""
    lr = np.log10(EDGES[1:-1])
    at = lr[None, :] - np.atleast_1d(log_lam)[:, None]
    v = np.interp(at, lr, y)
    v = np.where(at < lr[0], y[0] + (y[1] - y[0]) / (lr[1] - lr[0]) * (at - lr[0]), v)
    return np.where(at > lr[-1], y[-1] + (y[-1] - y[-2]) / (lr[-1] - lr[-2]) * (at - lr[-1]), v)


def tilt_axis(Y):
    """log10 r minus each profile's half-energy radius (where its cumulative log-odds crosses 0), rows of Y."""
    lr = np.log10(EDGES[1:-1])
    j = np.clip(np.argmax(Y >= 0, 1), 1, len(lr) - 1)
    j = np.where((Y >= 0).any(1), j, len(lr) - 1)
    y0, y1 = Y[np.arange(len(Y)), j - 1], Y[np.arange(len(Y)), j]
    f = np.clip(np.divide(-y0, y1 - y0, out=np.zeros(len(Y)), where=y1 > y0), 0.0, 1.0)
    return lr[None, :] - (lr[j - 1] + f * (lr[j] - lr[j - 1]))[:, None]


def stretch(q, log_lam, tau=0.0):
    """Ring shares of profile q stretched in radius by 10^log_lam and tilted by tau about its half-energy radius
    (tau x (log10 r - log10 r_half) added to the cumulative log-odds: steeper or flatter, same half-energy radius)."""
    Y = stretch_logodds(cum_logodds(q), log_lam)
    c = np.maximum.accumulate(1 / (1 + np.exp(-(Y + tau * tilt_axis(Y))[0])))
    return np.diff(np.r_[0.0, c, 1.0])


def plane_shapes(q_true, q_law, tilt=True):
    """Per plane, the log10 stretch (grid LOG_LAM) of its law profile that best matches the true profile on the
    cumulative log-odds, then the tilt of what the stretch leaves (least squares) -> (log10 lam, tau). In sequence,
    because fitted jointly the two trade off almost perfectly (residual correlation 0.82-0.99) and wander along that
    ridge; tilt=False gives tau = 0."""
    lam, tau = np.empty(len(q_true)), np.zeros(len(q_true))
    for p, (yl, yt) in enumerate(zip(cum_logodds(q_law), cum_logodds(q_true))):
        Y = stretch_logodds(yl, LOG_LAM)
        b = ((yt[None, :] - Y) ** 2).sum(1).argmin()
        lam[p] = LOG_LAM[b]
        if tilt:
            g = tilt_axis(Y[b:b + 1])[0]
            tau[p] = g @ (yt - Y[b]) / max(g @ g, 1e-12)
    return lam, tau


def ar1_law(x, v, unit, k, test):
    """A per-plane number ridge-linear in the features, AR(1) residual along a shower's planes -> (law on every plane:
    [beta..., sd, rho], results: held-out R2 of the training-half law, residual sd, rho, mean / sd of the number)."""
    e = v - x @ ridge(x[~test], v[~test])
    r2 = 1 - np.var(e[test]) / np.var(v[test])
    beta = ridge(x, v)
    e = v - x @ beta
    by = dict(zip(zip(unit, k), e))
    pairs = np.array([(a, by[(u, p + 1)]) for (u, p), a in by.items() if (u, p + 1) in by])
    rho = float(np.corrcoef(pairs.T)[0, 1])
    return np.r_[beta, e.std(), rho], {"held-out R2": float(r2), "residual sd": float(e.std()), "rho": rho,
                                       "mean / sd": [float(v.mean()), float(v.std())]}


def ring_check(x, r, B):
    """C0: the 2.3 ring law against the true shares inside OUTER, before and after the cut."""
    from scipy.special import softmax
    law = softmax(x @ B, axis=1)
    out, pooled = {}, {}
    for name, e in (("before cut", r["ring_energy"][:, :OUTER]), ("after cut", r["ring_energy_cut"][:, :OUTER])):
        tot = e.sum(1)
        q = e[tot > 0] / tot[tot > 0, None]
        kl = np.sum(q * (np.log(q + SHARE_EPS) - np.log(law[tot > 0] + SHARE_EPS)), 1)
        out[f"median KL data || law, {name}"] = float(np.median(kl))
        pooled[name] = q.mean(0)
        pooled.setdefault("law", law[tot > 0].mean(0))
    out["pooled shares"] = {k: v.round(4).tolist() for k, v in pooled.items()}
    return out, pooled


def burr_lam(c, k):
    """Burr XII scale that gives mean 1."""
    from scipy.special import betaln
    return np.exp(-np.log(k) - betaln(k - 1 / c, 1 + 1 / c))


def burr_cdf(v, c, k):
    """CDF of v ~ Burr XII (c, k) scaled to mean 1; 0 for v <= 0."""
    v = np.asarray(v, np.float64)
    with np.errstate(divide="ignore"):
        z = c * (np.log(np.where(v > 0, v, 1.0)) - np.log(burr_lam(c, k)))
    return np.where(v > 0, -np.expm1(-k * np.logaddexp(0.0, z)), 0.0)


def burr_params(a):
    """Unconstrained (a0, a1) -> (c, k) with c k > 1 (finite mean)."""
    c = np.exp(a[0])
    return c, (1 + np.exp(a[1])) / c


def burr_sample(rng, c, k, n):
    """n draws of v (mean 1)."""
    return burr_lam(c, k) * np.expm1(-np.log1p(-rng.uniform(size=n)) / k) ** (1 / c)


def energies(rng, c, k, mean, e_min, n):
    """n particle energies of a ring: E_min + (mean - E_min) v, v ~ Burr XII (c, k) of mean 1."""
    return e_min + max(mean - e_min, 0.0) * burr_sample(rng, c, k, n)


def v_edges(mean, e_min):
    """Edges of HIST (log10 E / ring mean) in v = (E - E_min) / (mean - E_min), one row per plane-ring; open ends."""
    e = mean[:, None] * 10.0 ** HIST[None, :]
    v = (e - e_min) / (mean - e_min)[:, None]
    v[:, 0], v[:, -1] = -np.inf, np.inf
    return v


def fit_burr(h, mean, e_min):
    """(c, k) by binned likelihood over plane-rings (histograms h over HIST, ring means) -> (c, k), W1 in log10 E
    between pooled data and law, pooled law shares."""
    from scipy.optimize import minimize
    v = v_edges(mean, e_min)
    w = h / h.sum()

    def probs(a):
        F = burr_cdf(np.where(np.isfinite(v), v, 0.0), *burr_params(a))
        F[:, -1] = 1.0
        return np.diff(F, axis=1)

    nll = lambda a: -np.sum(w * np.log(np.maximum(probs(a), 1e-300)))
    best = min((minimize(nll, a0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
                for a0 in ([np.log(3.0), 0.0], [np.log(1.5), 1.0], [np.log(8.0), -1.0])), key=lambda s: s.fun)
    p = (probs(best.x) * h.sum(1, keepdims=True)).sum(0) / h.sum()
    q = w.sum(0)
    w1 = float(np.sum(np.abs(np.cumsum(q) - np.cumsum(p))[:-1] * np.diff(HIST)[1:]))
    return burr_params(best.x), w1, p


def spread(hist, mean, e_min):
    """Burr XII (c, k) above E_min per ring from the plane-ring histograms (rings with fewer than MIN_PARTICLES take
    the nearest fitted ring) -> ((N_RING, 2) array, per fitted ring: particles, c, k, W1, data mean and sd of log10
    E / ring mean, pooled densities for the figure)."""
    mid = 0.5 * (HIST[1:] + HIST[:-1])
    ck, out = np.full((N_RING, 2), np.nan), {}
    for j in range(N_RING):
        h, m = hist[:, j].astype(np.float64), mean[:, j]
        use = (h.sum(1) > 0) & (m > e_min * (1 + 1e-6))
        if h[use].sum() < MIN_PARTICLES:
            continue
        (c, k), w1, p = fit_burr(h[use], m[use], e_min)
        ck[j] = c, k
        q = h[use].sum(0) / h[use].sum()
        mu = float(q @ mid)
        out[j] = {"particles": int(h[use].sum()), "c": float(c), "k": float(k), "W1 data vs law": w1, "mean": mu,
                  "sd": float(np.sqrt(q @ (mid - mu) ** 2)), "data": q / np.diff(HIST), "law": p / np.diff(HIST)}
    fitted = np.flatnonzero(np.isfinite(ck[:, 0]))
    for j in range(N_RING):
        if not np.isfinite(ck[j, 0]):
            ck[j] = ck[fitted[np.argmin(np.abs(fitted - j))]]
    return ck, out




def fig_rings(pooled):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True, sharey=True)
    lo = EDGES[:OUTER]
    for (s, cn), p in pooled.items():
        ax = axes[list(CLASS.values()).index(cn), list(PDG).index(s)]
        for name, col, ls, label in (("before cut", "black", "-", "data, before cut"),
                                     ("after cut", "#2a78d6", "--", "data, after cut"), ("law", "#eb6834", "-", "2.3 law"),
                                     ("v2 law", "#1b9e77", ":", "v2 ring law")):
            ax.step(lo + 1, p[name], where="post", color=col, ls=ls, lw=1.4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{s}s, {cn}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("1 + ring inner radius [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean share inside 1100 m")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_spread(spreads):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    mid = 0.5 * (HIST[1:] + HIST[:-1])
    cols = dict(zip(SPREAD_RINGS, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")))
    for (s, cn), sp in spreads.items():
        ax = axes[list(CLASS.values()).index(cn), list(PDG).index(s)]
        for j in SPREAD_RINGS:
            if j in sp:
                ax.plot(mid, sp[j]["data"], color=cols[j], lw=1.2, label=f"{EDGES[j]:g}-{EDGES[j + 1]:g} m")
                ax.plot(mid, sp[j]["law"], color=cols[j], lw=1.0, ls="--")
        ax.set_ylim(1e-6, None)
        ax.set_yscale("log")
        ax.set_title(f"{s}s, {cn}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("log10(E / ring mean energy)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def stage_fit(out, args, log):
    from symbolic import logs
    tabs = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(args.out, "tables", "*.npz")))]
    log.info("%d shower tables", len(tabs))
    log10e = np.array([float(t["log10E"]) for t in tabs])
    data = plane_rows(tabs)
    res, laws, pooled, spreads = {"rule": __doc__.split("Rule")[1].strip(), "c0": {}, "laws": {}, "spread": {}}, {}, {}, {}
    for s, r in data.items():
        test = split_showers(r["i"], r["cls"], log10e)
        tg = targets(r)
        for cn in CLASS.values():
            m = r["cls"] == cn
            B = np.load(os.path.join(RING_LAWS, f"conv_{s}", "ring_laws.npz"))[f"{s}.{cn}.rings"]
            res["c0"][f"{s} {cn}"], pooled[(s, cn)] = ring_check(r["x"][m], {w: v[m] for w, v in r.items()}, B)
            c0 = res["c0"][f"{s} {cn}"]
            log.info("C0 %-8s %-8s planes %4d | median KL law: before cut %.4f after cut %.4f", s, cn, m.sum(),
                     c0["median KL data || law, before cut"], c0["median KL data || law, after cut"])
            mean = r["ring_energy"][m] / np.maximum(r["ring_count"][m], 1)
            laws[f"{s}.{cn}.spread"], spreads[(s, cn)] = spread(r["hist"][m], mean, E_MIN[s])
            res["spread"][f"{s} {cn}"] = {f"{EDGES[j]:g}-{EDGES[j + 1]:g} m": {w: v for w, v in q.items()
                                                                              if w not in ("data", "law")}
                                          for j, q in spreads[(s, cn)].items()}
            w1 = [q["W1 data vs law"] for q in spreads[(s, cn)].values()]
            log.info("spread %-8s %-8s Burr XII on %d rings | W1 in log10 E data vs law median %.4f max %.4f | c, k at "
                     "%s", s, cn, len(w1), np.median(w1), np.max(w1), " ".join(
                         f"{EDGES[j]:g}m {laws[f'{s}.{cn}.spread'][j][0]:.2f},{laws[f'{s}.{cn}.spread'][j][1]:.2f}"
                         for j in SPREAD_RINGS))
            for name, (kind, y, ok) in tg.items():
                okm = ok & (m if ok.ndim == 1 else m[:, None])
                rr, Bl = run_law(kind, r["x"], y, okm, test, r["i"])
                laws[f"{s}.{cn}.{name}"] = Bl
                res["laws"][f"{s} {cn} {name}"] = rr
                if name == "rings":
                    q = predict(kind, r["x"][m], Bl)[:, :OUTER]
                    pooled[(s, cn)]["v2 law"] = (q / q.sum(1, keepdims=True)).mean(0)
                    true, law_q = r["ring_energy"][m] / r["E_total"][m, None], predict(kind, r["x"][m], Bl)
                    rows = (r["x"][m], r["i"][m], r["k"][m], test[m])
                    lam, _ = plane_shapes(true, law_q, tilt=False)
                    laws[f"{s}.{cn}.width"], res["laws"][f"{s} {cn} width"] = ar1_law(rows[0], lam, *rows[1:])
                    lam, tau = plane_shapes(true, law_q)
                    (w_l, r_l), (w_t, r_t) = ar1_law(rows[0], lam, *rows[1:]), ar1_law(rows[0], tau, *rows[1:])
                    e_l, e_t = lam - rows[0] @ w_l[:-2], tau - rows[0] @ w_t[:-2]
                    laws[f"{s}.{cn}.shape"] = np.r_[w_l, w_t]
                    res["laws"][f"{s} {cn} shape"] = {"width": r_l, "tilt": r_t,
                                                      "residual corr width-tilt": float(np.corrcoef(e_l, e_t)[0, 1])}
                    for w in ("width", "shape"):
                        log.info("law %-8s %-8s %-12s %s", s, cn, w, json.dumps(res["laws"][f"{s} {cn} {w}"]))
                if kind == "share":
                    log.info("law %-8s %-8s %-12s planes %4d showers %2d | bias %+.4f se %.4f R2 %.3f | %s", s, cn,
                             name, rr["planes"], rr["showers"], rr.get("bias", np.nan), rr.get("se", np.nan),
                             rr.get("R2", np.nan), {True: "PASS", False: "FAIL", None: "not judged"}[rr["pass"]])
                else:
                    v = [q["pass"] for q in rr.values()]
                    log.info("law %-8s %-8s %-12s rings pass %d / judged %d (not judged %d) | worst R2 %.3f", s, cn,
                             name, sum(q is True for q in v), sum(q is not None for q in v), sum(q is None for q in v),
                             min((q["R2"] for q in rr.values() if q["pass"] is not None), default=np.nan))
    np.savez(os.path.join(out, "particle_laws.npz"), edges=EDGES, features="annuli.features", **laws)
    logs.save_fig(fig_rings(pooled), out, "c0_ring_law", log)
    logs.save_fig(fig_spread(spreads), out, "energy_spread", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("tables", help="stage A, one shower directory per call")
    a.add_argument("sim")
    a.add_argument("--out", default=FOLDER)
    b = sub.add_parser("fit", help="stage B, laws from every table under <out>/tables")
    b.add_argument("--out", default=FOLDER)
    b.add_argument("--name", default="fit", help="output folder under <out>")
    args = ap.parse_args()
    if args.stage == "tables":
        stage_tables(args.sim, args.out)
    else:
        from symbolic import logs
        out, log = logs.start(args.name, args)
        logs.save(out, stage_fit(out, args, log), args, log)


if __name__ == "__main__":
    main()

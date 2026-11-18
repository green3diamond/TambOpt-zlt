"""One row per plane of the shower tables (data.py): the inputs every in-plane law takes, and the plane's ring sums,
times and particle-energy histograms the laws are fitted to. Rows are planes with at least MIN_CELLS particles and
energy (stored energy where a table has it)."""
import os
from multiprocessing import Pool

import numpy as np

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, lateral, longitudinal

C_LIGHT = 299792458.0
MIN_CELLS = 5
FIELDS = ("E_total", "ring_energy", "ring_count", "hist")


def features(age, log10_e, cos_z, ln_ek, r_ref):
    """Inputs of every in-plane law: 1, age, age^2, log10 E, |cos z|, ln |cos z|, ln plane energy, Moliere radius."""
    c = np.maximum(np.abs(cos_z), 1e-3)
    return np.column_stack([np.ones_like(age), age, age ** 2, log10_e - 5.5, c, np.log(c), ln_ek - 10.0,
                            np.log(r_ref / 100.0)])


def plane_inputs(x, k):
    """The inputs of the formula laws after ln r: age, ln plane number, log10 E, |cos z|, ln plane energy, ln Moliere
    radius, from feature rows x of planes k (0-based)."""
    return np.column_stack([x[:, 1], np.log(k + 1.0), x[:, 3], x[:, 4], x[:, 6], x[:, 7]])


def plane_rows(tabs, offset=0):
    """Per species: shower index i (from offset), class, plane k, features x (from the plane's total energy, age from
    the Xmax fitted to it), log10(1 + ns) of each ring's earliest and mean delay behind light on the axis, and FIELDS."""
    rows = {s: [] for s in SPECIES_NAMES}
    k = np.arange(N_PLANES)
    for i, t in enumerate(tabs):
        cz = float(t["direction"][2])
        X = atmosphere.plane_depths(np.degrees(np.arccos([cz])), N_PLANES)[0]
        r_ref = np.array([lateral.r_m_ref(p, cz) for p in k], np.float64)
        t_light = (k + 1) * atmosphere.PLANE_SPACING / C_LIGHT
        for s in SPECIES_NAMES:
            E = t[f"{s}.E_total"]
            fit = longitudinal.fit_one(X, longitudinal.log_energy(E))
            age = longitudinal.age(X, fit[1][0] if fit else np.nan)
            x = features(age, np.full(N_PLANES, float(t["log10E"])), np.full(N_PLANES, cz), np.log(np.where(E > 0, E, 1.0)), r_ref)
            lit = t[f"{s}.E_stored"] if f"{s}.E_stored" in t else E
            ok = (t[f"{s}.ring_count"].sum(1) >= MIN_CELLS) & (lit > 0)
            lead, mean = ((np.maximum(t[f"{s}.ring_{w}"] - t_light[:, None], 0.0) * 1e9) for w in ("tmin", "tmean"))
            r = {"i": np.full(N_PLANES, i + offset), "cls": np.full(N_PLANES, str(t["class"])), "k": k, "x": x,
                 "lead": np.log10(1 + lead), "delay": np.log10(1 + mean), **{w: t[f"{s}.{w}"] for w in FIELDS if f"{s}.{w}" in t}}
            rows[s].append({w: v[ok] for w, v in r.items()})
    return {s: {w: np.concatenate([r[w] for r in rs]) for w in rs[0]} for s, rs in rows.items()}


def _chunk(job):
    files, offset = job
    return plane_rows([dict(np.load(f)) for f in files], offset)


def build(files, workers=1, chunk=200):
    """plane_rows of every table file, in parallel chunks, with the shower list, log10 E and direction."""
    with Pool(workers) as pool:
        parts = pool.map(_chunk, [(files[a:a + chunk], a) for a in range(0, len(files), chunk)])
    meta = [np.load(f) for f in files]
    out = {f"{s}.{w}": np.concatenate([p[s][w] for p in parts]) for s in SPECIES_NAMES for w in parts[0][s]}
    out.update(showers=np.array([os.path.basename(f) for f in files]), log10E=np.array([float(t["log10E"]) for t in meta]),
               direction=np.array([t["direction"] for t in meta]))
    return out


def load(path):
    """A rows file -> ({species: {field: array}}, showers)."""
    z = np.load(path)
    data = {s: {} for s in SPECIES_NAMES}
    for key in z.files:
        s, _, w = key.partition(".")
        if s in data:
            data[s][w] = z[key]
    return data, z["showers"].tolist()

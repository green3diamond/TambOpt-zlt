"""CORSIKA showers on disk, one folder per simulation: per species an h5 file (plane_stats, stored cells) and, for the
raw runs, every particle per plane in parquet. Read into the shower tables the laws are fitted on, and into the stored
cells the judge and the detector check compare to."""
import glob
import json
import os
import subprocess

import h5py
import numpy as np

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import lateral, render

H5_NAME = {"electron": "electrons", "muon": "muons", "photon": "photons"}
PDG = {"electron": (11, -11), "muon": (13, -13), "photon": (22,)}
CAP = {"electron": 2000, "muon": 28000, "photon": 6000}       # cells the per-simulation preprocessing stores
CAP_V3 = {"electron": 4096, "muon": 25088, "photon": 8064}    # the older preprocessing's caps (raw tables' E_stored)
CLASS = {0: "EM", 1: "hadronic"}
LABEL = {"EM": 0, "hadronic": 1}
PRIMARY_CLASS = {"pdg_11": "EM", "pdg_-11": "EM", "pdg_111": "EM", "pdg_211": "hadronic", "pdg_-211": "hadronic"}
EDGES = np.array([0.0, 1, 2, 5, 10, 20, 30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600, 2300, 3200,
                  4500, 6400, 9000, np.inf])                # m, lower edges of the in-plane rings of plane_stats
N_RING = len(EDGES) - 1
HIST = np.linspace(-6.0, 4.0, 101)                           # log10(E / ring mean); outside values go to the end bins
E_TOL = 1e-4                                                 # relative, float32 sums in the preprocessing

if tuple(CAP) != tuple(SPECIES_NAMES):
    raise RuntimeError(f"species order {tuple(CAP)} differs from constants {SPECIES_NAMES}")


def class_of(sim):
    """EM or hadronic, from the pdg_ folder the simulation sits in."""
    return PRIMARY_CLASS[next(p for p in os.path.normpath(sim).split(os.sep) if p.startswith("pdg_"))]


def h5_path(sim, s):
    path = glob.glob(os.path.join(sim, f"*_{H5_NAME[s]}.h5"))
    if len(path) != 1:
        raise ValueError(f"{sim}: {len(path)} {s} h5 files")
    return path[0]


def git_head():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def summary_table(sim):
    """Table of one shower from plane_stats alone: plane and ring energies, counts and times, direction, log10 E.
    -> (table, None), or (None, reason) when the shower is unusable (no .done marker, missing or broken files)."""
    if not os.path.exists(os.path.join(sim, ".done")):
        return None, "no .done"
    tab = {}
    for s in SPECIES_NAMES:
        try:
            with h5py.File(h5_path(sim, s), "r") as f:
                d, energy = f["directions"][0].astype(np.float64), float(f["energies"][0])
                st = {k: f["plane_stats"][k][:] for k in ("energy", "ring_energy", "ring_count", "ring_tmin", "ring_tmean")}
        except (OSError, KeyError, ValueError) as exc:
            return None, f"{s} unreadable: {exc!r}"[:200]
        if not all(np.isfinite(st[k]).all() for k in ("energy", "ring_energy", "ring_count")):
            return None, f"{s} non-finite energies or counts"
        tab.update({f"{s}.E_total": st["energy"].astype(np.float64), f"{s}.ring_energy": st["ring_energy"].astype(np.float64),
                    f"{s}.ring_count": st["ring_count"].astype(np.float64), f"{s}.ring_tmin": st["ring_tmin"],
                    f"{s}.ring_tmean": st["ring_tmean"]})
    tab.update({"direction": d / np.linalg.norm(d), "log10E": np.log10(energy)})
    return tab, None


def raw_particles(sim, species):
    """Every particle of one species: (n, 5) [plane k, across, along, energy, time] about the axis, their in-plane
    radius from the raw local coordinates, the direction, primary energy and plane_stats."""
    import pyarrow.parquet as pq
    with h5py.File(h5_path(sim, species), "r") as f:
        d, axes = f["directions"][0].astype(np.float64), f["plane_axes"][:]
        energy = float(f["energies"][0])
        stats = {k: f["plane_stats"][k][:] for k in f["plane_stats"]}
    d = d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    parts, radius = [], []
    for path in sorted(glob.glob(os.path.join(sim, "particles*", "particles.parquet"))):
        name = os.path.basename(os.path.dirname(path))
        k = 23 if name == "particles" else int(name[len("particles"):]) - 1
        t = pq.read_table(path, columns=["x", "y", "pdg", "time", "kinetic_energy"]).to_pandas()
        t = t[t["pdg"].isin(PDG[species])]
        if not len(t):
            continue
        x, y = t["x"].to_numpy(np.float64), t["y"].to_numpy(np.float64)
        g = np.column_stack([x, y, np.zeros_like(x)]) @ axes[k][:, :2]       # ground offset from the axis point
        e = np.maximum(t["kinetic_energy"].to_numpy(np.float32), 0.0)
        parts.append(np.column_stack([np.full(len(x), k), g @ perp, g @ h / c, e, t["time"].to_numpy(np.float32)]))
        radius.append(np.hypot(x, y))
    if not parts:
        return np.zeros((0, 5)), np.zeros(0), d, energy, stats
    return np.concatenate(parts), np.concatenate(radius), d, energy, stats


def raw_table(sim):
    """Table of one raw shower: the summary table's fields plus, per plane and ring, the histogram of log10(E / ring
    mean) over its particles, and the energy each plane would store under CAP_V3. Our ring sums must equal plane_stats."""
    tab = {}
    for s in SPECIES_NAMES:
        p, r, d, energy, stats = raw_particles(sim, s)
        k, e = p[:, 0].astype(int), np.maximum(p[:, 3], 0.0)
        flat = k * N_RING + np.searchsorted(EDGES, r, side="right") - 1
        size = N_PLANES * N_RING
        ring_e = np.bincount(flat, e, size).reshape(N_PLANES, N_RING)
        ring_n = np.bincount(flat, None, size).reshape(N_PLANES, N_RING)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.log10(e / (ring_e.ravel() / np.maximum(ring_n.ravel(), 1))[flat])
        u[np.isnan(u)] = -np.inf                                            # 0 / 0: a ring of zero-energy particles
        b = np.clip(np.searchsorted(HIST, u, side="right") - 1, 0, len(HIST) - 2)
        hist = np.bincount(flat * (len(HIST) - 1) + b, None, size * (len(HIST) - 1)).reshape(N_PLANES, N_RING, -1)
        stored = render.render(p, d, CAP_V3[s])
        check_rings(sim, s, np.bincount(k, e, N_PLANES), ring_e, ring_n, stats)
        tab.update({f"{s}.E_total": np.bincount(k, e, N_PLANES), f"{s}.ring_energy": ring_e, f"{s}.ring_count": ring_n,
                    f"{s}.ring_tmin": stats["ring_tmin"], f"{s}.ring_tmean": stats["ring_tmean"],
                    f"{s}.hist": hist.astype(np.int32),
                    f"{s}.E_stored": np.bincount(stored[:, 2].astype(int), stored[:, 3], N_PLANES)})
    tab.update({"direction": d, "log10E": np.log10(energy)})
    return tab


def check_rings(sim, s, plane_e, ring_e, ring_n, stats):
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30), initial=0.0))
    ok = (np.array_equal(ring_n, stats["ring_count"]) and np.array_equal(ring_n.sum(1), stats["count"])
          and rel(ring_e, stats["ring_energy"]) <= E_TOL and rel(plane_e, stats["energy"]) <= E_TOL)
    if not ok:
        raise ValueError(f"{sim} {s}: ring sums differ from plane_stats")


def write_tables(sims, out, raw=False):
    """Table of every listed simulation -> <out>/<class>_<name>.npz; one JSON line per skipped shower, then a count."""
    os.makedirs(out, exist_ok=True)
    git, done = git_head(), 0
    for sim in sims:
        sim = os.path.normpath(sim)
        tab, why = (raw_table(sim), None) if raw else summary_table(sim)
        if tab is None:
            print(json.dumps({"sim": sim, "skipped": why}), flush=True)
            continue
        cls = class_of(sim)
        np.savez_compressed(os.path.join(out, f"{cls}_{os.path.basename(sim)}.npz"), git=git, sim=sim, edges=EDGES,
                            **{"class": cls}, **tab)
        done += 1
    print(json.dumps({"tables": done, "of": len(sims)}), flush=True)


def stored_cells(sim, direction):
    """Per species the shower's stored cells (energy > 0) from its h5 files, checked: as many cells as num_points, at or
    below the cap, plane energy at or below plane_stats' energy before the cut, direction as given."""
    out = {}
    for s in SPECIES_NAMES:
        with h5py.File(h5_path(sim, s), "r") as f:
            n, cap = int(f["num_points"][0]), int(f.attrs["nmax"])
            c = np.asarray(f["showers"][0], np.float64).reshape(n, -1)
            total = f["plane_stats"]["energy"][:].astype(np.float64)
            d = f["directions"][0].astype(np.float64)
        if cap != CAP[s] or n > cap:
            raise ValueError(f"{sim} {s}: cap {cap} (expected {CAP[s]}), {n} cells")
        if not np.allclose(d / np.linalg.norm(d), direction, atol=1e-6):
            raise ValueError(f"{sim} {s}: direction differs from the primaries")
        c = c[c[:, 3] > 0]
        if (np.bincount(np.rint(c[:, 2]).astype(int), c[:, 3], N_PLANES) > total * (1 + E_TOL) + 1e-6).any():
            raise ValueError(f"{sim} {s}: stored energy above the energy before the cut")
        out[s] = c
    return out

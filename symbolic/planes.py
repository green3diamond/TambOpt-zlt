"""Per-plane energy E_k and kept-cell count N_k of CORSIKA files and generated clouds, with columns (x, y, layer, energy, time)."""
import h5py
import numpy as np
import torch

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import sources

LAYER_TOL = 1e-3
CLASS = {0: "EM", 1: "hadronic"}      # the primary's class, column `pdg` of the CORSIKA files


def angles(directions):
    """Zenith and azimuth [deg] of (N, 3) directions (Up = column 2, azimuth from column 0)."""
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d[:, 2], -1, 1))), np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360


def _layers(layer):
    k = np.rint(layer).astype(np.int64)
    if np.any(np.abs(layer - k) > LAYER_TOL) or np.any(k < 0) or np.any(k >= N_PLANES):
        raise ValueError(f"layer column outside the integers 0..{N_PLANES - 1}")
    return k


def plane_sums(points):
    """(P, 5) points of one shower -> E (N_PLANES,) summed positive energy and N (N_PLANES,) points with energy > 0."""
    e = points[:, 3].astype(np.float64)
    if not np.all(np.isfinite(e)) or np.any(e < 0):
        raise ValueError("non-finite or negative point energy")
    k = _layers(points[:, 2])
    pos = e > 0
    return (np.bincount(k[pos], weights=e[pos], minlength=N_PLANES),
            np.bincount(k[pos], minlength=N_PLANES).astype(np.float64))


def from_corsika(split, rows, path=sources.corsika_path):
    """Planes of CORSIKA rows for every species, and the shared primaries (species files are row-matched)."""
    rows = np.sort(np.asarray(rows))
    out, meta = {}, None
    for s in SPECIES_NAMES:
        with h5py.File(path(s, split), "r") as f:
            m = {"ids": f["shower_ids"][rows], "energies": f["energies"][rows].reshape(-1).astype(np.float64),
                 "directions": f["directions"][rows].astype(np.float64), "labels": f["pdg"][rows].astype(np.int64)}
            E, N = np.zeros((len(rows), N_PLANES)), np.zeros((len(rows), N_PLANES))
            n_points = f["num_points"][rows]
            for i, r in enumerate(rows):
                p = np.asarray(f["showers"][r], np.float64).reshape(-1, 5)
                if len(p) != n_points[i]:
                    raise ValueError(f"{s} row {r}: {len(p)} points, num_points {n_points[i]}")
                E[i], N[i] = plane_sums(p)
        if meta is None:
            meta = m
        elif any(not np.array_equal(meta[k], m[k]) for k in m):
            raise ValueError(f"{s}: primaries differ from {SPECIES_NAMES[0]}; rows misaligned")
        out[s] = {"E": E, "N": N, "num_points": n_points.astype(np.int64)}
    return out, primaries(meta, rows)


def from_clouds(clouds, chunk=256):
    """(B, P, 5) tensor of padded clouds -> E, N (B, N_PLANES); padding rows carry zero energy."""
    E, N = np.zeros((len(clouds), N_PLANES)), np.zeros((len(clouds), N_PLANES))
    for lo in range(0, len(clouds), chunk):
        c = torch.as_tensor(clouds[lo:lo + chunk]).double()
        e = c[..., 3]
        if not torch.isfinite(e).all() or (e < 0).any():
            raise ValueError("non-finite or negative point energy")
        pos = e > 0
        k = torch.as_tensor(_layers(c[..., 2][pos].numpy()))
        b = torch.nonzero(pos)[:, 0]
        flat = b * N_PLANES + k
        size = len(c) * N_PLANES
        E[lo:lo + len(c)] = torch.zeros(size, dtype=torch.float64).index_add_(0, flat, e[pos]).view(-1, N_PLANES).numpy()
        N[lo:lo + len(c)] = torch.bincount(flat, minlength=size).view(-1, N_PLANES).double().numpy()
    return E, N


def primaries(meta, rows):
    zen, az = angles(meta["directions"])
    return {**meta, "rows": np.asarray(rows), "log10_E": np.log10(meta["energies"]), "zenith": zen, "azimuth": az,
            "cos_zenith": np.cos(np.radians(zen))}

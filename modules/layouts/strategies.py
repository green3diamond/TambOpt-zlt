"""(North, East) layout strategies.

The bbox anchor uses East (centroid col 2) and the East bbox; sampling and
projection go through `geometry.placement`. All generators return (North, East)
tensors on the mountain surface.
"""

import torch
from ..constants import N_DETECTORS
import numpy as np
from typing import Tuple

from ..geometry import Layouts as _v3_Layouts
from ..geometry import project_to_mountain_ne, sample_initial_layout_ne

# ── Layout generators (all return (North, East) tensors on the mountain) ─────

def _bbox_anchor(mountain) -> Tuple[float, float]:
    """Centroid nearest the mountain (North, East) bbox center."""
    cn = 0.5 * (mountain.n_min + mountain.n_max)
    ce = 0.5 * (mountain.east_lo + mountain.east_hi)
    d2 = (mountain.centroids_ENU[:, 1] - cn) ** 2 + (mountain.centroids_ENU[:, 0] - ce) ** 2
    k  = int(np.argmin(d2))
    return float(mountain.centroids_ENU[k, 1]), float(mountain.centroids_ENU[k, 0])


def layout_grid(mountain, n_det: int = N_DETECTORS,
                jitter_sigma: float = 20.0, rng=None):
    """NE grid + small Gaussian jitter + mountain projection."""
    if rng is None:
        rng = np.random.default_rng()
    E, N = sample_initial_layout_ne(mountain, n_units=n_det, scheme="grid")
    N = N + rng.normal(0.0, jitter_sigma, N.shape).astype(np.float32)
    E = E + rng.normal(0.0, jitter_sigma, E.shape).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    E_t = torch.as_tensor(E, dtype=torch.float32)
    return project_to_mountain_ne(mountain, E_t, N_t)


def layout_center_gaussian(mountain, n_det: int = N_DETECTORS,
                           sigma: float = 200.0, rng=None):
    """Gaussian cluster at the mountain (North, East) bbox-center anchor."""
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_e = _bbox_anchor(mountain)
    N = anchor_n + rng.normal(0.0, sigma, n_det).astype(np.float32)
    E = anchor_e + rng.normal(0.0, sigma, n_det).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    E_t = torch.as_tensor(E, dtype=torch.float32)
    return project_to_mountain_ne(mountain, E_t, N_t)


def layout_rings(mountain, n_det: int = N_DETECTORS,
                 outer_radius: float = 500.0, n_rings: int = 6,
                 jitter_sigma: float = 20.0, rng=None):
    """Concentric rings at the mountain (North, East) center (v3 `Layouts`
    with a random rotation), translated to the anchor and projected."""
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_e = _bbox_anchor(mountain)

    x, y = _v3_Layouts(
        n_detectors=n_det, n_rings=n_rings, radius=outer_radius,
        center=(0.0, 0.0), device="cpu",
    )
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    rot = float(rng.uniform(0.0, 2.0 * np.pi))
    cos_r, sin_r = float(np.cos(rot)), float(np.sin(rot))
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r

    N_t = x_rot + anchor_n + rng.normal(0.0, jitter_sigma, n_det).astype(np.float32)
    E_t = y_rot + anchor_e + rng.normal(0.0, jitter_sigma, n_det).astype(np.float32)
    return project_to_mountain_ne(mountain, E_t, N_t)


def layout_uniform_random(mountain, n_det: int = N_DETECTORS, rng=None):
    """Detectors drawn i.i.d. uniformly over the mountain (North, East) bbox, then
    projected to the surface — maximally unstructured coverage (no clustering, no
    lattice regularity)."""
    if rng is None:
        rng = np.random.default_rng()
    N = rng.uniform(mountain.n_min,   mountain.n_max,   n_det).astype(np.float32)
    E = rng.uniform(mountain.east_lo, mountain.east_hi, n_det).astype(np.float32)
    return project_to_mountain_ne(mountain,
                                  torch.as_tensor(E, dtype=torch.float32),
                                  torch.as_tensor(N, dtype=torch.float32))


def layout_latin_hypercube(mountain, n_det: int = N_DETECTORS, rng=None):
    """Latin-hypercube sample over the mountain (North, East) bbox: each axis is
    split into `n_det` equal strata with exactly one sample per stratum (the two
    axes permuted independently), giving more even space-filling coverage than
    i.i.d. uniform while staying irregular. Projected to the surface."""
    if rng is None:
        rng = np.random.default_rng()
    # One sample per stratum on [0, 1), then shuffle each axis independently so the
    # (North, East) pairing is random (the Latin-hypercube property).
    u_n = (np.arange(n_det) + rng.random(n_det)) / n_det
    u_e = (np.arange(n_det) + rng.random(n_det)) / n_det
    u_n = u_n[rng.permutation(n_det)]
    u_e = u_e[rng.permutation(n_det)]
    N = (mountain.n_min   + u_n * (mountain.n_max   - mountain.n_min)).astype(np.float32)
    E = (mountain.east_lo + u_e * (mountain.east_hi - mountain.east_lo)).astype(np.float32)
    return project_to_mountain_ne(mountain,
                                  torch.as_tensor(E, dtype=torch.float32),
                                  torch.as_tensor(N, dtype=torch.float32))


def _clip_project(mountain, E: np.ndarray, N: np.ndarray):
    """Clip (East, North) draws to the mountain bbox, then project to the surface."""
    E = np.clip(E, mountain.east_lo, mountain.east_hi).astype(np.float32)
    N = np.clip(N, mountain.n_min, mountain.n_max).astype(np.float32)
    return project_to_mountain_ne(mountain, torch.as_tensor(E), torch.as_tensor(N))


def _log_uniform(rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def layout_clusters(mountain, n_det: int = N_DETECTORS, r_min: float = 30.0,
                    r_max: float = 600.0, rng=None):
    """K clusters of M = n_det / K detectors: K a random divisor of n_det (2..n_det/2),
    centres uniform over the bbox, one radius per draw (log-uniform in [r_min, r_max]),
    members uniform in the disk."""
    if rng is None:
        rng = np.random.default_rng()
    ks = [k for k in range(2, n_det // 2 + 1) if n_det % k == 0]
    K = int(rng.choice(ks))
    M = n_det // K
    R = _log_uniform(rng, r_min, r_max)
    cE = rng.uniform(mountain.east_lo, mountain.east_hi, K)
    cN = rng.uniform(mountain.n_min, mountain.n_max, K)
    r = R * np.sqrt(rng.random(n_det))
    a = rng.uniform(0.0, 2.0 * np.pi, n_det)
    E = np.repeat(cE, M) + r * np.cos(a)
    N = np.repeat(cN, M) + r * np.sin(a)
    return _clip_project(mountain, E, N)


def layout_disk(mountain, n_det: int = N_DETECTORS, r_min: float = 200.0,
                r_max: float = 4000.0, rng=None):
    """Detectors uniform in one disk: centre uniform over the bbox, radius log-uniform
    in [r_min, r_max] (the part outside the bbox is clipped to its edge)."""
    if rng is None:
        rng = np.random.default_rng()
    R = _log_uniform(rng, r_min, r_max)
    cE = rng.uniform(mountain.east_lo, mountain.east_hi)
    cN = rng.uniform(mountain.n_min, mountain.n_max)
    r = R * np.sqrt(rng.random(n_det))
    a = rng.uniform(0.0, 2.0 * np.pi, n_det)
    return _clip_project(mountain, cE + r * np.cos(a), cN + r * np.sin(a))


def layout_grid_patch(mountain, n_det: int = N_DETECTORS, side_min: float = 500.0,
                      side_max: float = 6000.0, jitter_frac: float = 0.05, rng=None):
    """Square grid over a random square patch: side log-uniform in [side_min, side_max]
    (capped at the bbox), placed uniformly inside the bbox, jitter jitter_frac x spacing.
    n_det need not be a square: the ceil(sqrt) x ceil(sqrt) lattice is subsampled."""
    if rng is None:
        rng = np.random.default_rng()
    w_e = mountain.east_hi - mountain.east_lo
    w_n = mountain.n_max - mountain.n_min
    side = min(_log_uniform(rng, side_min, side_max), w_e, w_n)
    e0 = mountain.east_lo + rng.uniform(0.0, w_e - side)
    n0 = mountain.n_min + rng.uniform(0.0, w_n - side)
    k = int(np.ceil(np.sqrt(n_det)))
    u = (np.arange(k) + 0.5) / k
    UE, UN = np.meshgrid(u, u)
    sel = rng.permutation(k * k)[:n_det]
    E = e0 + side * UE.ravel()[sel]
    N = n0 + side * UN.ravel()[sel]
    sig = jitter_frac * side / k
    return _clip_project(mountain, E + rng.normal(0.0, sig, n_det), N + rng.normal(0.0, sig, n_det))


# ── Dataset builder ──────────────────────────────────────────────────────────

# Every layout strategy the builder knows how to run, keyed by the label that
# shows up in the logs. Which ones a dataset build actually uses is
# ACTIVE_STRATEGIES below — edit that tuple, not this table.
_ALL_STRATEGIES = {
    "grid_jit20":        ("layout_grid",            dict(jitter_sigma=20.0)),
    "grid_jit200":       ("layout_grid",            dict(jitter_sigma=200.0)),
    "center_gauss200":   ("layout_center_gaussian", dict(sigma=200.0)),
    "center_gauss400":   ("layout_center_gaussian", dict(sigma=400.0)),
    "center_gauss800":   ("layout_center_gaussian", dict(sigma=800.0)),   # run 19: 12 km plane
    "rings_R300":        ("layout_rings",           dict(outer_radius=300.0,  n_rings=5, jitter_sigma=200.0)),
    "rings_R800":        ("layout_rings",           dict(outer_radius=800.0,  n_rings=6, jitter_sigma=200.0)),
    "rings_R1800":       ("layout_rings",           dict(outer_radius=1800.0, n_rings=8, jitter_sigma=200.0)),
    "uniform_random_a":  ("layout_uniform_random",  dict()),
    "uniform_random_b":  ("layout_uniform_random",  dict()),
    "latin_hypercube_a": ("layout_latin_hypercube", dict()),
    "latin_hypercube_b": ("layout_latin_hypercube", dict()),
    "clusters_a":        ("layout_clusters",        dict()),
    "clusters_b":        ("layout_clusters",        dict()),
    "disk_a":            ("layout_disk",            dict(r_max=8000.0)),      # run 19: 12 km plane (run 18: 4000)
    "grid_patch_a":      ("layout_grid_patch",      dict(side_max=12000.0)),  # run 19: 12 km plane (run 18: 6000)
    "grid_patch_b":      ("layout_grid_patch",      dict(side_max=12000.0)),
}

# The strategies a dataset build runs, IN ORDER. Position here IS the
# `strategy_ids` value persisted in the dataset, so reordering or resizing this
# tuple invalidates every existing dataset and checkpoint — Step 2's
# `shower_level_split` derives n_showers from `strategy_ids.max() + 1`. Append,
# never insert.
#
# The paired _a/_b entries are deliberate duplicates: same generator, but each
# gets its own id and draws its own layouts, doubling the random-layout
# coverage. They used to share a label ("uniform_random" twice, "latin_hypercube"
# twice), which made two distinct ids indistinguishable in any log.
ACTIVE_STRATEGIES = (
    "center_gauss400",
    "uniform_random_a",
    "uniform_random_b",
    "latin_hypercube_a",
    "latin_hypercube_b",
    # Layout zoo (run 18, 2026-10-06): clusters, disks and partial grids, so the
    # recon also sees uneven coverage. Appended, so ids 0-4 keep their meaning.
    "clusters_a",
    "clusters_b",
    "disk_a",
    "grid_patch_a",
    # Run 19 (2026-10-09): a second random grid patch per shower (grid patches 1/9 -> 2/10 of the rows).
    "grid_patch_b",
    # Run 19 (2026-10-09): a wider centre blob for the 12 km plane (center_gauss400 is ~1.5 km across there).
    "center_gauss800",
)

_STRATEGIES = [(name, *_ALL_STRATEGIES[name]) for name in ACTIVE_STRATEGIES]

_STRATEGY_FNS = {
    "layout_grid":            layout_grid,
    "layout_center_gaussian": layout_center_gaussian,
    "layout_rings":           layout_rings,
    "layout_uniform_random":  layout_uniform_random,
    "layout_latin_hypercube": layout_latin_hypercube,
    "layout_clusters":        layout_clusters,
    "layout_disk":            layout_disk,
    "layout_grid_patch":      layout_grid_patch,
}

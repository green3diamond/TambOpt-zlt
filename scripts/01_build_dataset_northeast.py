"""Stage 1: kernel labels for every shower under layouts from the strategy zoo (modules/layouts/strategies.py).

Detectors are placed by (East, North) and lifted onto the mesh (SurfaceUpMap); labels E, T per detector come from the
kernel (modules/data/dataset_builder.build_training_pairs). Resumable after a preemption (build_resume.pt).
Output in TRAINING_DATASET_FOLDER: primary, xy (East, North), E (log1p), T, strategy_ids, species_ids, blob_ids,
norm_stats (.pt).

    python scripts/01_build_dataset_northeast.py [--strategies 0 3 5]
"""
import argparse
import os
import sys
import time

# Make `modules` importable regardless of caller CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.data import build_training_pairs
from modules.surrogates import compute_normalization
from modules.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, NUM_SHOWERS, BATCH_SIZE,
    DUAL_SHOWER_CACHE_PATH, DATASET_FRACTION, SPECIES_NAMES,
    TRAINING_DATASET_FOLDER,
)
from modules.geometry import load_tr_mountain, SurfaceUpMap


# Corpus rows are species blocks (electron, muon, photon) of the same events; Stage 2 splits them by species_ids.pt.
# DATASET_FRACTION < 1 caps the rows loaded (RAM); 1.0 -> 0 = every row in the corpus file (NUM_SHOWERS is only
# approximate, so it must not set the cap).
MAX_SHOWERS = (0 if DATASET_FRACTION >= 1.0 else
               int(DATASET_FRACTION * len(SPECIES_NAMES) * NUM_SHOWERS))
SEED        = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", type=int, nargs="+", default=None,
                    help="label only these strategy ids (default: all of ACTIVE_STRATEGIES)")
    args = ap.parse_args()
    os.makedirs(TRAINING_DATASET_FOLDER, exist_ok=True)

    print("=" * 72)
    print("v6/01_build_dataset_northeast.py")
    print("=" * 72)
    print(f"shower cache : {DUAL_SHOWER_CACHE_PATH}")
    print(f"geometry     : {GEOMETRY_PATH_RESOLVED}")
    print(f"output dir   : {TRAINING_DATASET_FOLDER}")
    print(f"batch size   : {BATCH_SIZE}")
    print(f"max showers  : {MAX_SHOWERS}")
    print(f"device       : {DEVICE}")
    print(f"strategies   : {args.strategies or 'all'}")

    # Mountain + surface map (Up = g(N, East))
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    print(f"[load] mountain + surface in {time.time() - t0:.1f}s")
    print(f"  N=[{mountain.n_min:.0f}, {mountain.n_max:.0f}]  "
          f"Up=[{mountain.u_min:.0f}, {mountain.u_max:.0f}]  "
          f"East=[{mountain.east_lo:.0f}, {mountain.east_hi:.0f}]")

    # Build training pairs
    t0 = time.time()
    primary, xy, E, T, strat, species, blob = build_training_pairs(
        mountain=mountain,
        surface=surface,
        shower_cache_path=DUAL_SHOWER_CACHE_PATH,
        batch_size=BATCH_SIZE,
        max_showers=MAX_SHOWERS,
        seed=SEED,
        device=DEVICE,
        verbose=True,
        east_entry=EAST_ENTRY,                       # placement at the real decay vertices (<corpus>_positions.pt)
        layer_east_dx=LAYER_EAST_DX,
        resume_path=os.path.join(TRAINING_DATASET_FOLDER, "build_resume.pt"),
        strategies=args.strategies,
    )
    print(f"[build] training pairs in {time.time() - t0:.1f}s")
    print(f"  primary : {tuple(primary.shape)}  dtype={primary.dtype}")
    print(f"  xy      : {tuple(xy.shape)}       dtype={xy.dtype}   (East, North)")
    print(f"  E       : {tuple(E.shape)}        dtype={E.dtype}")
    print(f"  T       : {tuple(T.shape)}        dtype={T.dtype}")
    print(f"  strat   : {tuple(strat.shape)}    unique={sorted(strat.unique().tolist())}")
    print(f"  species : {tuple(species.shape)}  unique={sorted(species.unique().tolist())}")
    print(f"  blob    : {tuple(blob.shape)}    flagged={int(blob.sum())} "
          f"({100 * blob.float().mean():.2f}%)")

    E = torch.log1p(E)                               # compresses the heavy tail

    print(f"[log1p] E range [{E.min():.4g}, {E.max():.4g}]  "
          f"T range [{T.min():.4g}, {T.max():.4g}]")

    # Sanity: non-zero E on at least some samples
    n_nonzero = int((E.abs().sum(dim=1) > 0).sum())
    print(f"  samples with any nonzero E : {n_nonzero}/{E.shape[0]}")

    # Z-score stats without the degenerate (blob) showers, which would set out_std alone; Stage 2 drops them too.
    _clean = ~blob.bool()
    stats = compute_normalization(primary[_clean], xy[_clean], E[_clean], T[_clean])
    print(f"[norm] in_mean[:5]  = {stats['in_mean'][:5].tolist()}")
    print(f"[norm] in_std[:5]   = {stats['in_std'][:5].tolist()}")
    print(f"[norm] out_mean (E) = {stats['out_mean'][:5].tolist()} ...")
    print(f"[norm] out_std  (E) = {stats['out_std'][:5].tolist()} ...")

    # Persist
    t0 = time.time()
    torch.save(primary, os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"))
    torch.save(xy,      os.path.join(TRAINING_DATASET_FOLDER, "xy.pt"))
    torch.save(E,       os.path.join(TRAINING_DATASET_FOLDER, "E.pt"))
    torch.save(T,       os.path.join(TRAINING_DATASET_FOLDER, "T.pt"))
    torch.save(strat,   os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt"))
    torch.save(species, os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt"))
    torch.save(blob,    os.path.join(TRAINING_DATASET_FOLDER, "blob_ids.pt"))
    torch.save(stats,   os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[save] tensors in {time.time() - t0:.1f}s  ->  {TRAINING_DATASET_FOLDER}")


if __name__ == "__main__":
    main()

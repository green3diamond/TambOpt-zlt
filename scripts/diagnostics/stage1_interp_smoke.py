"""Smoke-test Step 1 with the run's kernel (constants.KERNEL_INTERP_K) on a small
slice: checks the build runs end to end, times it, and prints label zero
fractions. Writes nothing into the run folder; the resume file goes to --tmp.

    python scripts/diagnostics/stage1_interp_smoke.py --max-showers 300
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _V6)
from _pathfix import V6_ROOT  # noqa: F401,E402

import torch  # noqa: E402

import modules  # noqa: F401,E402
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    DUAL_SHOWER_CACHE_PATH, KERNEL_INTERP_K, SIGMA_SPATIAL, BATCH_SIZE, DATASET_FRACTION,
    SPECIES_NAMES, NUM_SHOWERS,
)
from modules.data.dataset_builder import build_training_pairs  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-showers", type=int, default=300)
    ap.add_argument("--tmp", default="slurm_logs/stage1_smoke")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)
    print(f"corpus={DUAL_SHOWER_CACHE_PATH}\nsigma={SIGMA_SPATIAL} interp_k={KERNEL_INTERP_K} "
          f"batch={BATCH_SIZE} device={DEVICE}")
    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    t0 = time.time()
    primary, xy, E, T, strat, species, blob = build_training_pairs(
        mountain=mountain, surface=surface, shower_cache_path=DUAL_SHOWER_CACHE_PATH,
        batch_size=BATCH_SIZE, max_showers=args.max_showers, seed=0, device=DEVICE,
        verbose=True, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
        resume_path=os.path.join(args.tmp, "build_resume.pt"))
    dt = time.time() - t0
    n_rows = int(primary.shape[0]) // max(int(strat.max()) + 1, 1)
    full_rows = int(DATASET_FRACTION * len(SPECIES_NAMES) * NUM_SHOWERS)
    print(f"\n[time] {n_rows} shower rows x {int(strat.max()) + 1} strategies in {dt:.1f}s "
          f"-> {dt / max(n_rows, 1):.3f}s/row; the DATASET_FRACTION build ({full_rows:,} rows) "
          f"extrapolates to ~{dt / max(n_rows, 1) * full_rows / 3600:.1f} h")
    for sv, name in enumerate(SPECIES_NAMES):
        m = (species == sv) & ~blob
        if m.any():
            print(f"[{name}] rows={int(m.sum())}  E==0 {100 * float((E[m] == 0).float().mean()):.1f}%  "
                  f"T==0 {100 * float((T[m] == 0).float().mean()):.1f}%  "
                  f"finite={bool(torch.isfinite(E[m]).all() and torch.isfinite(T[m]).all())}")


if __name__ == "__main__":
    main()

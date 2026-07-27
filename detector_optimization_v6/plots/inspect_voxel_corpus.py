"""One-shot inspection of a built voxel corpus: range/finiteness of every
column, and which events carry pathologically large voxel counts (the outliers
that can inflate a padded chunk and, if the underlying value overflows float32,
propagate a non-finite E/T through the kernel).

    cd TambOpt/detector_optimization_v6
    python plots/inspect_voxel_corpus.py --delta 31.25
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from modules_v6.voxel_corpus import voxel_cache_path

COL_NAMES = ["N_centroid", "U_centroid", "layer_centroid", "sum_energy", "sum_time_raw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=31.25)
    ap.add_argument("--path", type=str, default=None)
    args = ap.parse_args()

    path = args.path or voxel_cache_path(args.delta)
    print(f"loading {path}")
    d = torch.load(path, map_location="cpu")
    print("keys:", list(d.keys()))
    print(f"P (padded points) = {d['P']}   n_pairs = {d['n_pairs']}   delta = {d['delta']}")

    for s in ("elec", "muon"):
        vals = d[s + "_vals"]
        counts = d[s + "_counts"]
        print(f"\n== {s} ==  vals {tuple(vals.shape)}  counts {tuple(counts.shape)}")
        print(f"  counts: min={int(counts.min())} max={int(counts.max())} "
              f"mean={counts.float().mean():.1f}  sum={int(counts.sum())}")

        for c, name in enumerate(COL_NAMES):
            col = vals[:, c]
            finite = torch.isfinite(col)
            n_bad = int((~finite).sum())
            if finite.any():
                v = col[finite]
                print(f"  col{c} {name:15s} min={v.min().item():.4e} "
                      f"max={v.max().item():.4e} mean={v.mean().item():.4e} "
                      f"nonfinite={n_bad}")
            else:
                print(f"  col{c} {name:15s} ALL NON-FINITE ({n_bad})")

        top_counts, top_ev = counts.float().topk(10)
        print(f"  top-10 voxel-count events: idx={top_ev.tolist()} "
              f"count={top_counts.long().tolist()}")

        # Per-event max |sum_energy| and |sum_time_raw|, to catch outlier
        # VOXELS (not just outlier voxel *counts*) that could overflow the
        # kernel downstream (energy_kernel, arrival_time accumulation).
        starts = torch.cumsum(counts, 0) - counts
        for c, name in ((3, "sum_energy"), (4, "sum_time_raw")):
            col = vals[:, c].abs()
            # scatter-max per event via index_reduce (torch >= 1.12)
            ev_idx = torch.repeat_interleave(torch.arange(counts.numel()), counts)
            per_ev_max = torch.zeros(counts.numel())
            per_ev_max.index_reduce_(0, ev_idx, col, "amax", include_self=False)
            top_v, top_i = per_ev_max.topk(10)
            print(f"  top-10 per-event max |{name}|: idx={top_i.tolist()} "
                  f"val={[f'{x:.3e}' for x in top_v.tolist()]}")


if __name__ == "__main__":
    main()

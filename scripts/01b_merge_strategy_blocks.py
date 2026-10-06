"""Concatenate two Stage-1 datasets built from the same corpus with different strategies.

`01_build_dataset_northeast.py --strategies 5 6 7 8` labels only the new strategies; this
appends those blocks to an existing build (ids 0-4) so the result is the dataset a full
build of all ids would have written. Rows are strategy-major (then species, then event),
so the merge is a plain concatenation. Checked first: same block length, the same events
in every block (primary and species rows equal block 0 of the base), ids contiguous.
norm_stats.pt is recomputed on the clean rows, as Stage 1 does.

    python scripts/01b_merge_strategy_blocks.py --base <..._northeast> --add <..._northeast_zoo_new> \\
        --out <..._northeast_zoo>
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _pathfix import V6_ROOT  # noqa: F401,E402
import torch

from modules.surrogates import compute_normalization

FILES = ("primary.pt", "xy.pt", "E.pt", "T.pt", "strategy_ids.pt", "species_ids.pt", "blob_ids.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    load = lambda d, f: torch.load(os.path.join(d, f), map_location="cpu")  # noqa: E731

    st_b, st_a = load(args.base, "strategy_ids.pt"), load(args.add, "strategy_ids.pt")
    ids_b, ids_a = st_b.unique().tolist(), st_a.unique().tolist()
    n_blk = int((st_b == ids_b[0]).sum())
    assert ids_b + ids_a == list(range(len(ids_b) + len(ids_a))), (ids_b, ids_a)
    assert st_b.numel() == n_blk * len(ids_b) and st_a.numel() == n_blk * len(ids_a), \
        (st_b.numel(), st_a.numel(), n_blk)
    for f in ("primary.pt", "species_ids.pt"):
        b, a = load(args.base, f), load(args.add, f)
        for k in range(len(ids_a)):
            assert torch.equal(a[k * n_blk:(k + 1) * n_blk], b[:n_blk]), f"{f}: block {ids_a[k]} != base block 0"
    print(f"[check] base ids {ids_b} + add ids {ids_a}; {n_blk} rows per block; same events in every block")

    for f in FILES:
        t0 = time.time()
        x = torch.cat([load(args.base, f), load(args.add, f)])
        torch.save(x, os.path.join(args.out, f))
        print(f"[save] {f} {tuple(x.shape)} in {time.time() - t0:.1f}s")
        del x

    keep = ~load(args.out, "blob_ids.pt").bool()
    stats = compute_normalization(load(args.out, "primary.pt")[keep], load(args.out, "xy.pt")[keep],
                                  load(args.out, "E.pt")[keep], load(args.out, "T.pt")[keep])
    torch.save(stats, os.path.join(args.out, "norm_stats.pt"))
    print(f"[save] norm_stats.pt -> {args.out}")


if __name__ == "__main__":
    main()

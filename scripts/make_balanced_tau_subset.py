"""Write energy-balanced subsets of a tau_wholesky HDF5 corpus.

The whole-sky throw follows a falling spectrum, so `load_tau_primaries`'s
in-band selection is dominated by the low decades: over [1e5, 1e8] GeV the
1e5-1e6 decade holds 390,791 events and 1e7-1e8 only 51,945. A surrogate
trained on that sees the top decade ~8x less often than the bottom, which is
exactly where the AllShowers generator is already weakest.

This script re-samples to a flat distribution in log10(E), writing a new file
with the SAME dataset layout as the input (position/direction (3, N), energy
(N,), pdg (N,), oneweight (N,)), so `load_tau_primaries` reads it unchanged --
point TAU_WHOLESKY_PATH at the output and nothing else in the pipeline moves.

Two modes:

  --n 0   maximum perfectly-flat subset: every bin gets min(counts), so the
          thinnest bin sets the size and the histogram is exactly level.

  --n N   exactly N events, water-filled: the per-bin quota starts at N/nbins,
          and bins that cannot fill it hand their deficit back to the bins that
          can. Flattest achievable shape at that size. When the target is well
          above the flat maximum this degenerates towards the input spectrum,
          so the printed max/min ratio is the number to judge it by, not the
          fact that it returned N rows.

Balancing DISCARDS events, which breaks the flux normalisation `oneweight`
encodes. The per-event `sampling_fraction` dataset records what fraction of its
bin each kept event stands for, so a flux-weighted analysis can recover the
original spectrum with `oneweight / sampling_fraction`. Nothing in the current
pipeline uses oneweight -- it is carried for provenance.

    python scripts/make_balanced_tau_subset.py \\
        --in decay_locations/tau_wholesky_n750k.h5 \\
        --band 5 8 --bin-width 0.1 --n 0 \\
        --out decay_locations/tau_wholesky_balanced_1e5_1e8_max.h5
"""
import argparse
import os

import h5py
import numpy as np

DSETS_COL = ("position", "direction")      # (3, N) — sliced on axis 1
DSETS_ROW = ("energy", "pdg", "oneweight")  # (N,)  — sliced on axis 0


def water_fill(counts, target):
    """Per-bin quotas summing to `target`, as flat as the counts allow.

    Starts every bin at target/nbins; a bin holding less than its quota is
    pinned to everything it has and its deficit is redistributed over the bins
    that still have room. Repeats until no bin is over-subscribed, so the result
    is the flattest histogram of size `target` that this data can produce.
    Returns quotas summing to min(target, counts.sum())."""
    counts = np.asarray(counts, np.int64)
    take = np.zeros_like(counts)
    free = np.ones(counts.size, bool)
    remaining = int(min(target, counts.sum()))
    while remaining > 0 and free.any():
        share = remaining // int(free.sum())
        if share == 0:
            # Fewer events left than free bins: hand out one each, widest bins
            # first, so the remainder does not all land in one place.
            order = np.argsort(-counts * free)[:remaining]
            take[order] += 1
            break
        capped = free & (counts - take <= share)
        if capped.any():                     # pin the bins that cannot fill
            remaining -= int((counts - take)[capped].sum())
            take[capped] = counts[capped]
            free &= ~capped
        else:                                # every free bin can take `share`
            take[free] += share
            remaining -= share * int(free.sum())
    return take


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="input tau_wholesky .h5")
    ap.add_argument("--out", required=True, help="output .h5")
    ap.add_argument("--band", nargs=2, type=float, default=(5.0, 8.0),
                    metavar=("LOG_E_MIN", "LOG_E_MAX"),
                    help="log10(E/GeV) band to balance over (default 5 8)")
    ap.add_argument("--bin-width", type=float, default=0.1,
                    help="bin width in dex (default 0.1)")
    ap.add_argument("--n", type=int, default=0,
                    help="target event count; 0 = maximum perfectly-flat subset")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for which events are drawn within each bin")
    args = ap.parse_args()

    lo, hi = float(args.band[0]), float(args.band[1])
    with h5py.File(args.src, "r") as f:
        energy = np.asarray(f["energy"][...], np.float64)
        have = {k: k in f for k in DSETS_COL + DSETS_ROW}
    log_e = np.log10(energy)

    # Closed at both ends, matching load_tau_primaries' `>= e_min & <= e_max`.
    in_band = np.nonzero((log_e >= lo) & (log_e <= hi))[0]
    edges = np.arange(lo, hi + 1e-9, args.bin_width)
    nb = len(edges) - 1
    # digitize on the interior edges, so the closed top edge lands in the last
    # bin rather than a phantom bin nb.
    bin_of = np.clip(np.digitize(log_e[in_band], edges[1:-1]), 0, nb - 1)
    counts = np.bincount(bin_of, minlength=nb)

    flat_max = int(counts.min()) * nb
    target = flat_max if args.n <= 0 else int(args.n)
    take = water_fill(counts, target)

    print(f"source      : {args.src}")
    print(f"band        : log10(E/GeV) in [{lo:g}, {hi:g}]  ->  {in_band.size:,} events")
    print(f"bins        : {nb} x {args.bin_width:g} dex")
    print(f"flat maximum: {flat_max:,}  (thinnest bin {counts.min():,})")
    print(f"target      : {'max flat' if args.n <= 0 else f'{target:,}'}")

    rng = np.random.default_rng(args.seed)
    picked, frac = [], []
    for b in range(nb):
        pool = in_band[bin_of == b]
        k = int(take[b])
        if k <= 0:
            continue
        sel = pool if k >= pool.size else rng.choice(pool, size=k, replace=False)
        picked.append(sel)
        frac.append(np.full(sel.size, k / pool.size, np.float64))
    keep = np.concatenate(picked)
    frac = np.concatenate(frac)

    # Sort by original row so the output preserves input order; h5py fancy
    # indexing requires an increasing index list anyway.
    order = np.argsort(keep)
    keep, frac = keep[order], frac[order]

    kept = np.bincount(np.clip(np.digitize(log_e[keep], edges[1:-1]), 0, nb - 1),
                       minlength=nb)
    nz = kept[kept > 0]
    print(f"kept        : {keep.size:,}  ({100 * keep.size / in_band.size:.1f}% of band)")
    print(f"flatness    : per-bin {nz.min():,}-{nz.max():,}  "
          f"(max/min = {nz.max() / nz.min():.2f}, 1.00 is exactly flat)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.src, "r") as f, h5py.File(args.out, "w") as g:
        for k in DSETS_COL:
            if have[k]:
                g.create_dataset(k, data=np.asarray(f[k][...])[:, keep])
        for k in DSETS_ROW:
            if have[k]:
                g.create_dataset(k, data=np.asarray(f[k][...])[keep])
        # Lets a flux-weighted analysis undo the balancing: this event stands
        # for 1/sampling_fraction of its bin in the original throw.
        g.create_dataset("sampling_fraction", data=frac)
        g.create_dataset("source_index", data=keep)
        g.attrs.update(source=os.path.abspath(args.src), band_log10_gev=(lo, hi),
                       bin_width_dex=args.bin_width, seed=args.seed,
                       n_requested=args.n, n_kept=keep.size)
    print(f"[save] {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    print("\nper-bin counts (source -> kept):")
    for b in range(nb):
        print(f"  {edges[b]:.2f}-{edges[b+1]:.2f}: {counts[b]:>7,} -> {kept[b]:>7,}")


if __name__ == "__main__":
    main()

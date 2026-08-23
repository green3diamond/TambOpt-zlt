"""Measure the degenerate-shower ("blob") rate across a whole corpus, and check
that a `total_deposited / E_primary` cut has empty space around it.

The Step-0 guard re-rolls a shower whose deposited energy is absurd against its
primary. That only works if the threshold sits in a gap: this scan reads every
row of a corpus and reports where the population actually lies, so the number is
read off the data instead of inferred from the three rows in tests/README.md.

    sbatch slurm/run_scan_blobs.sh                    # full 07 corpus
    python tests/scan_corpus_blobs.py --stride 50     # quick 2% pass

Writes per-row statistics to an .npz so the thresholds can be re-cut without
re-reading 197 GB.
"""
import argparse
import os
import time

import h5py
import numpy as np

DEFAULT_CORPUS = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/"
                  "detector_optimization_v6/07_750k_primaires_meanvar/v6_run_00/"
                  "cashed_showers_tau_dual.pt")
N_PLANES = 24
NCOL = 5            # x, y, layer, energy, time
E_COL, LAYER_COL = 3, 2

# Decades of total/E_prim to report counts in. The good population sits at ~2 and
# the known blobs at 5.8e5 / 5.0e7; the point of the table is to show what, if
# anything, lives in between.
BANDS = [0, 10, 30, 100, 300, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf]


def scan_chunk(flat_rows, num_points):
    """Per-row statistics for one chunk of ragged rows, without a Python loop
    over points. Rows are concatenated and reduced with reduceat/bincount."""
    n = len(num_points)
    pts = np.concatenate([r.reshape(-1, NCOL) for r in flat_rows], axis=0) \
        if n else np.zeros((0, NCOL), np.float32)
    e = pts[:, E_COL].astype(np.float64)
    lay = np.clip(np.rint(pts[:, LAYER_COL]).astype(np.int64), 0, N_PLANES - 1)

    # Row boundaries in the concatenated array. reduceat needs a start index per
    # row and chokes on empty rows, so guard them explicitly.
    starts = np.zeros(n, np.int64)
    np.cumsum(num_points[:-1], out=starts[1:])
    ok = num_points > 0
    safe = np.where(ok, starts, 0)

    total = np.where(ok, np.add.reduceat(e, safe) if len(e) else 0.0, 0.0)
    emax = np.where(ok, np.maximum.reduceat(e, safe) if len(e) else 0.0, 0.0)

    # Per-(row, plane) energy via a single flat bincount, then max over planes.
    row_of_pt = np.repeat(np.arange(n), num_points)
    plane_e = np.bincount(row_of_pt * N_PLANES + lay, weights=e,
                          minlength=n * N_PLANES).reshape(n, N_PLANES)
    return total, emax, plane_e.max(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--chunk", type=int, default=512, help="rows per read")
    ap.add_argument("--stride", type=int, default=1,
                    help="scan every Nth row (1 = every row). A quick pass at "
                         "50 still sees ~23k rows and bounds the rate.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    ap.add_argument("--out", default=None, help="per-row .npz (default: beside the corpus)")
    args = ap.parse_args()

    sidecar = os.path.splitext(args.corpus)[0] + "_species.pt"
    species_ids = None
    if os.path.exists(sidecar):
        import torch
        species_ids = torch.load(sidecar).long().numpy()

    with h5py.File(args.corpus, "r") as f:
        d, npts_all, eprim_all = f["showers"], f["num_points"][:], f["energies"][:, 0]
        n_rows = len(npts_all)
        rows = np.arange(0, n_rows, args.stride)
        if args.limit:
            rows = rows[:args.limit]
        print(f"[scan] {args.corpus}\n[scan] {len(rows):,} of {n_rows:,} rows "
              f"(stride {args.stride})", flush=True)

        tot = np.zeros(len(rows)); emx = np.zeros(len(rows)); pmx = np.zeros(len(rows))
        t0 = time.time()
        for i in range(0, len(rows), args.chunk):
            sel = rows[i:i + args.chunk]
            # h5py fancy-indexes a vlen dataset only with a sorted list; a
            # contiguous slice is much faster, so use one when stride allows.
            raw = d[sel[0]:sel[-1] + 1] if args.stride == 1 else d[sel.tolist()]
            npts = npts_all[sel].astype(np.int64)
            tot[i:i + len(sel)], emx[i:i + len(sel)], pmx[i:i + len(sel)] = \
                scan_chunk(list(raw), npts)
            if i and i % (args.chunk * 100) == 0:
                el = time.time() - t0
                print(f"  {i:>9,}/{len(rows):,}  {i/el:7.0f} rows/s  "
                      f"eta {(len(rows)-i)/max(i/el,1e-9)/60:6.1f} min", flush=True)

    eprim = eprim_all[rows].astype(np.float64)
    npts = npts_all[rows].astype(np.int64)
    ratio = tot / np.maximum(eprim, 1e-30)
    plane_ratio = pmx / np.maximum(eprim, 1e-30)

    out = args.out or os.path.join(os.path.dirname(args.corpus), "blob_scan.npz")
    np.savez_compressed(out, rows=rows, n_points=npts, e_prim=eprim, total=tot,
                        e_max=emx, plane_max=pmx,
                        species=(species_ids[rows] if species_ids is not None
                                 else np.zeros(len(rows), np.int64)))
    print(f"\n[save] {out}  ({len(rows):,} rows, {time.time()-t0:.0f}s)")

    blocks = {"all": np.ones(len(rows), bool)}
    if species_ids is not None:
        for s in np.unique(species_ids[rows]):
            blocks[f"species{s}"] = species_ids[rows] == s

    for name, m in blocks.items():
        r = ratio[m]
        if not r.size:
            continue
        print(f"\n=== {name}  ({r.size:,} rows) ===")
        qs = [50, 90, 99, 99.9, 99.99, 99.999, 100]
        print("  total/E_prim  " + "  ".join(f"p{q}={np.percentile(r, q):.4g}" for q in qs))
        pr = plane_ratio[m]
        print("  maxplane/prim " + "  ".join(f"p{q}={np.percentile(pr, q):.4g}" for q in qs))
        print("  band                 count      fraction")
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            k = int(((r >= lo) & (r < hi)).sum())
            print(f"   [{lo:>8.4g}, {hi:>8.4g})  {k:>10,}  {k/r.size:12.3e}"
                  + ("   <-- EMPTY" if k == 0 else ""))


if __name__ == "__main__":
    main()

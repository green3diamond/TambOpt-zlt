"""Profile fits of the energy per plane of every CORSIKA shower (symbolic/longitudinal.py), written to OUT/profiles.

Stages:
  slice  a fixed random slice of N_SLICE training showers -> fits_slice.npz (the in-plane laws are fitted on it),
         and the whole test file -> fits_test.npz
  part   one contiguous part of the training file (CPU array task; a part already on disk is skipped)
  merge  all parts -> fits_train.npz (the sampler is fitted on it); must agree with fits_slice.npz on shared rows
"""
import argparse
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np

from analysis_core import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, longitudinal, planes, sources

FOLDER = "profiles"
SLICE, TEST, TRAIN = (os.path.join(FOLDER, f"fits_{k}.npz") for k in ("slice", "test", "train"))
N_SLICE = 10000
CHUNK = 64
FIT_RTOL = 1e-6
MAX_FIT_MISMATCH = 1e-3        # share of slice rows allowed to land in another local minimum


def n_rows(split):
    with h5py.File(sources.corsika_path(SPECIES_NAMES[0], split), "r") as f:
        return len(f["shower_ids"])


def part_rows(n, n_parts, i):
    return np.array_split(np.arange(n), n_parts)[i]


def part_path(folder, i, n_parts):
    return os.path.join(folder, "parts", f"part_{i:04d}_of_{n_parts:04d}.npz")


def check_cover(rows, n):
    """The concatenated part rows must be exactly 0..n-1 in order."""
    if len(rows) != n or not np.array_equal(rows, np.arange(n)):
        raise ValueError(f"parts cover {len(np.unique(rows))} distinct of {n} rows ({len(rows)} entries)")


def save(path, arrays, log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))


def fit_rows(split, rows, workers, log):
    """Energy per plane, slant depths and profile fits of the given rows, as a flat dict of arrays."""
    data, meta = planes.from_corsika(split, rows)
    X = atmosphere.plane_depths(meta["zenith"], N_PLANES)
    arrays = {**{f"meta.{k}": v for k, v in meta.items()}, "X": X}
    with Pool(workers) as pool:
        for s in SPECIES_NAMES:
            Y = longitudinal.log_energy(data[s]["E"])
            parts = pool.map(longitudinal.fit_block, [(X[i:i + CHUNK], Y[i:i + CHUNK]) for i in range(0, len(Y), CHUNK)])
            fit = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            arrays.update({f"{s}.{k}": v for k, v in {**data[s], **fit}.items()})
            log.info("%s %-8s %d rows, fitted %.4f", split, s, len(rows), np.isfinite(fit["ln_n"]).mean())
    return arrays


def stage_slice(args, log):
    n = args.limit or N_SLICE
    rows = np.sort(np.random.default_rng(0).choice(n_rows("train"), n, replace=False))
    save(os.path.join(args.out, SLICE), fit_rows("train", rows, args.workers, log), log)
    test = np.sort(np.random.default_rng(1).choice(n_rows("test"), args.limit, replace=False)) if args.limit \
        else np.arange(n_rows("test"))
    save(os.path.join(args.out, TEST), fit_rows("test", test, args.workers, log), log)
    return {"n_slice": len(rows), "n_test": len(test)}


def stage_part(args, log):
    rows = part_rows(args.limit or n_rows("train"), args.n_parts, args.part)
    path = part_path(os.path.join(args.out, FOLDER), args.part, args.n_parts)
    if os.path.exists(path):
        with np.load(path) as z:
            if np.array_equal(z["meta.rows"], rows):
                log.info("part %d already done, skipped", args.part)
                return {"part": args.part, "skipped": True}
        raise ValueError(f"{path} exists with other rows")
    save(path, fit_rows("train", rows, args.workers, log), log)
    return {"part": args.part, "rows": [int(rows[0]), int(rows[-1])]}


def gate_slice(merged, slice_path, log):
    """On the slice rows the energies must match exactly and the fits to FIT_RTOL."""
    z = np.load(slice_path)
    r = z["meta.rows"]
    r = r[r < len(merged["meta.rows"])]
    i = np.searchsorted(z["meta.rows"], r)
    res = {}
    for s in SPECIES_NAMES:
        if not np.array_equal(merged[f"{s}.E"][r], z[f"{s}.E"][i]):
            raise ValueError(f"{s}: energies per plane differ from the slice fits")
        a = np.column_stack([merged[f"{s}.ln_n"][r], merged[f"{s}.theta"][r]])
        b = np.column_stack([z[f"{s}.ln_n"][i], z[f"{s}.theta"][i]])
        frac = float(1 - np.all(np.isclose(a, b, rtol=FIT_RTOL, atol=0, equal_nan=True), 1).mean())
        log.info("gate %-8s %d slice rows: energies equal, fits differ on %.4f", s, len(r), frac)
        if frac > MAX_FIT_MISMATCH:
            raise ValueError(f"{s}: {frac:.4f} of the slice rows fit differently")
        res[s] = frac
    return res


def stage_merge(args, log):
    folder = os.path.join(args.out, FOLDER)
    parts = [dict(np.load(part_path(folder, i, args.n_parts))) for i in range(args.n_parts)]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    check_cover(merged["meta.rows"], args.limit or n_rows("train"))
    gate = gate_slice(merged, os.path.join(args.out, SLICE), log)
    save(os.path.join(args.out, TRAIN), merged, log)
    return {"n_rows": len(merged["meta.rows"]), "slice_fit_mismatch": gate,
            "fitted": {s: float(np.isfinite(merged[f"{s}.ln_n"]).mean()) for s in SPECIES_NAMES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["slice", "part", "merge"])
    ap.add_argument("--n-parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--limit", type=int, default=0, help="smoke run: LIMIT rows of each file")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage != "slice" and args.n_parts < 1:
        raise ValueError("--n-parts is required for part and merge")
    if args.stage == "part" and not 0 <= args.part < args.n_parts:
        raise ValueError(f"--part {args.part} outside 0..{args.n_parts - 1}")
    name = f"part_{args.part:04d}" if args.stage == "part" else args.stage
    log_dir, log = logs.start(os.path.join(FOLDER, "logs", name), args)
    stage = {"slice": stage_slice, "part": stage_part, "merge": stage_merge}[args.stage]
    logs.save(log_dir, stage(args, log), args, log)


if __name__ == "__main__":
    main()

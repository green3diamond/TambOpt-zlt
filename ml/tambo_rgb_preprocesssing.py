#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process runs base-id by base-id to improve cache locality and reduce memory.
For each base id (e.g. 44893315 from 44893315_1), we:
  1) Process all its subruns (particles dirs)
  2) Save per-run .npy RGBs (unchanged behavior)
  3) Immediately write a single grouped NPZ bundle for that base id
Then move on to the next base id.

This minimizes the number of times we switch back and forth between widely
separated folders and prevents holding all records in memory at once.
"""

import os
import re
import shlex
import yaml
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import random
from collections import defaultdict
import pandas as pd  # NEW

random.seed(42)  # optional for reproducibility

# =============================
# Config
# =============================
paths_file = "/n/home04/hhanif/output_paths.txt"  # file with run dirs (one per line)
use_normalized_xy = True    # clamp X,Y to [0,1] for binning
log_counts = True           # log1p on density channel before min-max
max_folders = 50000         # limit processed folders (total particles/dirs)
save_npys = True            # save rgb arrays to disk as .npy files (per-run)

# Grouped NPZ settings (ONLY grouped bundles are produced)
save_npzs_grouped = True
npz_dirname = "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_pre_processed_files_500000_simulations"
npz_compressed = True       # use np.savez_compressed

# =============================
# Robust arg parsing helpers
# =============================
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_args_string(args_str: str):
    tokens = shlex.split(args_str) if isinstance(args_str, str) else []
    arg_dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if "=" in tok[2:]:  # --key=value
                key, val = tok[2:].split("=", 1)
                arg_dict[key] = val
            else:               # --key value  OR  --flag
                key = tok[2:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    arg_dict[key] = tokens[i + 1]
                    i += 1
                else:
                    arg_dict[key] = True
        i += 1
    return arg_dict

def regex_flag_float(args_str: str, names):
    if not isinstance(args_str, str):
        return None
    for name in names:
        m = re.search(rf"--{re.escape(name)}(?:\s*=\s*|\s+)\s*({_NUM})", args_str)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def to_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def get_float_flag(args_str, arg_dict, keys, default=0.0):
    for k in keys:
        if k in arg_dict:
            v = to_float(arg_dict[k], None)
            if v is not None:
                return v
    v = regex_flag_float(args_str, keys)
    return default if v is None else v

# =============================
# Helper functions
# =============================

def minmax01(a, lo=1, hi=99, gamma=1.0):
    """
    Scale array a -> [0,1] with percentile clipping and optional gamma.

    lo, hi : percentiles used as lower/upper bounds (e.g. 1, 99)
    gamma  : apply y = x**gamma after scaling (gamma<1 brightens)
    """
    a = np.asarray(a, float)
    mask = np.isfinite(a)
    if not mask.any():
        return np.zeros_like(a)

    v = a[mask]
    v_lo, v_hi = np.nanpercentile(v, [lo, hi])
    if v_hi <= v_lo:
        out = np.zeros_like(a)
        out[mask] = 1.0
        return out

    v_clipped = np.clip(v, v_lo, v_hi)
    v_scaled = (v_clipped - v_lo) / (v_hi - v_lo)

    if gamma != 1.0:
        v_scaled = np.power(v_scaled, gamma)

    out = np.zeros_like(a)
    out[mask] = v_scaled
    return out


def rgb_from_df(df, bins=32, log_counts=True):
    """
    Build an RGB image from a dataframe with X_transformed, Y_transformed,
    time, and kinetic_energy columns.
    """
    x = df["X_transformed"].to_numpy()
    y = df["Y_transformed"].to_numpy()

    t = df.get("time", pd.Series(np.zeros_like(x))).to_numpy()
    ke = df.get("kinetic_energy", pd.Series(np.zeros_like(x))).to_numpy()

    x_edges = np.linspace(0, 1, bins + 1)
    y_edges = np.linspace(0, 1, bins + 1)

    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    t_sum,  _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=t)
    ke_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=ke)

    eps = 1e-12
    t_mean  = t_sum  / (counts + eps)
    ke_mean = ke_sum / (counts + eps)

    # Base channels
    R = np.log1p(counts) if log_counts else counts
    G = t_mean
    B = ke_mean

    # Mask pure background
    mask_nonzero = counts > 0
    R[~mask_nonzero] = 0.0
    G[~mask_nonzero] = 0.0
    B[~mask_nonzero] = 0.0

    # Stronger contrast via percentile clipping + gamma
    R = minmax01(R, lo=20, hi=99, gamma=0.7)  # compress very high-count pixels
    G = minmax01(G, lo=5,  hi=95, gamma=1.2)  # stretch time variation
    B = minmax01(B, lo=5,  hi=95, gamma=1.2)  # stretch energy variation

    rgb = np.dstack([R, G, B])
    return rgb


def radians_sin_cos(angle):
    """
    Compute sin/cos assuming radians by default.
    If |angle| looks like degrees (> 2π), auto-convert from degrees.
    """
    if np.isfinite(angle) and abs(angle) > 2 * np.pi + 1e-6:
        angle = np.deg2rad(angle)
    return np.sin(angle), np.cos(angle)


def _ensuredir(path):
    os.makedirs(path, exist_ok=True)
    return path


def warn(msg: str):
    """Nice printing that plays well with tqdm."""
    try:
        tqdm.write(str(msg))
    except Exception:
        print(str(msg))


def is_valid_parquet(path: str, min_size_bytes: int = 8) -> bool:
    """
    Quick sanity check for truncated/corrupt Parquet files.
    - checks existence and minimum footer size
    - attempts to open metadata via pyarrow.parquet.ParquetFile
    """
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < min_size_bytes:
            return False
        pq.ParquetFile(path)  # raises if corrupt/truncated
        return True
    except Exception:
        return False

# =============================
# Read run paths from file
# =============================

def read_run_dirs_from_txt(txt_path):
    runs = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Paths file not found: {txt_path}")
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            runs.append(os.path.normpath(p))
    seen = set()
    unique = []
    for r in runs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique

# =============================
# Per-folder processing
# =============================

def process_one_folder(particles_dir, bins_list,
                       use_normalized_xy=True, log_counts=True, save_npys=True):
    run_dir = os.path.dirname(particles_dir)
    base_config_path = os.path.join(run_dir, "config.yaml")
    particles_config_path = os.path.join(particles_dir, "config.yaml")
    parquet_path = os.path.join(particles_dir, "particles.parquet")

    if not (os.path.exists(particles_config_path) and os.path.exists(parquet_path)):
        return None, "missing_config_or_parquet"

    # Skip corrupt/truncated parquet early
    if not is_valid_parquet(parquet_path):
        size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else -1
        warn(f"Skipping {particles_dir} (invalid parquet: {size} bytes)")
        return None, "invalid_parquet"

    # ---- Load configs ----
    try:
        with open(particles_config_path) as f:
            particles_cfg = yaml.safe_load(f) or {}
    except Exception:
        return None, "bad_particles_config"

    base_cfg = {}
    try:
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_cfg = yaml.safe_load(f) or {}
    except Exception:
        # non-fatal; just proceed with particles_cfg
        base_cfg = {}

    args_str = base_cfg.get("args") or particles_cfg.get("args") or ""
    arg_dict = parse_args_string(args_str)

    energy  = get_float_flag(args_str, arg_dict, ["energy", "primary-energy"], default=0.0)
    zenith  = get_float_flag(args_str, arg_dict, ["zenith"], default=0.0)
    azimuth = get_float_flag(args_str, arg_dict, ["azimuth"], default=0.0)

    sin_zenith,  cos_zenith  = radians_sin_cos(zenith)
    sin_azimuth, cos_azimuth = radians_sin_cos(azimuth)

    # ---- Load particles (awkward) ----
    try:
        akf = ak.from_parquet(parquet_path)
    except Exception as e:
        warn(f"Skipping {particles_dir} (failed to read parquet: {type(e).__name__}: {e})")
        return None, "read_parquet_failed"

    # ---- Filter by PDG ----
    if "pdg" in akf.fields:
        mask = (akf["pdg"] == 11) | (akf["pdg"] == 13) | (akf["pdg"] == 22)
        akf = akf[mask]
    else:
        warn(f"Skipping {particles_dir} (missing pdg field)")
        return None, "missing_pdg"

    # If empty after filtering, skip
    if len(akf) == 0:
        warn(f"Skipping {particles_dir} (no hits with pdg 11,13,22)")
        return None, "no_valid_pdg"

    # Required geometry keys
    try:
        center = np.asarray(particles_cfg["plane"]["center"], dtype=float)  # (3,)
        zhat   = np.asarray(particles_cfg["plane"]["normal"], dtype=float)  # (3,)
        xhat   = np.asarray(particles_cfg["x-axis"], dtype=float)           # (3,)
        yhat   = np.asarray(particles_cfg["y-axis"], dtype=float)           # (3,)
    except Exception:
        warn(f"Skipping {particles_dir} (missing/invalid geometry in particles config)")
        return None, "bad_geometry"

    mat = np.vstack([xhat, yhat, zhat])  # (3,3), rows are basis vectors

    # ===== Vectorized coordinate transform =====
    try:
        x_local = np.asarray(akf["x"])
        y_local = np.asarray(akf["y"])
    except Exception:
        warn(f"Skipping {particles_dir} (missing x/y columns)")
        return None, "missing_xy"

    n = x_local.shape[0]
    if n == 0:
        warn(f"Skipping {particles_dir} (empty parquet)")
        return None, "empty_parquet"

    xyz_local = np.vstack([x_local, y_local, np.zeros(n, dtype=float)])  # (3,N)
    xyzs = (mat @ xyz_local) + center.reshape(3, 1)                       # (3,N)

    # Extra channels if present
    time_vals = np.asarray(akf["time"]) if "time" in akf.fields else np.full(n, np.nan, dtype=float)
    ke_vals   = np.asarray(akf["kinetic_energy"]) if "kinetic_energy" in akf.fields else np.full(n, np.nan, dtype=float)

    # ===== Normalize (optionally in-place) =====
    def norm01_inplace(arr):
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            return arr
        v = arr[mask]
        vmin = v.min()
        vmax = v.max()
        if vmax == vmin:
            arr[mask] = 0.0
        else:
            arr[mask] = (v - vmin) / (vmax - vmin)
        return arr

    if use_normalized_xy:
        X_t = norm01_inplace(xyzs[0].astype(float))
        Y_t = norm01_inplace(xyzs[1].astype(float))
        _Z  = norm01_inplace(xyzs[2].astype(float))  # Z normalized although unused in RGB
    else:
        X_t = xyzs[0].astype(float)
        Y_t = xyzs[1].astype(float)
        _Z  = xyzs[2].astype(float)

    time_vals = norm01_inplace(time_vals.astype(float))
    ke_vals   = norm01_inplace(ke_vals.astype(float))

    # ===== Build DataFrame and RGB tensors =====
    df = pd.DataFrame(
        {
            "X_transformed": X_t,
            "Y_transformed": Y_t,
            "time": time_vals,
            "kinetic_energy": ke_vals,
        }
    )

    rgb_by_bins = {}
    for b in bins_list:
        rgb_by_bins[b] = rgb_from_df(df, bins=b, log_counts=log_counts)

    # ===== Optional: save to disk as .npy (existing behavior) =====
    if save_npys:
        out32 = os.path.join(particles_dir, "rgb_32.npy")
        np.save(out32, rgb_by_bins[32])

    # One record per run (for DF and later grouping)
    rec = {
        "run_id": os.path.basename(run_dir),   # e.g., 44893315_2
        "energy": energy,
        "sin_zenith": sin_zenith,
        "cos_zenith": cos_zenith,
        "sin_azimuth": sin_azimuth,
        "cos_azimuth": cos_azimuth,
        "rgb_32": rgb_by_bins[32],
        "parent_dir": os.path.dirname(run_dir),  # for placing grouped npz
    }
    return rec, "ok"


def base_id(run_id: str):
    parts = run_id.split("_", 1)
    return parts[0] if parts else run_id


# =============================
# Group writer (per base id)
# =============================

def write_group_npz(bid, recs, out_root, compressed=True):
    """Write a grouped NPZ for a base id from its per-run records."""
    _ensuredir(out_root)
    out_path = os.path.join(out_root, f"{bid}.npz")
    saver = np.savez_compressed if compressed else np.savez

    payload = {}
    manifest_rows = []
    for rec in sorted(recs, key=lambda r: r["run_id"]):
        rid = rec["run_id"]          # e.g., 44893315_1
        # unique keys per subrun
        payload[f"{rid}_rgb_32"] = rec["rgb_32"]
        payload[f"{rid}_energy"] = np.float32(rec["energy"])
        payload[f"{rid}_sin_zenith"] = np.float32(rec["sin_zenith"])
        payload[f"{rid}_cos_zenith"] = np.float32(rec["cos_zenith"])
        payload[f"{rid}_sin_azimuth"] = np.float32(rec["sin_azimuth"])
        payload[f"{rid}_cos_azimuth"] = np.float32(rec["cos_azimuth"])

        # manifest columns: run_id, suffix, energy
        suffix = rid.split("_", 1)[-1] if "_" in rid else rid
        manifest_rows.append((rid, suffix, rec["energy"]))

    payload["manifest_run_ids"] = np.array([m[0] for m in manifest_rows], dtype=object)
    payload["manifest_suffix"]  = np.array([m[1] for m in manifest_rows], dtype=object)
    payload["manifest_energy"]  = np.array([m[2] for m in manifest_rows], dtype=np.float32)
    payload["base_id"]          = np.array(bid)

    saver(out_path, **payload)
    return out_path


# =============================
# Main (process by base id)
# =============================

def main():
    # Build the folder list
    run_dirs = read_run_dirs_from_txt(paths_file)
    particle_dirs_all = [os.path.join(rd, "particles") for rd in run_dirs]
    particle_dirs_all = [d for d in particle_dirs_all if os.path.isdir(d)]

    if not particle_dirs_all:
        raise RuntimeError("No 'particles' directories found from the provided paths file.")

    # Map base id -> list of particles dirs
    # Derive run_id = basename(run_dir); base id = split at first underscore
    groups = defaultdict(list)
    for pdir in particle_dirs_all:
        run_dir = os.path.dirname(pdir)
        rid = os.path.basename(run_dir)
        bid = base_id(rid)
        groups[bid].append(pdir)

    # Optional: deterministic yet randomized traversal order of base ids
    bids = list(groups.keys())
    random.shuffle(bids)

    # If limiting by max_folders, trim traversal to the first N particle dirs cumulatively
    # while respecting base-id boundaries when possible.
    if max_folders is not None and max_folders > 0:
        limited_bids = []
        count = 0
        for bid in bids:
            size = len(groups[bid])
            if count + size <= max_folders:
                limited_bids.append(bid)
                count += size
            else:
                # If we haven't added any yet or must partially include this bid,
                # take as many as we can from this last bid (still process base-id fully if room allows).
                if count == 0:
                    # fall back: include as many as fit from this bid only
                    groups[bid] = groups[bid][:max_folders]
                    limited_bids.append(bid)
                    count = max_folders
                break
        bids = limited_bids

    bins_list = [32]

    # Stats
    stats = {
        "ok": 0,
        "missing_config_or_parquet": 0,
        "invalid_parquet": 0,
        "bad_particles_config": 0,
        "bad_geometry": 0,
        "read_parquet_failed": 0,
        "missing_xy": 0,
        "empty_parquet": 0,
        "other": 0,
    }

    total_dirs = sum(len(groups[bid]) for bid in bids)
    processed_dirs = 0

    parent_for_npz = _ensuredir(os.path.join(npz_dirname)) if save_npzs_grouped else None

    # Process base id by base id
    for bid in tqdm(bids, desc="Processing base ids", unit="baseid"):
        records_for_bid = []
        pdirs = groups[bid]

        for particles_dir in tqdm(pdirs, leave=False, desc=f"{bid}", unit="run"):
            try:
                rec, status = process_one_folder(
                    particles_dir,
                    bins_list=bins_list,
                    use_normalized_xy=use_normalized_xy,
                    log_counts=log_counts,
                    save_npys=save_npys,
                )
                if status == "ok" and rec is not None:
                    records_for_bid.append(rec)
                    stats["ok"] += 1
                else:
                    stats[status] = stats.get(status, 0) + 1
            except Exception as e:
                warn(f"Skipping {particles_dir} (unexpected error: {type(e).__name__}: {e})")
                stats["other"] += 1
            processed_dirs += 1

        # Immediately emit grouped NPZ for this base id
        if save_npzs_grouped and records_for_bid:
            try:
                write_group_npz(bid, records_for_bid, parent_for_npz, compressed=npz_compressed)
            except Exception as e:
                warn(f"Failed to write NPZ for base id {bid}: {type(e).__name__}: {e}")

        # Free memory held by arrays from this group
        del records_for_bid

    # =============================
    # Summary
    # =============================
    warn("\n=== Summary ===")
    warn(f"Base ids considered: {len(bids)}")
    warn(f"Total particle dirs considered: {total_dirs}")
    warn(f"Processed OK    : {stats['ok']}")
    for k, v in stats.items():
        if k == "ok" or v == 0:
            continue
        warn(f"Skipped ({k}): {v}")


if __name__ == "__main__":
    main()

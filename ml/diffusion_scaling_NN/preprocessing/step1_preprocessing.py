#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 Preprocessing: Transform particle coordinates from local to global frame.

This script processes raw simulation directories and combines particle data from multiple planes.

Summary of steps:
1. Read simulation directory paths from a text file
2. For each simulation directory:
   - Find all particles*/particles.parquet files (one per detector plane)
   - Load parquet files with awkward arrays
   - Filter particles by PDG code (keep electrons, muons, photons: 11, 13, 22)
   - Extract simulation parameters (energy, zenith, azimuth) from config files
   - Load plane geometry (center, normal, x-axis, y-axis) from config.yaml
   - Transform particle coordinates from local plane frame to global 3D coordinates
   - Add metadata columns: plane_index, class_id, energy, angles
3. Concatenate all planes into a single parquet file per simulation
4. Validate output files (check all 24 planes present with ≥10 particles each)
5. Write valid file paths to a manifest file (valid_files.txt) using file locking
6. Support batch processing with --batch-start and --batch-end for parallel execution

Output: Combined parquet files with global (x, y, z) coordinates and metadata for each simulation.
"""

import os
import re
import argparse
import math
import shlex
import numpy as np
import awkward as ak
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import fcntl
import time


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
            else:  # --key value  OR  --flag
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


def radians_sin_cos(angle):
    """
    Compute sin/cos assuming radians by default.
    If |angle| looks like degrees (> 2π), auto-convert from degrees.
    """
    if np.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        angle = math.radians(angle)
    return math.sin(angle), math.cos(angle)


# =============================
# File locking helper
# =============================

def append_to_manifest_safely(manifest_path: str, file_paths: list, max_retries=10):
    """
    Append file paths to manifest using file locking to prevent race conditions.
    """
    if not file_paths:
        return
    
    for attempt in range(max_retries):
        try:
            with open(manifest_path, 'a') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for path in file_paths:
                        f.write(f"{path}\n")
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return  # Success
        except IOError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: Failed to write to manifest after {max_retries} attempts: {e}")


# =============================
# Plane index helpers
# =============================

def get_plane_index(folder_name: str) -> int:
    """
    Returns plane index for particle folders.
    - "particles" -> 23
    - "particlesN" -> N-1
    """
    if folder_name == "particles":
        return 23
    
    m = re.match(r"^particles(\d+)$", folder_name)
    if m:
        return int(m.group(1)) - 1
    
    return -1


# =============================
# Processing functions
# =============================

def process_particle_folder(folder_path: str, plane_idx: int, class_id: int, run_dir: str):
    """
    Process a single particles folder:
    - Read parquet file with awkward
    - Filter by PDG (11, 13, 22)
    - Load geometry from config
    - Transform coordinates to global frame
    - Extract simulation parameters (energy, zenith, azimuth)
    - Return table with plane_index and class_id
    """
    parquet_path = os.path.join(folder_path, "particles.parquet")
    config_path = os.path.join(folder_path, "config.yaml")
    base_config_path = os.path.join(run_dir, "config.yaml")
    
    # Check files exist
    if not os.path.exists(parquet_path):
        return None, "missing_parquet"
    if not os.path.exists(config_path):
        return None, "missing_config"
    
    # Load parquet with awkward
    try:
        akf = ak.from_parquet(parquet_path)
    except Exception as e:
        return None, f"read_error: {e}"
    
    if len(akf) == 0:
        return None, "empty_data"
    
    # Check required columns
    if "x" not in akf.fields or "y" not in akf.fields:
        return None, "missing_xy"
    if "pdg" not in akf.fields:
        return None, "missing_pdg"
    
    # Filter by PDG: keep only electrons (11), muons (13), and photons (22)
    mask = (akf["pdg"] == 11) | (akf["pdg"] == 13) | (akf["pdg"] == 22)
    akf = akf[mask]
    
    if len(akf) == 0:
        return None, "no_valid_pdg"
    
    # Load configs
    try:
        with open(config_path) as f:
            particles_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return None, f"bad_config: {e}"
    
    base_cfg = {}
    try:
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_cfg = yaml.safe_load(f) or {}
    except Exception:
        base_cfg = {}
    
    # Extract simulation parameters
    args_str = base_cfg.get("args") or particles_cfg.get("args") or ""
    arg_dict = parse_args_string(args_str)
    
    p_energy = get_float_flag(args_str, arg_dict, ["energy"], default=0.0)
    zenith = get_float_flag(args_str, arg_dict, ["zenith"], default=0.0)
    azimuth = get_float_flag(args_str, arg_dict, ["azimuth"], default=0.0)
    
    # Normalize energy and compute angles
    p_energy_norm = p_energy / 5e7
    sin_zenith, cos_zenith = radians_sin_cos(zenith)
    sin_azimuth, cos_azimuth = radians_sin_cos(azimuth)
    
    # Extract geometry
    try:
        center = np.array(particles_cfg["plane"]["center"])
        zhat = np.array(particles_cfg["plane"]["normal"])
        xhat = np.array(particles_cfg["x-axis"])
        yhat = np.array(particles_cfg["y-axis"])
    except Exception as e:
        return None, f"bad_geometry: {e}"
    
    # Build transformation matrix
    mat = np.array([xhat, yhat, zhat])
    
    # Transform coordinates
    xyzs = np.zeros((3, len(akf)))
    
    for idx, (x, y) in enumerate(zip(akf["x"], akf["y"])):
        xyz = np.array([x, y, 0.0])
        xyzs[:, idx] = np.matmul(mat, xyz) + center
    
    # Build new table with transformed coordinates
    new_columns = {
        'x': pa.array(xyzs[0], type=pa.float64()),
        'y': pa.array(xyzs[1], type=pa.float64()),
        'z': pa.array(xyzs[2], type=pa.float64()),
        'pdg': pa.array(np.asarray(akf["pdg"]), type=pa.int32()),
        'time': pa.array(np.asarray(akf["time"]), type=pa.float64()),
        'kinetic_energy': pa.array(np.asarray(akf["kinetic_energy"]), type=pa.float64()),
        'plane_index': pa.array([plane_idx] * len(akf), type=pa.int32()),
        'class_id': pa.array([class_id] * len(akf), type=pa.int32()),
        'p_energy': pa.array([p_energy_norm] * len(akf), type=pa.float32()),
        'sin_zenith': pa.array([sin_zenith] * len(akf), type=pa.float32()),
        'cos_zenith': pa.array([cos_zenith] * len(akf), type=pa.float32()),
        'sin_azimuth': pa.array([sin_azimuth] * len(akf), type=pa.float32()),
        'cos_azimuth': pa.array([cos_azimuth] * len(akf), type=pa.float32()),
    }
    
    # Create new table
    new_table = pa.table(new_columns)
    
    return new_table, "ok"


def process_simulation_directory(sim_dir: str, output_dir: str, class_id: int):
    """
    Process a single simulation directory:
    - Find all particles*/particles.parquet files
    - Transform coordinates for each plane
    - Combine with plane_index and class_id columns
    - Save as single parquet file
    """
    sim_name = os.path.basename(sim_dir)
    output_path = os.path.join(output_dir, f"{sim_name}.parquet")
    
    # Find all particle folders
    particle_folders = []
    for item in os.listdir(sim_dir):
        full_path = os.path.join(sim_dir, item)
        if os.path.isdir(full_path) and re.match(r"^particles(\d+)?$", item):
            plane_idx = get_plane_index(item)
            if plane_idx >= 0:
                particle_folders.append((plane_idx, full_path))
    
    if not particle_folders:
        return None, 0, 0
    
    # Sort by plane index
    particle_folders.sort(key=lambda x: x[0])
    
    # Process each folder
    tables = []
    success_count = 0
    fail_count = 0
    
    for plane_idx, folder_path in particle_folders:
        table, status = process_particle_folder(folder_path, plane_idx, class_id, sim_dir)
        
        if status == "ok" and table is not None:
            tables.append(table)
            success_count += 1
        else:
            fail_count += 1
    
    if not tables:
        return None, success_count, fail_count
    
    # Concatenate all tables
    combined_table = pa.concat_tables(tables)
    
    # Write combined parquet
    pq.write_table(combined_table, output_path)
    
    return output_path, success_count, fail_count


def worker_process_simulation(sim_dir: str, output_dir: str, class_id: int):
    """
    Worker function to process a single simulation.
    Returns (sim_dir, success, planes_ok, planes_failed, error_msg)
    """
    try:
        result, planes_ok, planes_failed = process_simulation_directory(
            sim_dir, output_dir, class_id
        )
        
        if result:
            return (sim_dir, True, planes_ok, planes_failed, None)
        else:
            return (sim_dir, False, planes_ok, planes_failed, "no_output")
    
    except Exception as e:
        return (sim_dir, False, 0, 0, f"{type(e).__name__}: {e}")


def read_sim_dirs_from_txt(txt_path: str, start_idx: int = None, end_idx: int = None):
    """
    Read simulation directories from text file.
    Optionally slice to [start_idx:end_idx] range.
    """
    sims = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Paths file not found: {txt_path}")
    
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            sims.append(os.path.normpath(p))
    
    # Apply slicing if specified
    if start_idx is not None or end_idx is not None:
        sims = sims[start_idx:end_idx]
    
    return sims


def main():
    parser = argparse.ArgumentParser(
        description="Transform and combine particles parquet files with plane indices (multiprocessing)"
    )
    parser.add_argument("paths_file", help="Text file with simulation directories (one per line)")
    parser.add_argument("--class-id", type=int, required=True,
                        help="Class ID to assign to all particles in this dataset")
    parser.add_argument("--out-dir", default="./combined_parquet",
                        help="Directory to store combined parquet files")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--batch-start", type=int, default=None,
                        help="Start index for batch processing (inclusive)")
    parser.add_argument("--batch-end", type=int, default=None,
                        help="End index for batch processing (exclusive)")
    args = parser.parse_args()
    
    # Read simulation directories (with optional slicing)
    sim_dirs = read_sim_dirs_from_txt(args.paths_file, args.batch_start, args.batch_end)
    
    if not sim_dirs:
        print("No simulations to process in this batch!")
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Determine number of workers
    n_workers = args.workers if args.workers else os.cpu_count()
    print(f"Using {n_workers} worker processes")
    print(f"Processing batch: indices [{args.batch_start}:{args.batch_end}]")
    print(f"Processing {len(sim_dirs)} simulations...")
    
    # Process simulations in parallel
    successful_sims = 0
    failed_sims = 0
    total_planes_ok = 0
    total_planes_failed = 0
    
    # Create partial function with fixed arguments
    worker_func = partial(worker_process_simulation, 
                         output_dir=args.out_dir, 
                         class_id=args.class_id)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        futures = [executor.submit(worker_func, sim_dir) for sim_dir in sim_dirs]
        
        # Process results with progress bar
        for future in tqdm(futures, desc="Processing simulations", unit="sim"):
            try:
                sim_dir, success, planes_ok, planes_failed, error_msg = future.result()
                
                total_planes_ok += planes_ok
                total_planes_failed += planes_failed
                
                if success:
                    successful_sims += 1
                else:
                    failed_sims += 1
                    if error_msg and error_msg != "no_output":
                        print(f"\nError processing {sim_dir}: {error_msg}")
            
            except Exception as e:
                failed_sims += 1
                print(f"\nUnexpected error: {e}")
    
    # Validate output files
    print("\nValidating output files...")
    valid_files = []
    
    for sim_dir in tqdm(sim_dirs, desc="Validating", unit="file"):
        sim_name = os.path.basename(sim_dir)
        output_path = os.path.join(args.out_dir, f"{sim_name}.parquet")
        
        if not os.path.exists(output_path):
            continue
        
        try:
            table = pq.read_table(output_path)
            
            # Check if file has plane_index column
            if 'plane_index' not in table.column_names:
                continue
            
            # Count rows per plane
            plane_indices = table['plane_index'].to_numpy()
            
            # Check if all planes 0-23 are present AND each has at least 10 rows
            valid = True
            for plane_idx in range(24):
                count = np.sum(plane_indices == plane_idx)
                if count < 10:
                    valid = False
                    break
            
            if valid:
                valid_files.append(os.path.abspath(output_path))
        
        except Exception as e:
            print(f"Error validating {output_path}: {e}")
            continue
    
    # Append valid files to shared manifest using file locking
    manifest_path = os.path.join(args.out_dir, "valid_files.txt")
    append_to_manifest_safely(manifest_path, valid_files)
    
    # Summary
    print("\n=== Batch Summary ===")
    print(f"Batch range: [{args.batch_start}:{args.batch_end}]")
    print(f"Simulations processed in this batch: {len(sim_dirs)}")
    print(f"Successful simulations: {successful_sims}")
    print(f"Failed simulations: {failed_sims}")
    print(f"Total planes processed: {total_planes_ok}")
    print(f"Total planes failed: {total_planes_failed}")
    print(f"Valid files from this batch: {len(valid_files)}")
    print(f"Appended to manifest: {manifest_path}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
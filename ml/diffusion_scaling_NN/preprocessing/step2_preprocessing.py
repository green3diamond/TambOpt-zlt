#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 Preprocessing: Create 3D histogram tensors from preprocessed parquet files.

This script converts particle data into multi-channel 2D histograms for each detector plane.

Summary of steps:
1. Read list of preprocessed parquet files from step 1 (from valid_files.txt)
2. For each parquet file and each of 24 detector planes:
   - Extract particle positions (x, y), kinetic energy, and time
   - Remove outliers using configurable methods (IQR, MAD, Z-score, percentile)
   - Compute plane-specific bounding box ranges AFTER outlier removal
   - Create three 2D histograms per plane (bins × bins):
     * Channel 0: Particle density (count)
     * Channel 1: Average kinetic energy per bin
     * Channel 2: Average time per bin
   - Apply Gaussian smoothing to histograms (optional, controlled by --sigma)
3. Stack histograms into tensor shape: (N_samples, 24_planes, 3_channels, H, W)
4. Store bounding box ranges for each plane: (N_samples, 24_planes, 4) [xmin, xmax, ymin, ymax] (only change in comparison to default diffusion model preprocessing)
5. Include simulation metadata: primary energy, zenith/azimuth angles, class ID
6. Save data in batches to manage memory, then combine into final .pt file
7. Clean up temporary batch files

Output: Single PyTorch .pt file containing histograms, bbox_ranges, and metadata for all valid simulations.
"""

import os
import argparse
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.ndimage import gaussian_filter
import gc


def create_2d_histograms(x, y, values, bins=64, x_range=None, y_range=None):
    """Create 2D histogram for given particle positions and values."""
    if x_range is None:
        x_range = (x.min(), x.max())
    if y_range is None:
        y_range = (y.min(), y.max())

    hist, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[x_range, y_range],
        weights=values
    )

    return hist


def calculate_outlier_thresholds_iqr(data, iqr_multiplier=1.5):
    """Calculate outlier thresholds using Interquartile Range (IQR) method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_threshold = q1 - iqr_multiplier * iqr
    upper_threshold = q3 + iqr_multiplier * iqr

    return lower_threshold, upper_threshold


def calculate_outlier_thresholds_mad(data, mad_multiplier=3.0):
    """Calculate outlier thresholds using Median Absolute Deviation (MAD) method."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    lower_threshold = median - mad_multiplier * mad / 0.6745
    upper_threshold = median + mad_multiplier * mad / 0.6745

    return lower_threshold, upper_threshold


def calculate_outlier_thresholds_zscore(data, z_threshold=3.0):
    """Calculate outlier thresholds using Z-score method."""
    mean = np.mean(data)
    std = np.std(data)

    lower_threshold = mean - z_threshold * std
    upper_threshold = mean + z_threshold * std

    return lower_threshold, upper_threshold


def calculate_outlier_thresholds_percentile(data, contamination=0.1):
    """Calculate outlier thresholds using percentile method."""
    lower_percentile = (contamination / 2) * 100
    upper_percentile = (1 - contamination / 2) * 100

    lower_threshold = np.percentile(data, lower_percentile)
    upper_threshold = np.percentile(data, upper_percentile)

    return lower_threshold, upper_threshold


def get_automated_thresholds(kinetic_energy, time, method='iqr', **kwargs):
    """Get automated outlier thresholds for kinetic energy and time."""
    methods = {
        'iqr': calculate_outlier_thresholds_iqr,
        'mad': calculate_outlier_thresholds_mad,
        'zscore': calculate_outlier_thresholds_zscore,
        'percentile': calculate_outlier_thresholds_percentile
    }

    if method == 'combined':
        # Use IQR for initial filtering, then MAD for refinement
        ke_lower_iqr, ke_upper_iqr = calculate_outlier_thresholds_iqr(kinetic_energy)
        time_lower_iqr, time_upper_iqr = calculate_outlier_thresholds_iqr(time)

        mask = (
            (kinetic_energy >= ke_lower_iqr) & (kinetic_energy <= ke_upper_iqr) &
            (time >= time_lower_iqr) & (time <= time_upper_iqr)
        )

        # If mask wipes everything, fall back to IQR thresholds
        if np.count_nonzero(mask) == 0:
            ke_lower, ke_upper = ke_lower_iqr, ke_upper_iqr
            time_lower, time_upper = time_lower_iqr, time_upper_iqr
        else:
            ke_lower, ke_upper = calculate_outlier_thresholds_mad(kinetic_energy[mask])
            time_lower, time_upper = calculate_outlier_thresholds_mad(time[mask])
    else:
        threshold_func = methods.get(method, calculate_outlier_thresholds_iqr)
        ke_lower, ke_upper = threshold_func(kinetic_energy, **kwargs)
        time_lower, time_upper = threshold_func(time, **kwargs)

    return ke_lower, ke_upper, time_lower, time_upper


def remove_outliers_auto(x, y, kinetic_energy, time, method='iqr', **kwargs):
    """Remove outliers using automated threshold detection."""
    ke_lower, ke_upper, time_lower, time_upper = get_automated_thresholds(
        kinetic_energy, time, method=method, **kwargs
    )

    mask = (
        (kinetic_energy >= ke_lower) & (kinetic_energy <= ke_upper) &
        (time >= time_lower) & (time <= time_upper)
    )

    return x[mask], y[mask], kinetic_energy[mask], time[mask]


def process_single_file(
    file_path,
    bins=64,
    sigma=1.0,
    min_particles_per_plane=30,
    outlier_method='iqr',
    outlier_params=None
):
    """
    Process a single parquet file and create 3D histograms for all planes.

    Returns:
    --------
    dict with keys:
        'histograms': torch.Tensor (24, 3, bins, bins)  [planes, channels, H, W]
        'bbox_ranges': torch.Tensor (24, 4) [planes, (xmin, xmax, ymin, ymax)]
        'p_energy', 'sin_zenith', 'cos_zenith', 'sin_azimuth', 'cos_azimuth'
        'class_id': int
        'valid': bool
    """
    if outlier_params is None:
        outlier_params = {}

    try:
        table = pq.read_table(file_path)

        x = table['x'].to_numpy()
        y = table['y'].to_numpy()
        kinetic_energy = table['kinetic_energy'].to_numpy()
        time = table['time'].to_numpy()
        plane_index = table['plane_index'].to_numpy()

        # metadata (same for all rows)
        p_energy = float(table['p_energy'][0].as_py())
        sin_zenith = float(table['sin_zenith'][0].as_py())
        cos_zenith = float(table['cos_zenith'][0].as_py())
        sin_azimuth = float(table['sin_azimuth'][0].as_py())
        cos_azimuth = float(table['cos_azimuth'][0].as_py())
        class_id = int(table['class_id'][0].as_py())

        # histograms: (24 planes, 3 channels, bins, bins)
        histograms = np.zeros((24, 3, bins, bins), dtype=np.float32)

        # bbox_ranges: (24 planes, 4) - [xmin, xmax, ymin, ymax] for each plane
        bbox_ranges = np.zeros((24, 4), dtype=np.float32)

        valid = True

        for plane_idx in range(24):
            mask = plane_index == plane_idx
            plane_x = x[mask]
            plane_y = y[mask]
            plane_ke = kinetic_energy[mask]
            plane_time = time[mask]

            if len(plane_x) < min_particles_per_plane:
                valid = False
                continue

            if outlier_method:
                plane_x, plane_y, plane_ke, plane_time = remove_outliers_auto(
                    plane_x, plane_y, plane_ke, plane_time,
                    method=outlier_method,
                    **outlier_params
                )

            if len(plane_x) < min_particles_per_plane:
                valid = False
                continue

            # Compute plane-specific ranges AFTER outlier removal
            x_range = (float(plane_x.min()), float(plane_x.max()))
            y_range = (float(plane_y.min()), float(plane_y.max()))

            # Store bbox ranges: [xmin, xmax, ymin, ymax]
            bbox_ranges[plane_idx] = [x_range[0], x_range[1], y_range[0], y_range[1]]

            # Create 2D histograms
            hist_density = create_2d_histograms(
                plane_x, plane_y,
                np.ones_like(plane_x),
                bins=bins,
                x_range=x_range,
                y_range=y_range
            )

            hist_energy = create_2d_histograms(
                plane_x, plane_y,
                plane_ke,
                bins=bins,
                x_range=x_range,
                y_range=y_range
            )

            hist_time = create_2d_histograms(
                plane_x, plane_y,
                plane_time,
                bins=bins,
                x_range=x_range,
                y_range=y_range
            )

            if sigma > 0:
                hist_density = gaussian_filter(hist_density, sigma=sigma)
                hist_energy = gaussian_filter(hist_energy, sigma=sigma)
                hist_time = gaussian_filter(hist_time, sigma=sigma)

            # averages per bin
            mask_nonzero = hist_density > 0
            hist_energy_avg = np.zeros_like(hist_energy)
            hist_time_avg = np.zeros_like(hist_time)

            hist_energy_avg[mask_nonzero] = hist_energy[mask_nonzero] / hist_density[mask_nonzero]
            hist_time_avg[mask_nonzero] = hist_time[mask_nonzero] / hist_density[mask_nonzero]

            histograms[plane_idx, 0] = hist_density
            histograms[plane_idx, 1] = hist_energy_avg
            histograms[plane_idx, 2] = hist_time_avg

        histograms_torch = torch.from_numpy(histograms)
        bbox_ranges_torch = torch.from_numpy(bbox_ranges)

        return {
            'histograms': histograms_torch,
            'bbox_ranges': bbox_ranges_torch,
            'p_energy': p_energy,
            'sin_zenith': sin_zenith,
            'cos_zenith': cos_zenith,
            'sin_azimuth': sin_azimuth,
            'cos_azimuth': cos_azimuth,
            'class_id': class_id,
            'valid': valid,
            'file_path': file_path
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'file_path': file_path
        }


def worker_process_file(file_path, bins, sigma, min_particles_per_plane,
                        outlier_method, outlier_params):
    """Worker function for multiprocessing."""
    return process_single_file(
        file_path, bins, sigma, min_particles_per_plane,
        outlier_method, outlier_params
    )


def read_file_list(txt_path):
    """Read file paths from text file."""
    files = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File list not found: {txt_path}")

    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            files.append(p)

    return files


def save_batch(batch_samples, batch_idx, batch_dir, args):
    """Save a batch of samples to a separate file in the batch directory."""
    if len(batch_samples) == 0:
        return None

    batch_file = os.path.join(batch_dir, f"batch_{batch_idx:04d}.pt")

    batch_dataset = {
        'histograms': torch.stack([s['histograms'] for s in batch_samples]),
        'bbox_ranges': torch.stack([s['bbox_ranges'] for s in batch_samples]),
        'p_energy': torch.tensor([s['p_energy'] for s in batch_samples], dtype=torch.float32),
        'sin_zenith': torch.tensor([s['sin_zenith'] for s in batch_samples], dtype=torch.float32),
        'cos_zenith': torch.tensor([s['cos_zenith'] for s in batch_samples], dtype=torch.float32),
        'sin_azimuth': torch.tensor([s['sin_azimuth'] for s in batch_samples], dtype=torch.float32),
        'cos_azimuth': torch.tensor([s['cos_azimuth'] for s in batch_samples], dtype=torch.float32),
        'class_id': torch.tensor([s['class_id'] for s in batch_samples], dtype=torch.long),
        'file_paths': [s['file_path'] for s in batch_samples],
    }

    torch.save(batch_dataset, batch_file)
    return batch_file


def combine_batches(batch_files, final_output, args):
    """Combine all batch files into final output."""
    print(f"\nCombining {len(batch_files)} batch files into final dataset...")

    all_histograms = []
    all_bbox_ranges = []
    all_p_energy = []
    all_sin_zenith = []
    all_cos_zenith = []
    all_sin_azimuth = []
    all_cos_azimuth = []
    all_class_id = []
    all_file_paths = []

    for batch_file in tqdm(batch_files, desc="Loading batches"):
        if not os.path.exists(batch_file):
            print(f"WARNING: Batch file not found: {batch_file}")
            continue

        batch_data = torch.load(batch_file)
        all_histograms.append(batch_data['histograms'])
        all_bbox_ranges.append(batch_data['bbox_ranges'])
        all_p_energy.append(batch_data['p_energy'])
        all_sin_zenith.append(batch_data['sin_zenith'])
        all_cos_zenith.append(batch_data['cos_zenith'])
        all_sin_azimuth.append(batch_data['sin_azimuth'])
        all_cos_azimuth.append(batch_data['cos_azimuth'])
        all_class_id.append(batch_data['class_id'])
        all_file_paths.extend(batch_data['file_paths'])

        # Clean up batch file immediately
        del batch_data
        gc.collect()

    if len(all_histograms) == 0:
        raise ValueError("No valid batch files found!")

    final_dataset = {
        'histograms': torch.cat(all_histograms, dim=0),
        'bbox_ranges': torch.cat(all_bbox_ranges, dim=0),
        'p_energy': torch.cat(all_p_energy, dim=0),
        'sin_zenith': torch.cat(all_sin_zenith, dim=0),
        'cos_zenith': torch.cat(all_cos_zenith, dim=0),
        'sin_azimuth': torch.cat(all_sin_azimuth, dim=0),
        'cos_azimuth': torch.cat(all_cos_azimuth, dim=0),
        'class_id': torch.cat(all_class_id, dim=0),
        'file_paths': all_file_paths,
        'metadata': {
            'bins': args.bins,
            'sigma': args.sigma,
            'min_particles_per_plane': args.min_particles,
            'outlier_method': args.outlier_method if args.outlier_method != 'none' else None,
            'outlier_params': {},
            'n_samples': len(all_file_paths),
        }
    }

    torch.save(final_dataset, final_output)

    # Clean up batch files
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            try:
                os.remove(batch_file)
            except:
                pass

    return final_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D histogram tensors from preprocessed parquet files"
    )
    parser.add_argument("file_list", help="Text file with parquet file paths (one per line)")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--bins", type=int, default=64,
                        help="Number of bins for 2D histograms (default: 64)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian filter sigma (default: 1.0, use 0 for no filtering)")
    parser.add_argument("--min-particles", type=int, default=30,
                        help="Minimum particles per plane (default: 30)")
    parser.add_argument("--outlier-method", type=str, default='iqr',
                        choices=['iqr', 'mad', 'zscore', 'percentile', 'combined', 'none'],
                        help="Outlier detection method (default: iqr)")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5,
                        help="IQR multiplier for outlier detection (default: 1.5)")
    parser.add_argument("--mad-multiplier", type=float, default=3.0,
                        help="MAD multiplier for outlier detection (default: 3.0)")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Z-score threshold for outlier detection (default: 3.0)")
    parser.add_argument("--contamination", type=float, default=0.1,
                        help="Contamination rate for percentile method (default: 0.1)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 2)")
    parser.add_argument("--batch-size", type=int, default=3000,
                        help="Number of samples per batch (default: 3000)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Number of files to submit at once (default: 3000)")

    args = parser.parse_args()

    file_paths = read_file_list(args.file_list)
    print(f"Found {len(file_paths)} files to process")

    outlier_method = None if args.outlier_method == 'none' else args.outlier_method
    outlier_params = {}

    if outlier_method == 'iqr':
        outlier_params['iqr_multiplier'] = args.iqr_multiplier
    elif outlier_method == 'mad':
        outlier_params['mad_multiplier'] = args.mad_multiplier
    elif outlier_method == 'zscore':
        outlier_params['z_threshold'] = args.z_threshold
    elif outlier_method == 'percentile':
        outlier_params['contamination'] = args.contamination

    print(f"Outlier detection method: {outlier_method if outlier_method else 'None'}")
    if outlier_params:
        print(f"Parameters: {outlier_params}")

    # Leave some CPUs for main process and system
    n_workers = args.workers if args.workers else max(1, os.cpu_count() - 2)
    print(f"Using {n_workers} worker processes")
    print(f"Batch size: {args.batch_size} samples")
    print(f"Chunk size: {args.chunk_size} files")

    # Create temporary directory for batch files
    output_dir = os.path.dirname(os.path.abspath(args.output))
    batch_dir = os.path.join(output_dir, "temp_batches")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"Batch directory: {batch_dir}")

    batch_samples = []
    batch_files = []
    invalid_count = 0
    batch_idx = 0

    worker_func = partial(
        worker_process_file,
        bins=args.bins,
        sigma=args.sigma,
        min_particles_per_plane=args.min_particles,
        outlier_method=outlier_method,
        outlier_params=outlier_params
    )

    # Process files in chunks to avoid memory issues
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process in chunks
        for chunk_start in range(0, len(file_paths), args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, len(file_paths))
            chunk_files = file_paths[chunk_start:chunk_end]
            
            print(f"\nProcessing chunk {chunk_start//args.chunk_size + 1}/{(len(file_paths)-1)//args.chunk_size + 1} "
                  f"({chunk_start}-{chunk_end} of {len(file_paths)} files)")
            
            # Submit only this chunk's futures
            futures = {executor.submit(worker_func, fp): fp for fp in chunk_files}
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {chunk_start//args.chunk_size + 1}"):
                try:
                    result = future.result()
                    total_processed += 1

                    if result.get('valid', False):
                        batch_samples.append(result)
                        
                        # Save batch when it reaches the batch size
                        if len(batch_samples) >= args.batch_size:
                            batch_file = save_batch(batch_samples, batch_idx, batch_dir, args)
                            if batch_file:
                                batch_files.append(batch_file)
                                print(f"\nSaved batch {batch_idx} with {len(batch_samples)} samples")
                            batch_samples = []
                            batch_idx += 1
                            gc.collect()  # Force garbage collection
                    else:
                        invalid_count += 1
                        if 'error' in result:
                            print(f"\nError in {os.path.basename(result['file_path'])}: {result['error']}")

                except Exception as e:
                    invalid_count += 1
                    print(f"\nUnexpected error: {e}")
            
            # Clear futures for this chunk
            del futures
            gc.collect()

    # Save any remaining samples
    if len(batch_samples) > 0:
        batch_file = save_batch(batch_samples, batch_idx, batch_dir, args)
        if batch_file:
            batch_files.append(batch_file)
            print(f"\nSaved final batch {batch_idx} with {len(batch_samples)} samples")

    print(f"\n=== Processing Summary ===")
    print(f"Total files processed: {total_processed}")
    print(f"Total batches created: {len(batch_files)}")
    print(f"Invalid samples: {invalid_count}")

    if len(batch_files) == 0:
        print("\nERROR: No valid samples to save!")
        return

    # Combine all batches into final output
    final_dataset = combine_batches(batch_files, args.output, args)
    
    # Clean up temporary directory
    try:
        os.rmdir(batch_dir)
        print(f"Removed temporary batch directory: {batch_dir}")
    except Exception as e:
        print(f"Could not remove batch directory: {e}")

    print(f"\n=== SUCCESS ===")
    print(f"Saved {final_dataset['histograms'].shape[0]} samples to {args.output}")
    print(f"Dataset shape: {final_dataset['histograms'].shape}")


if __name__ == "__main__":
    main()
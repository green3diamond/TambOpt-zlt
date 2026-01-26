#!/usr/bin/env python3
"""
Evaluation script for FNN bbox range prediction using PyTorch Lightning checkpoints
Generates bbox predictions and saves results
"""
import os
import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from TambOpt.ml.scaling_NN.FNN.lightning_training_fnn import PlaneDataset, PlaneFNNModule


def load_standardization_stats(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load bbox standardization statistics"""
    stats_path = os.path.join(data_dir, "global_bbox_stats.pt")
    if not os.path.exists(stats_path):
        # Try alternative path
        stats_path = os.path.join(data_dir, "..", "global_bbox_stats.pt")

    if not os.path.exists(stats_path):
        # Try global stats from lightning training
        stats_path = os.path.join(data_dir, "global_bbox_stats.pt")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Standardization stats not found. Tried multiple locations.")

    stats = torch.load(stats_path, map_location="cpu")

    # Handle different stat formats
    if "mean" in stats and "std" in stats:
        mean = stats["mean"]
        std = stats["std"]
    elif "bbox_mean" in stats and "bbox_std" in stats:
        # Global stats format (scalars)
        mean = stats["bbox_mean"]
        std = stats["bbox_std"]
        # Convert to tensor if needed
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
            std = torch.tensor(std)
    else:
        raise ValueError(f"Unrecognized stats format in {stats_path}")

    print(f"Loaded standardization stats from: {stats_path}")
    print(f"  Mean shape: {mean.shape if isinstance(mean, torch.Tensor) and mean.ndim > 0 else 'scalar'}")
    print(f"  Std shape: {std.shape if isinstance(std, torch.Tensor) and std.ndim > 0 else 'scalar'}")

    return mean, std


def denormalize_standardized(standardized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Convert from standardized (zero mean, unit variance) back to original scale

    Args:
        standardized: (B, 24, 4) or (24, 4) - standardized bounding boxes
        mean: (24, 4) or scalar - per-plane/per-coordinate means or global mean
        std: (24, 4) or scalar - per-plane/per-coordinate stds or global std

    Returns:
        denormalized: same shape as input, in original scale
    """
    # Handle scalar stats (global normalization)
    if mean.ndim == 0 or (isinstance(mean, torch.Tensor) and mean.numel() == 1):
        return standardized * std + mean

    # Handle per-plane, per-coordinate stats
    if standardized.dim() == 2:
        # (24, 4)
        return standardized * std + mean
    elif standardized.dim() == 3:
        # (B, 24, 4)
        return standardized * std[None, :, :] + mean[None, :, :]
    else:
        raise ValueError(f"Unexpected dimension: {standardized.dim()}")


def compute_bbox_metrics(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for bbox prediction

    Args:
        pred_bbox: (B, 24, 4) predicted bboxes
        gt_bbox: (B, 24, 4) ground truth bboxes

    Returns:
        Dictionary of metrics
    """
    # Mean absolute error
    mae = torch.abs(pred_bbox - gt_bbox).mean().item()

    # Mean squared error
    mse = ((pred_bbox - gt_bbox) ** 2).mean().item()

    # Root mean squared error
    rmse = np.sqrt(mse)

    # Per-coordinate MAE
    mae_xmin = torch.abs(pred_bbox[:, :, 0] - gt_bbox[:, :, 0]).mean().item()
    mae_xmax = torch.abs(pred_bbox[:, :, 1] - gt_bbox[:, :, 1]).mean().item()
    mae_ymin = torch.abs(pred_bbox[:, :, 2] - gt_bbox[:, :, 2]).mean().item()
    mae_ymax = torch.abs(pred_bbox[:, :, 3] - gt_bbox[:, :, 3]).mean().item()

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mae_xmin": mae_xmin,
        "mae_xmax": mae_xmax,
        "mae_ymin": mae_ymin,
        "mae_ymax": mae_ymax,
    }


def visualize_bbox_comparison(pred_bbox: np.ndarray, gt_bbox: np.ndarray,
                              sample_idx: int, class_id: int, out_dir: str):
    """
    Visualize predicted vs ground truth bboxes for all planes

    Args:
        pred_bbox: (24, 4) predicted bboxes
        gt_bbox: (24, 4) ground truth bboxes
        sample_idx: sample index for filename
        class_id: particle class ID
        out_dir: output directory
    """
    fig, axes = plt.subplots(6, 4, figsize=(16, 20))
    fig.suptitle(f"Sample {sample_idx} - Class {class_id} - Bbox Comparison (FNN)", fontsize=16)

    for plane_idx in range(24):
        row = plane_idx // 4
        col = plane_idx % 4
        ax = axes[row, col]

        # Extract bbox coordinates
        pred_xmin, pred_xmax, pred_ymin, pred_ymax = pred_bbox[plane_idx]
        gt_xmin, gt_xmax, gt_ymin, gt_ymax = gt_bbox[plane_idx]

        # Plot ground truth bbox (blue)
        ax.plot([gt_xmin, gt_xmax, gt_xmax, gt_xmin, gt_xmin],
                [gt_ymin, gt_ymin, gt_ymax, gt_ymax, gt_ymin],
                'b-', linewidth=2, label='GT')

        # Plot predicted bbox (red)
        ax.plot([pred_xmin, pred_xmax, pred_xmax, pred_xmin, pred_xmin],
                [pred_ymin, pred_ymin, pred_ymax, pred_ymax, pred_ymin],
                'r--', linewidth=2, label='Pred')

        ax.set_title(f'Plane {plane_idx}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"sample_{sample_idx:04d}_class_{class_id}_bbox_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate(args):
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load standardization stats
    mean, std = load_standardization_stats(args.data_dir)
    mean = mean.to(device)
    std = std.to(device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = PlaneDataset(
        chunk_dir=args.data_dir,
        split='test',
        cache_size=args.cache_size,
        prewarm_cache=False,
        bbox_mean=None,
        bbox_std=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # Load model from checkpoint
    print(f"\nLoading model from checkpoint: {args.ckpt}")
    pl_module = PlaneFNNModule.load_from_checkpoint(
        args.ckpt,
        map_location=device
    )
    pl_module.eval()
    pl_module.to(device)

    # Apply EMA weights if available
    if hasattr(pl_module, 'model_ema') and pl_module.model_ema is not None:
        print("Using EMA weights for evaluation")
        for name, param in pl_module.model.named_parameters():
            if name in pl_module.model_ema:
                param.data = pl_module.model_ema[name].to(device)

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Evaluation loop
    all_metrics = []
    all_predictions = []
    all_ground_truths = []
    total_time = 0
    num_samples = 0

    print("\nStarting evaluation...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if num_samples >= args.num_samples:
            break

        # Get batch data
        gt_bbox_std = batch['bboxes'].to(device)  # (B, 24, 4) - standardized
        p_energy = batch['p_energy'].to(device)
        sin_zenith = batch['sin_zenith'].to(device)
        cos_zenith = batch['cos_zenith'].to(device)
        sin_azimuth = batch['sin_azimuth'].to(device)
        cos_azimuth = batch['cos_azimuth'].to(device)
        class_id = batch['class_id'].to(device)

        B = gt_bbox_std.shape[0]
        take = min(B, args.num_samples - num_samples)

        # Truncate batch if needed
        if take < B:
            gt_bbox_std = gt_bbox_std[:take]
            p_energy = p_energy[:take]
            sin_zenith = sin_zenith[:take]
            cos_zenith = cos_zenith[:take]
            sin_azimuth = sin_azimuth[:take]
            cos_azimuth = cos_azimuth[:take]
            class_id = class_id[:take]
            B = take

        # Generate predictions
        start_time = time.time()

        # Forward pass - direct prediction (no diffusion sampling needed)
        pred_bbox_flat_std = pl_module.model(
            p_energy,
            class_id,
            sin_zenith,
            cos_zenith,
            sin_azimuth,
            cos_azimuth
        )  # (B, 96) - standardized

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # Reshape to (B, 24, 4)
        pred_bbox_std = pred_bbox_flat_std.reshape(B, 24, 4)

        # Denormalize
        pred_bbox = denormalize_standardized(pred_bbox_std, mean, std)
        gt_bbox = gt_bbox_std

        # Compute metrics
        metrics = compute_bbox_metrics(pred_bbox, gt_bbox)
        all_metrics.append(metrics)

        # Store predictions
        all_predictions.append(pred_bbox.cpu().numpy())
        all_ground_truths.append(gt_bbox.cpu().numpy())

        # Visualize first few samples
        if num_samples < args.num_visualize:
            for i in range(min(B, args.num_visualize - num_samples)):
                visualize_bbox_comparison(
                    pred_bbox[i].cpu().numpy(),
                    gt_bbox[i].cpu().numpy(),
                    num_samples + i,
                    int(class_id[i].item()),
                    viz_dir
                )

        num_samples += B

        # Print batch statistics
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}: MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}")
            print(f"  Time: {elapsed_time:.4f}s ({elapsed_time/B:.6f}s per sample)")

    # Aggregate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS (FNN)")
    print("="*60)
    print(f"Samples evaluated: {num_samples}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per sample: {total_time/num_samples:.6f}s")
    print("\nMetrics:")

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        print(f"  {key}: {avg_metrics[key]:.6f}")

    # Save results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    results_path = os.path.join(args.out_dir, "evaluation_results.npz")
    np.savez(
        results_path,
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        metrics=avg_metrics,
        config=vars(args)
    )

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {viz_dir}")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FNN bbox prediction model')

    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='./eval_results_fnn',
                       help='Output directory for results')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--num_visualize', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--cache_size', type=int, default=10,
                       help='Dataset cache size')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("="*60)
    print("Evaluation Configuration (FNN):")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60)

    evaluate(args)

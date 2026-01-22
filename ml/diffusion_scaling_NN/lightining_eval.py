#!/usr/bin/env python3
# eval_ddim_allplanes_1d_xy_rgb.py
#
# Autoregressively generate ALL 24 planes with DDIM, then for each sample
# save ONE figure per event with ONLY 1D projections:
#   - 24 rows (planes 0..23)
#   - 6 columns: [Den-X, Den-Y, Eng-X, Eng-Y, Tim-X, Tim-Y]
# Each subplot overlays GT vs Pred.
# Includes denormalization using training statistics.
#
# Key fixes vs your version:
#  1) Correct X/Y projections for (H,W) maps:
#        X projection (len W) = sum over Y = axis=0
#        Y projection (len H) = sum over X = axis=1
#  2) Clamp density >= 0 after denormalization before using density * avg
#  3) Mask empty bins when reconstructing sum maps (optional safety)
#  4) Clean up naming: out_path instead of out_png, and save as PDF if extension is .pdf

#global values
# python /n/home04/hhanif/tam/diffusion_ml/lightining_eval.py    --data_dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_3rd_step/     --ckpt /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/checkpoints/tam_unet/epoch_epoch=839-val_loss_val_loss=0.0340.ckpt     --stats_path /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_3rd_step/standardization_stats_train_only.pt     --out_dir /n/home04/hhanif/tam/plots/lighting_eval_plots_w2_ddim50     --num_samples 100     --ddim_steps 50     --eta 0.0     --guidance_w 2

import os
import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from lightning_training import PlaneDataset, PlaneDiffusionModule
from diffusion import DDIMSamplerPlanes


def get_particle_type_name(class_id: int) -> str:
    particle_types = {
        1: r"Lepton ($e^\pm$)",
        2: r"Charged hadron ($\pi^\pm$)",
        3: r"Neutral Hadron ($\pi^0$)",
    }
    return particle_types.get(class_id, f"Unknown (class {class_id})")


def retrieve_angles_from_sincos(sin_zenith, cos_zenith, sin_azimuth, cos_azimuth):
    # Zenith in [0, pi]
    zenith_rad = np.arctan2(sin_zenith, cos_zenith)
    if zenith_rad < 0:
        zenith_rad += 2 * np.pi
    if zenith_rad > np.pi:
        zenith_rad = 2 * np.pi - zenith_rad

    # Azimuth in [0, 2pi)
    azimuth_rad = np.arctan2(sin_azimuth, cos_azimuth)
    if azimuth_rad < 0:
        azimuth_rad += 2 * np.pi

    return zenith_rad, azimuth_rad


def load_standardization_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Standardization stats not found: {stats_path}")

    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"]  # (24, 3)
    std = stats["std"]    # (24, 3)

    print(f"Loaded standardization stats from: {stats_path}")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")

    return mean, std


@torch.no_grad()
def denormalize_histograms(
    histograms: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    # x_orig = x_norm * std + mean
    if histograms.ndim == 5:
        mean_b = mean[None, :, :, None, None]
        std_b  = std[None, :, :, None, None]
    elif histograms.ndim == 4:
        mean_b = mean[:, :, None, None]
        std_b  = std[:, :, None, None]
    else:
        raise ValueError(f"Expected 4 or 5 dims, got {histograms.ndim}")

    return histograms * std_b + mean_b


def seed_all(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    module = PlaneDiffusionModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    module.to(device)
    return module


@torch.no_grad()
def ddim_generate_all_planes(
    net: torch.nn.Module,
    sampler: DDIMSamplerPlanes,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    start_time = time.time()

    gt_all = batch["planes"].to(device)  # (B,24,3,H,W)
    B, P, C, H, W = gt_all.shape
    assert P == 24 and C == 3, f"Expected (B,24,3,H,W), got {gt_all.shape}"

    p_energy = batch["p_energy"].to(device)
    class_id = batch["class_id"].to(device)
    sin_zenith = batch["sin_zenith"].to(device)
    cos_zenith = batch["cos_zenith"].to(device)
    sin_azimuth = batch["sin_azimuth"].to(device)
    cos_azimuth = batch["cos_azimuth"].to(device)

    pred_all = torch.zeros_like(gt_all)
    past = torch.zeros((B, 3, H, W), device=device)

    for plane_idx in range(P):
        plane_idx_tensor = torch.full((B,), plane_idx, device=device, dtype=torch.long)
        x = torch.randn((B, 3, H, W), device=device)

        pred = sampler(
            x,
            p_energy,
            class_id,
            sin_zenith,
            cos_zenith,
            sin_azimuth,
            cos_azimuth,
            plane_idx_tensor,
            past,
        )
        pred_all[:, plane_idx] = pred
        past = pred

    elapsed_time = time.time() - start_time
    return gt_all, pred_all, elapsed_time


def plot_1d_projection(
    ax,
    gt_projection,
    pred_projection,
    xlabel,
    force_nonneg=False,
    gaussian_sigma=1.0,
):
    gt_projection = np.asarray(gt_projection, dtype=np.float64)
    pred_projection = np.asarray(pred_projection, dtype=np.float64)
    coords = np.arange(len(gt_projection))

    ax.plot(coords, gt_projection, color="blue", linewidth=1.1, alpha=0.9, label="GT")
    ax.plot(coords, pred_projection, color="red", linewidth=1.1, linestyle="--", alpha=0.9, label="Pred")

    if gaussian_sigma and gaussian_sigma > 0:
        gt_smoothed = gaussian_filter1d(gt_projection, sigma=float(gaussian_sigma), mode="nearest")
        pred_smoothed = gaussian_filter1d(pred_projection, sigma=float(gaussian_sigma), mode="nearest")
        ax.plot(coords, gt_smoothed, color="cyan", linewidth=1.4, alpha=0.7, linestyle="-")
        ax.plot(coords, pred_smoothed, color="orange", linewidth=1.4, alpha=0.7, linestyle="--")

    ax.set_xlim(0, len(gt_projection) - 1)
    if force_nonneg:
        ax.set_ylim(bottom=0)

    ax.margins(x=0.02)
    ax.tick_params(labelsize=6)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _reconstruct_sum_map(density: np.ndarray, avg: np.ndarray, clamp_density: bool = True) -> np.ndarray:
    """
    Reconstruct sum map from density and per-bin average: sum = density * avg.
    Safety:
      - clamp density >= 0 (recommended after denorm)
      - zero out bins with (density <= 0) to avoid weird negatives / noise
    """
    den = density
    if clamp_density:
        den = np.clip(den, 0.0, None)

    out = den * avg
    out = np.where(den > 0.0, out, 0.0)
    return out


def plot_allplanes_1d_xy_rgb(
    gt_all: torch.Tensor,
    pred_all: torch.Tensor,
    out_path: str,
    title: str,
    force_nonneg: bool = False,
    gaussian_sigma: float = 0.0,
):
    """
    Layout:
      rows = 24 planes
      cols = 6 = [Den-X, Den-Y, Eng-X, Eng-Y, Tim-X, Tim-Y]

    For a (H, W) map:
      - X projection (len W): sum over Y => axis=0
      - Y projection (len H): sum over X => axis=1
    """
    gt = gt_all.detach().cpu().float().numpy()     # (24,3,H,W)
    pr = pred_all.detach().cpu().float().numpy()   # (24,3,H,W)
    P, C, H, W = gt.shape
    assert P == 24 and C == 3

    col_titles = ["Den-X", "Den-Y", "Eng-X", "Eng-Y", "Tim-X", "Tim-Y"]

    fig_h = max(2, P * 1.6)
    fig_w = 18
    fig, axes = plt.subplots(P, 6, figsize=(fig_w, fig_h), sharex=False)
    if P == 1:
        axes = axes.reshape(1, -1)

    for p in range(P):
        gt_density = gt[p, 0]      # (H, W)
        gt_energy_avg = gt[p, 1]   # (H, W)
        gt_time_avg = gt[p, 2]     # (H, W)

        pr_density = pr[p, 0]
        pr_energy_avg = pr[p, 1]
        pr_time_avg = pr[p, 2]

        # ---- Reconstruct sum maps from avg maps (with safety) ----
        gt_energy_sum_map = _reconstruct_sum_map(gt_density, gt_energy_avg, clamp_density=True)
        pr_energy_sum_map = _reconstruct_sum_map(pr_density, pr_energy_avg, clamp_density=True)

        gt_time_sum_map = _reconstruct_sum_map(gt_density, gt_time_avg, clamp_density=True)
        pr_time_sum_map = _reconstruct_sum_map(pr_density, pr_time_avg, clamp_density=True)

        # ---- Clamp density before projecting (recommended after denorm) ----
        gt_density_clamped = np.clip(gt_density, 0.0, None)
        pr_density_clamped = np.clip(pr_density, 0.0, None)

        # ---- Correct projections for (H, W) ----
        # X projection: sum over Y axis (rows) -> axis=0 -> (W,)
        # Y projection: sum over X axis (cols) -> axis=1 -> (H,)
        gt_den_x = np.sum(gt_density_clamped, axis=0)
        gt_den_y = np.sum(gt_density_clamped, axis=1)
        pr_den_x = np.sum(pr_density_clamped, axis=0)
        pr_den_y = np.sum(pr_density_clamped, axis=1)

        gt_eng_x = np.sum(gt_energy_sum_map, axis=0)
        gt_eng_y = np.sum(gt_energy_sum_map, axis=1)
        pr_eng_x = np.sum(pr_energy_sum_map, axis=0)
        pr_eng_y = np.sum(pr_energy_sum_map, axis=1)

        gt_tim_x = np.sum(gt_time_sum_map, axis=0)
        gt_tim_y = np.sum(gt_time_sum_map, axis=1)
        pr_tim_x = np.sum(pr_time_sum_map, axis=0)
        pr_tim_y = np.sum(pr_time_sum_map, axis=1)

        axes[p, 0].set_ylabel(f"P{p}", fontsize=7, rotation=0, labelpad=18, va="center")

        plot_1d_projection(axes[p, 0], gt_den_x, pr_den_x, "Den-X (bin)", force_nonneg, gaussian_sigma)
        plot_1d_projection(axes[p, 1], gt_den_y, pr_den_y, "Den-Y (bin)", force_nonneg, gaussian_sigma)
        plot_1d_projection(axes[p, 2], gt_eng_x, pr_eng_x, "Eng-X (sum)", force_nonneg, gaussian_sigma)
        plot_1d_projection(axes[p, 3], gt_eng_y, pr_eng_y, "Eng-Y (sum)", force_nonneg, gaussian_sigma)
        plot_1d_projection(axes[p, 4], gt_tim_x, pr_tim_x, "Tim-X (sum)", force_nonneg, gaussian_sigma)
        plot_1d_projection(axes[p, 5], gt_tim_y, pr_tim_y, "Tim-Y (sum)", force_nonneg, gaussian_sigma)

    for j, title_text in enumerate(col_titles):
        axes[0, j].set_title(title_text, fontsize=9, fontweight="bold")

    axes[0, 5].legend(fontsize=7, loc="upper right")

    for j in range(6):
        axes[-1, j].set_xlabel("bin", fontsize=8)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If you pass .pdf, matplotlib will write PDF; if .png, it writes PNG.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./eval_ddim_allplanes_1d_xy")
    ap.add_argument(
        "--stats_path",
        type=str,
        required=True,
        help="Path to standardization_stats_train_only.pt for denormalization",
    )

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--num_samples", type=int, default=4)

    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--guidance_w", type=float, default=0.0)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--beta_1", type=float, default=1e-4)
    ap.add_argument("--beta_T", type=float, default=0.02)

    ap.add_argument("--force_nonneg", action="store_true", help="Force non-negative y-axis")
    ap.add_argument("--gaussian_sigma", type=float, default=0.0, help="Gaussian smoothing sigma (0 to disable)")

    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    mean, std = load_standardization_stats(args.stats_path)
    mean = mean.to(device)
    std = std.to(device)

    test_ds = PlaneDataset(args.data_dir, split="test", cache_size=8, prewarm_cache=False)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    pl_module = load_model_from_ckpt(args.ckpt, device)
    net = pl_module.model

    sampler = DDIMSamplerPlanes(
        net,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        eta=args.eta,
        ddim_steps=args.ddim_steps,
        w=args.guidance_w,
    ).to(device)
    sampler.eval()

    done = 0
    total_sampling_time = 0.0

    for batch in loader:
        B = batch["planes"].shape[0]
        take = min(B, args.num_samples - done)
        if take <= 0:
            break

        batch_small = {k: v[:take] for k, v in batch.items()}
        gt_all, pred_all, batch_time = ddim_generate_all_planes(net, sampler, batch_small, device)

        # Denormalize
        gt_all_denorm = denormalize_histograms(gt_all, mean, std)
        pred_all_denorm = denormalize_histograms(pred_all, mean, std)

        time_per_sample = batch_time / take
        total_sampling_time += batch_time

        for i in range(take):
            class_i = int(batch_small["class_id"][i].item())
            particle_type = get_particle_type_name(class_i)

            p_energy_val = float(batch_small["p_energy"][i].item())
            p_energy_scaled = p_energy_val * 5e7

            sin_zenith = float(batch_small["sin_zenith"][i].item())
            cos_zenith = float(batch_small["cos_zenith"][i].item())
            sin_azimuth = float(batch_small["sin_azimuth"][i].item())
            cos_azimuth = float(batch_small["cos_azimuth"][i].item())

            zenith_rad, azimuth_rad = retrieve_angles_from_sincos(
                sin_zenith, cos_zenith, sin_azimuth, cos_azimuth
            )

            title = (
                f"Sample {done+i:04d} | "
                f"E={p_energy_scaled:.6f} | "
                f"Primary Particle: {particle_type} | "
                f"zen={zenith_rad:.3f} | "
                f"azi={azimuth_rad:.3f} | "
                f"DDIM steps={args.ddim_steps} eta={args.eta} w={args.guidance_w}"
            )

            out_path = os.path.join(args.out_dir, f"sample_{done+i:04d}_class_{class_i}_1d.pdf")

            plot_allplanes_1d_xy_rgb(
                gt_all_denorm[i],
                pred_all_denorm[i],
                out_path,
                title,
                force_nonneg=args.force_nonneg,
                gaussian_sigma=args.gaussian_sigma,
            )

        done += take
        print(
            f"[eval] done {done}/{args.num_samples} | "
            f"Batch time: {batch_time:.2f}s | Time per sample: {time_per_sample:.2f}s"
        )

        if done >= args.num_samples:
            break

    avg_time_per_sample = total_sampling_time / done if done > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"Saved outputs to: {args.out_dir}")
    print(f"Total sampling time: {total_sampling_time:.2f}s")
    print(f"Average time per sample: {avg_time_per_sample:.2f}s")
    print("Data plotted in ORIGINAL (denormalized) scale")
    print("=" * 60)


if __name__ == "__main__":
    main()

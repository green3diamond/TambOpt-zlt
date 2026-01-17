#!/usr/bin/env python3
# eval_ddim_allplanes_1d_xy_rgb.py
#
# Autoregressively generate ALL 24 planes with DDIM, then for each sample
# save ONE figure per event with ONLY 1D projections:
#   - 24 rows (planes 0..23)
#   - 6 columns: [C0-X, C1-X, C2-X, C0-Y, C1-Y, C2-Y]
# Each subplot overlays GT vs Pred.

import os
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lightning_training import PlaneDataset, PlaneDiffusionModule
from diffusion import DDIMSamplerPlanes

# python /n/home04/hhanif/tam/unet/lightining_eval.py     --data_dir /n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulation_nov_25/pre_processed_3rd_step/        --ckpt /n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulation_nov_25/checkpoints/tam_unet/epoch_epoch\=689-val_loss_val_loss\=0.0344.ckpt      --out_dir /n/home04/hhanif/tam/plots/lighting_eval_plots_w_1.8      --num_samples 10     --ddim_steps 50     --eta 0.0     --guidance_w 1.8

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
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    return gt_all, pred_all


def plot_allplanes_1d_xy_rgb(
    gt_all: torch.Tensor,
    pred_all: torch.Tensor,
    out_png: str,
    title: str,
):
    """
    Layout:
      rows = 24 planes
      cols = 6 = [C0-X, C1-X, C2-X, C0-Y, C1-Y, C2-Y]
    X projection: sum over Y -> f(X) (length W)
    Y projection: sum over X -> f(Y) (length H)
    """
    gt = gt_all.detach().cpu().float().numpy()     # (24,3,H,W)
    pr = pred_all.detach().cpu().float().numpy()   # (24,3,H,W)
    P, C, H, W = gt.shape
    assert P == 24 and C == 3

    col_titles = ["C0-X", "C1-X", "C2-X", "C0-Y", "C1-Y", "C2-Y"]

    # size tuning: 24 rows -> tall figure; adjust if you want more compact
    fig_h = 48
    fig_w = 18
    fig, axes = plt.subplots(P, 6, figsize=(fig_w, fig_h), sharex=False)

    fig.suptitle(title, fontsize=10)

    for p in range(P):
        for c in range(3):
            # X projection
            axx = axes[p, c]
            gt_x = gt[p, c].sum(axis=0)   # (W,)
            pr_x = pr[p, c].sum(axis=0)
            axx.plot(gt_x, linewidth=1.0, label="GT")
            axx.plot(pr_x, linewidth=1.0, linestyle="--", label="Pred")
            axx.tick_params(labelsize=6)

            # Y projection
            axy = axes[p, c + 3]
            gt_y = gt[p, c].sum(axis=1)   # (H,)
            pr_y = pr[p, c].sum(axis=1)
            axy.plot(gt_y, linewidth=1.0, label="GT")
            axy.plot(pr_y, linewidth=1.0, linestyle="--", label="Pred")
            axy.tick_params(labelsize=6)

        # plane label on far-left subplot
        axes[p, 0].set_ylabel(f"p{p:02d}", fontsize=7, rotation=0, labelpad=18, va="center")

    # titles + single legend
    for j in range(6):
        axes[0, j].set_title(col_titles[j], fontsize=8)
    axes[0, 5].legend(fontsize=7, loc="upper right")

    # bottom xlabels
    for j in range(3):
        axes[-1, j].set_xlabel("X bin (sum over Y)", fontsize=8)
    for j in range(3, 6):
        axes[-1, j].set_xlabel("Y bin (sum over X)", fontsize=8)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./eval_ddim_allplanes_1d_xy")

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--num_samples", type=int, default=4)

    # DDIM
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--guidance_w", type=float, default=0.0)

    # Diffusion schedule (match training)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--beta_1", type=float, default=1e-4)
    ap.add_argument("--beta_T", type=float, default=0.02)

    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

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
    for batch in loader:
        B = batch["planes"].shape[0]
        take = min(B, args.num_samples - done)
        if take <= 0:
            break

        batch_small = {k: v[:take] for k, v in batch.items()}
        gt_all, pred_all = ddim_generate_all_planes(net, sampler, batch_small, device)

        for i in range(take):
            class_i = int(batch_small["class_id"][i].item())
            title = f"sample={done+i:04d} class_id={class_i}  DDIM steps={args.ddim_steps} eta={args.eta} w={args.guidance_w}"
            out_png = os.path.join(args.out_dir, f"sample_{done+i:04d}_allplanes_1d_xy.pdf")
            plot_allplanes_1d_xy_rgb(gt_all[i], pred_all[i], out_png, title)

        done += take
        print(f"[eval] done {done}/{args.num_samples}")
        if done >= args.num_samples:
            break

    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
import time
import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class PlaneDiffusionEvaluator:
    """
    A class for evaluating plane diffusion models with DDIM sampling.

    Example usage:
        evaluator = PlaneDiffusionEvaluator(
            data_dir="/path/to/data",
            checkpoint_path="/path/to/ckpt.ckpt",
            output_dir="./eval_output",
            device="cuda:0"
        )
        evaluator.run_full_pipeline(num_samples=10, ddim_steps=50)
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_path: str,
        output_dir: str,
        device: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 2,
        ddim_steps: int = 50,
        eta: float = 0.0,
        guidance_w: float = 0.0,
        T: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        seed: int = 24,
        imports_path: Optional[str] = None
    ):
        """
        Initialize the PlaneDiffusionEvaluator.

        Args:
            data_dir: Directory containing test data
            checkpoint_path: Path to the trained model checkpoint
            output_dir: Directory to save evaluation plots
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None
            batch_size: Batch size for data loading
            num_workers: Number of data loading workers
            ddim_steps: Number of DDIM sampling steps
            eta: DDIM eta parameter (0.0 = deterministic)
            guidance_w: Guidance weight for classifier-free guidance
            T: Total diffusion timesteps
            beta_1: Starting beta value
            beta_T: Ending beta value
            seed: Random seed for reproducibility
            imports_path: Path to add to sys.path for imports
        """
        # Add path if provided
        if imports_path:
            sys.path.append(imports_path)

        # Import dependencies (after path is set)
        from lightning_training import PlaneDataset, PlaneDiffusionModule
        from diffusion import DDIMSamplerPlanes

        self.PlaneDataset = PlaneDataset
        self.PlaneDiffusionModule = PlaneDiffusionModule
        self.DDIMSamplerPlanes = DDIMSamplerPlanes

        # Store parameters
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.guidance_w = guidance_w
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.seed = seed

        # Initialize containers
        self.pl_module = None
        self.net = None
        self.sampler = None
        self.test_dataset = None
        self.test_loader = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seeds
        self._seed_all(self.seed)

        print(f"Initialized PlaneDiffusionEvaluator")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def _seed_all(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self):
        """Load the trained model checkpoint and create DDIM sampler."""
        start_time = time.time()

        # Load checkpoint
        self.pl_module = self.PlaneDiffusionModule.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        self.pl_module.eval()
        self.pl_module.to(self.device)

        print(f"Loaded checkpoint: {self.checkpoint_path}")

        # Extract model
        self.net = self.pl_module.model

        # Build DDIM Sampler
        self.sampler = self.DDIMSamplerPlanes(
            self.net,
            beta_1=self.beta_1,
            beta_T=self.beta_T,
            T=self.T,
            eta=self.eta,
            ddim_steps=self.ddim_steps,
            w=self.guidance_w,
        ).to(self.device)
        self.sampler.eval()

        print(f"DDIM sampler built (ddim_steps={self.ddim_steps}, eta={self.eta}, w={self.guidance_w})")
        print(f"Model loading time: {time.time() - start_time:.2f}s")

    def setup_data(self):
        """Setup test dataset and dataloader."""
        self.test_dataset = self.PlaneDataset(
            self.data_dir,
            split="test",
            cache_size=8,
            prewarm_cache=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"Test dataloader ready. Number of batches: {len(self.test_loader)}")

    @torch.no_grad()
    def generate_all_planes(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively generate all 24 planes with DDIM.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (ground_truth, predictions) tensors
        """
        gt_all = batch["planes"].to(self.device)  # (B,24,3,H,W)
        B, P, C, H, W = gt_all.shape
        assert P == 24 and C == 3, f"Expected (B,24,3,H,W), got {gt_all.shape}"

        p_energy = batch["p_energy"].to(self.device)
        class_id = batch["class_id"].to(self.device)
        sin_zenith = batch["sin_zenith"].to(self.device)
        cos_zenith = batch["cos_zenith"].to(self.device)
        sin_azimuth = batch["sin_azimuth"].to(self.device)
        cos_azimuth = batch["cos_azimuth"].to(self.device)

        pred_all = torch.zeros_like(gt_all)
        past = torch.zeros((B, 3, H, W), device=self.device)

        for plane_idx in range(P):
            plane_idx_tensor = torch.full((B,), plane_idx, device=self.device, dtype=torch.long)
            x = torch.randn((B, 3, H, W), device=self.device)

            pred = self.sampler(
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
        self,
        gt_all: torch.Tensor,
        pred_all: torch.Tensor,
        out_png: str,
        title: str,
    ):
        """
        Create 1D projection plots for all planes.

        Layout:
          rows = 24 planes
          cols = 6 = [C0-X, C1-X, C2-X, C0-Y, C1-Y, C2-Y]

        Args:
            gt_all: Ground truth tensor (24,3,H,W)
            pred_all: Prediction tensor (24,3,H,W)
            out_png: Output file path
            title: Plot title
        """
        gt = gt_all.detach().cpu().float().numpy()
        pr = pred_all.detach().cpu().float().numpy()
        P, C, H, W = gt.shape
        assert P == 24 and C == 3

        col_titles = ["C0-X", "C1-X", "C2-X", "C0-Y", "C1-Y", "C2-Y"]

        fig_h = 48
        fig_w = 18
        fig, axes = plt.subplots(P, 6, figsize=(fig_w, fig_h), sharex=False)

        fig.suptitle(title, fontsize=10)

        for p in range(P):
            for c in range(3):
                # X projection
                axx = axes[p, c]
                gt_x = gt[p, c].sum(axis=0)
                pr_x = pr[p, c].sum(axis=0)
                axx.plot(gt_x, linewidth=1.0, label="GT")
                axx.plot(pr_x, linewidth=1.0, linestyle="--", label="Pred")
                axx.tick_params(labelsize=6)

                # Y projection
                axy = axes[p, c + 3]
                gt_y = gt[p, c].sum(axis=1)
                pr_y = pr[p, c].sum(axis=1)
                axy.plot(gt_y, linewidth=1.0, label="GT")
                axy.plot(pr_y, linewidth=1.0, linestyle="--", label="Pred")
                axy.tick_params(labelsize=6)

            axes[p, 0].set_ylabel(f"p{p:02d}", fontsize=7, rotation=0, labelpad=18, va="center")

        for j in range(6):
            axes[0, j].set_title(col_titles[j], fontsize=8)
        axes[0, 5].legend(fontsize=7, loc="upper right")

        for j in range(3):
            axes[-1, j].set_xlabel("X bin (sum over Y)", fontsize=8)
        for j in range(3, 6):
            axes[-1, j].set_xlabel("Y bin (sum over X)", fontsize=8)

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        print(f"Saved plot: {out_png}")

    def evaluate_samples(self, num_samples: int = 4):
        """
        Evaluate and plot samples from the test set.

        Args:
            num_samples: Number of samples to evaluate
        """
        if self.sampler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.test_loader is None:
            raise RuntimeError("Data not setup. Call setup_data() first.")

        done = 0
        start_time = time.time()

        for batch in self.test_loader:
            B = batch["planes"].shape[0]
            take = min(B, num_samples - done)
            if take <= 0:
                break

            batch_small = {k: v[:take] for k, v in batch.items()}
            gt_all, pred_all = self.generate_all_planes(batch_small)

            for i in range(take):
                class_i = int(batch_small["class_id"][i].item())
                title = (f"sample={done+i:04d} class_id={class_i}  "
                        f"DDIM steps={self.ddim_steps} eta={self.eta} w={self.guidance_w}")
                out_png = os.path.join(self.output_dir, f"sample_{done+i:04d}_allplanes_1d_xy.pdf")
                self.plot_allplanes_1d_xy_rgb(gt_all[i], pred_all[i], out_png, title)

            done += take
            print(f"[eval] done {done}/{num_samples}")
            if done >= num_samples:
                break

        print(f"Total evaluation time: {time.time() - start_time:.2f}s")

    def run_full_pipeline(self, num_samples: int = 4):
        """
        Run the complete evaluation pipeline.

        Args:
            num_samples: Number of samples to evaluate
        """
        print("=" * 60)
        print("Starting full evaluation pipeline")
        print("=" * 60)

        self.load_model()
        self.setup_data()
        self.evaluate_samples(num_samples=num_samples)

        print("=" * 60)
        print("Pipeline complete!")
        print(f"Saved outputs to: {self.output_dir}")
        print("=" * 60)

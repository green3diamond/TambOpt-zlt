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
        
        self.test_images = []
        self.test_conditions = []
        self.generated_sets = []

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


    def extract_test_samples(self, num_conditions: int = 10):
        """
        Extract test samples and their conditions.

        Args:
            num_conditions: Number of test conditions to extract
        """
        self.test_images = []
        self.test_conditions = []
        count = 0

        for batch in self.test_loader:
            planes = batch["planes"]
            p_energy = batch["p_energy"]
            class_id = batch["class_id"]
            sin_zenith = batch["sin_zenith"]
            cos_zenith = batch["cos_zenith"]
            sin_azimuth = batch["sin_azimuth"]
            cos_azimuth = batch["cos_azimuth"]
            
            B = planes.size(0)

            for i in range(B):
                # Store planes (24,3,H,W)
                self.test_images.append(planes[i].clone())

                # Store condition (7,) - includes class_id
                self.test_conditions.append(torch.stack([
                    p_energy[i],
                    class_id[i],
                    sin_zenith[i],
                    cos_zenith[i],
                    sin_azimuth[i],
                    cos_azimuth[i],
                ]))

                count += 1
                if count == num_conditions:
                    break

            if count == num_conditions:
                break

        print(f"Collected {len(self.test_conditions)} conditioning vectors.")
        if len(self.test_conditions) > 0:
            print(f"First conditioning vector (cpu): {self.test_conditions[0]}")


    def generate_samples(
        self, 
        num_samples: int = 1000, 
        num_conditions: Optional[int] = None,
        chunk_size: int = 200
    ):
        """
        Generate samples for each test condition using autoregressive plane generation.

        Args:
            num_samples: Number of samples to generate per condition
            num_conditions: Number of conditions to use (None = use all extracted)
            chunk_size: Batch size for generation (to avoid OOM)
        """
        if self.sampler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(self.test_conditions) == 0:
            raise RuntimeError("No test samples extracted. Call extract_test_samples() first.")

        # Determine how many conditions to process
        conditions_to_process = self.test_conditions
        if num_conditions is not None:
            conditions_to_process = self.test_conditions[:num_conditions]

        self.generated_sets = []
        start_time = time.time()

        for idx, cond_vec in enumerate(conditions_to_process):
            cond_vec = cond_vec.to(self.device)  # (6,)

            # Get the ground truth planes shape from test_images
            gt_planes = self.test_images[idx]  # (24,3,H,W)
            P, C, H, W = gt_planes.shape

            all_samples = []
            samples_done = 0

            while samples_done < num_samples:
                bs = min(chunk_size, num_samples - samples_done)

                # Prepare condition batch
                p_energy = cond_vec[0].unsqueeze(0).expand(bs)
                class_id = cond_vec[1].unsqueeze(0).expand(bs).long()
                sin_zenith = cond_vec[2].unsqueeze(0).expand(bs)
                cos_zenith = cond_vec[3].unsqueeze(0).expand(bs)
                sin_azimuth = cond_vec[4].unsqueeze(0).expand(bs)
                cos_azimuth = cond_vec[5].unsqueeze(0).expand(bs)

                # Generate all 24 planes autoregressively
                pred_all = torch.zeros((bs, P, C, H, W), device=self.device)
                past = torch.zeros((bs, C, H, W), device=self.device)

                with torch.no_grad():
                    for plane_idx in range(P):
                        plane_idx_tensor = torch.full((bs,), plane_idx, device=self.device, dtype=torch.long)
                        noise = torch.randn((bs, C, H, W), device=self.device)

                        pred = self.sampler(
                            noise,
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

                all_samples.append(pred_all.cpu())

                # Free GPU memory
                del noise, pred, pred_all, past
                torch.cuda.empty_cache()

                samples_done += bs

            # Concatenate all chunks -> (num_samples, 24, 3, H, W)
            gen_imgs_all = torch.cat(all_samples, dim=0)
            self.generated_sets.append({
                "condition": cond_vec.cpu(),
                "images": gen_imgs_all,
            })

            # Extra safety between conditions
            torch.cuda.empty_cache()

        total_images = sum([s["images"].shape[0] for s in self.generated_sets])
        print(f"✔ Done: generated {total_images} images across {len(self.generated_sets)} conditions.")
        print(f"Total generation time: {time.time() - start_time:.2f}s")
        return self.generated_sets


    def plot_xy_profiles(self, num_conditions: Optional[int] = None, save_dir: Optional[str] = None):
        """
        Plot average X and Y profiles across all generated samples.

        Args:
            num_conditions: Number of conditions to plot (None = all)
            save_dir: Directory to save plots (None = display only)
        """
        if len(self.generated_sets) == 0:
            print("No generated sets to plot.")
            return

        condition_names = ['energy', 'class_id', 'sin_zenith', 'cos_zenith', 'sin_azimuth', 'cos_azimuth']
        num_to_plot = num_conditions if num_conditions else len(self.generated_sets)
        num_to_plot = min(num_to_plot, len(self.generated_sets))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for idx in range(num_to_plot):
            output = self.generated_sets[idx]
            
            # Print conditions
            print(f"\nCondition {idx + 1}:")
            for icondition, condition_val in enumerate(output['condition']):
                print(f"  {condition_names[icondition]}: {condition_val.item():.4f}")

            images = output['images']  # Shape: (N, 24, 3, H, W)
            images_np = images.cpu().numpy()

            # Average over samples (N) and channels (3) -> (24, H, W)
            avg_profiles = images_np.mean(axis=(0, 2))  # Shape: (24, H, W)

            # X profile: average over Y (axis=1) -> (24, W)
            avg_x_profile = avg_profiles.mean(axis=1)
            
            # Y profile: average over X (axis=2) -> (24, H)
            avg_y_profile = avg_profiles.mean(axis=2)

            # Plot
            fig, axes = plt.subplots(4, 6, figsize=(24, 16))
            fig.suptitle(f"Condition {idx + 1}: X and Y Profiles", fontsize=14)

            for plane_idx in range(24):
                ax = axes[plane_idx // 6, plane_idx % 6]
                
                ax.plot(avg_x_profile[plane_idx], label='X Profile', linewidth=1.5)
                ax.plot(avg_y_profile[plane_idx], label='Y Profile', linewidth=1.5, linestyle='--')
                
                ax.set_title(f"Plane {plane_idx}", fontsize=9)
                ax.set_xlabel('Position', fontsize=8)
                ax.set_ylabel('Average Intensity', fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            if save_dir:
                save_path = os.path.join(save_dir, f"xy_profiles_condition_{idx + 1}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()
            else:
                plt.show()
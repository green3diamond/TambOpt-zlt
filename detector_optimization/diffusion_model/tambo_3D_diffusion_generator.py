
import sys
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple


class TamboDiffusionGenerator:
    """
    A class for generating samples from a trained Tambo diffusion model.

    Example usage:
        generator = TamboDiffusionGenerator(
            checkpoint_path="/path/to/ckpt.ckpt",
            output_dir="diffusion_model/run_2",
            device="cuda:0"
        )
        generator.load_model()
        generator.setup_data()
        generator.generate_samples(num_samples=1000, num_conditions=10, chunk_size=200)
        generator.save_results()
        generator.plot_results()
    """

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        device: Optional[str] = None,
        ddim_steps: int = 100,
        ddim_eta: float = 0.0,
        batch_size: int = 64,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        num_workers: int = 4,
        seed: int = 42,
        tambo_optimization_path: Optional[str] = None
    ):
        """
        Initialize the TamboDiffusionGenerator.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            output_dir: Directory to save generated samples and plots
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None
            ddim_steps: Number of DDIM sampling steps
            ddim_eta: DDIM eta parameter (0.0 = deterministic)
            batch_size: Batch size for data loading
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            num_workers: Number of data loading workers
            seed: Random seed for reproducibility
            tambo_optimization_path: Path to add to sys.path for imports
        """
        # Add path if provided
        if tambo_optimization_path:
            sys.path.append(tambo_optimization_path)

        # Import dependencies (after path is set)
        from diffusion_train import (
            TamboDataModule, 
            TamboDiffusionLitModel, 
            df_cleaned
        )
        from models.DiffusionCondition import DDIMSampler

        self.TamboDataModule = TamboDataModule
        self.TamboDiffusionLitModel = TamboDiffusionLitModel
        self.df_cleaned = df_cleaned
        self.DDIMSampler = DDIMSampler

        # Store parameters
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.seed = seed

        # Initialize containers
        self.lit_model = None
        self.model = None
        self.sampler = None
        self.data_module = None
        self.test_loader = None
        self.test_images = []
        self.test_conditions = []
        self.generated_sets = []

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Initialized TamboDiffusionGenerator")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def load_model(self):
        """Load the trained model checkpoint and create DDIM sampler."""
        start_time = time.time()

        # Load checkpoint
        self.lit_model = self.TamboDiffusionLitModel.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device,
        ).to(self.device)
        self.lit_model.eval()

        print(f"Loaded checkpoint: {self.checkpoint_path}")

        # Extract model
        self.model = self.lit_model.model
        self.model.eval()
        print("Model set to eval mode.")

        # Build DDIM Sampler
        self.sampler = self.DDIMSampler(
            self.model,
            beta_1=self.lit_model.hparams.beta_1,
            beta_T=self.lit_model.hparams.beta_T,
            T=self.lit_model.hparams.T,
            eta=self.ddim_eta,
            ddim_steps=self.ddim_steps,
        )

        print(f"DDIM sampler built (ddim_steps={self.ddim_steps}, eta={self.ddim_eta})")
        print(f"Model loading time: {time.time() - start_time:.2f}s")

    def setup_data(self):
        """Setup data module and test dataloader."""
        self.data_module = self.TamboDataModule(
            df=self.df_cleaned,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            num_workers=self.num_workers,
            seed=self.seed,
        )
        self.data_module.setup("test")

        self.test_loader = self.data_module.test_dataloader()
        print(f"Test dataloader ready. Number of batches: {len(self.test_loader)}")

    def extract_test_samples(self, num_conditions: int = 10):
        """
        Extract test samples and their conditions.

        Args:
            num_conditions: Number of test conditions to extract
        """
        self.test_images = []
        self.test_conditions = []
        count = 0

        for imgs, cond in self.test_loader:
            energy, sin_z, cos_z, sin_a, cos_a = cond
            B = energy.size(0)

            for i in range(B):
                # Store image (3,32,32)
                self.test_images.append(imgs[i].clone())

                # Store condition (5,)
                self.test_conditions.append(torch.stack([
                    energy[i], sin_z[i], cos_z[i], sin_a[i], cos_a[i]
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
        Generate samples for each test condition.

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
            # print(f"Starting generation for condition {idx+1}/{len(conditions_to_process)}")
            # print(f"Condition (cpu): {cond_vec}")
            cond_vec = cond_vec.to(self.device)  # (5,)

            images_chunks = []
            samples_done = 0

            while samples_done < num_samples:
                bs = min(chunk_size, num_samples - samples_done)

                # noise: (bs, 3, 32, 32)
                noise = torch.randn(bs, 3, 32, 32, device=self.device)

                # (bs, 5) condition batch
                cond_batch = cond_vec.view(1, 5).expand(bs, 5)

                with torch.no_grad():
                    gen_imgs_chunk = self.sampler(
                        noise,
                        cond_batch[:, 0],  # energy
                        cond_batch[:, 1],  # sin z
                        cond_batch[:, 2],  # cos z
                        cond_batch[:, 3],  # sin a
                        cond_batch[:, 4],  # cos a
                    )  # (bs, 3, 32, 32)

                images_chunks.append(gen_imgs_chunk.cpu())

                # Free GPU memory for this chunk
                del noise, cond_batch, gen_imgs_chunk
                torch.cuda.empty_cache()

                samples_done += bs
                # print(f"Condition {idx+1}/{len(conditions_to_process)}: {samples_done}/{num_samples} samples done")

            # Concatenate all chunks -> (num_samples, 3, 32, 32)
            gen_imgs_all = torch.cat(images_chunks, dim=0)
            self.generated_sets.append({
                "condition": cond_vec.cpu(),
                "images": gen_imgs_all,
            })
            # print(f"Concatenated generated images for condition {idx+1}: {gen_imgs_all.shape}")

            # Extra safety between conditions
            torch.cuda.empty_cache()
            # print(f"Finished condition {idx+1}/{len(conditions_to_process)}")

        total_images = sum([s["images"].shape[0] for s in self.generated_sets])
        # print(f"✔ Done: generated {total_images} images across {len(self.generated_sets)} conditions.")
        # print(f"Total generation time: {time.time() - start_time:.2f}s")
        return self.generated_sets

    def save_results(self):
        """Save generated images and conditions as numpy bundles."""
        if len(self.generated_sets) == 0:
            print("No generated sets to save.")
            return

        # Save per-condition bundles
        for i, s in enumerate(self.generated_sets):
            cond = s["condition"].cpu().numpy()
            imgs = s["images"].numpy()

            # Ground-truth image if available
            gt = None
            if i < len(self.test_images):
                gt = self.test_images[i].cpu().numpy()

            meta = {
                "condition_index": i + 1,
                "num_generated": int(imgs.shape[0]),
            }

            bundle = {
                "input": cond,    # condition vector (5,)
                "target": gt,     # ground-truth image (3,H,W) or None
                "output": imgs,   # generated images (N,3,H,W)
                "meta": meta,
            }

            out_path = os.path.join(self.output_dir, f"condition_{i+1}.npz")
            np.savez_compressed(out_path, bundle=bundle)
            print(f"Saved numpy bundle: {out_path} (images shape: {imgs.shape})")

        # Save summary file
        all_conds = np.stack([s["condition"].cpu().numpy() for s in self.generated_sets])
        total_images = sum([s["images"].shape[0] for s in self.generated_sets])

        summary = {
            "all_conditions": all_conds,
            "total_images": int(total_images),
            "num_conditions": len(self.generated_sets),
        }
        summary_path = os.path.join(self.output_dir, "summary.npz")
        np.savez_compressed(summary_path, summary=summary)
        print(f"Saved summary numpy bundle: {summary_path} (conditions: {all_conds.shape})")

    def plot_results(self, num_conditions: Optional[int] = None, dpi: int = 300):
        """
        Create comparison plots for ground truth vs generated samples.

        Args:
            num_conditions: Number of conditions to plot (None = all)
            dpi: DPI for saved plots
        """
        if len(self.generated_sets) == 0:
            print("No generated sets to plot.")
            return

        num_to_plot = num_conditions if num_conditions else len(self.generated_sets)
        num_to_plot = min(num_to_plot, len(self.generated_sets))

        for idx in range(num_to_plot):
            # Ground truth image
            gt_img = self.test_images[idx].cpu().numpy()   # (3, H, W)

            gt_R = gt_img[0]
            gt_G = gt_img[1]
            gt_B = gt_img[2]

            gt_R_x, gt_R_y = gt_R.mean(axis=0), gt_R.mean(axis=1)
            gt_G_x, gt_G_y = gt_G.mean(axis=0), gt_G.mean(axis=1)
            gt_B_x, gt_B_y = gt_B.mean(axis=0), gt_B.mean(axis=1)

            # Generated images
            gen_imgs = self.generated_sets[idx]["images"].cpu().numpy()  # (N, 3, H, W)

            R_batch = gen_imgs[:, 0]
            G_batch = gen_imgs[:, 1]
            B_batch = gen_imgs[:, 2]

            R_x_mean = R_batch.mean(axis=1).mean(axis=0)
            R_y_mean = R_batch.mean(axis=2).mean(axis=0)

            G_x_mean = G_batch.mean(axis=1).mean(axis=0)
            G_y_mean = G_batch.mean(axis=2).mean(axis=0)

            B_x_mean = B_batch.mean(axis=1).mean(axis=0)
            B_y_mean = B_batch.mean(axis=2).mean(axis=0)

            # Condition values
            cond_vec = self.generated_sets[idx]["condition"].cpu().numpy()  # (5,)
            energy, sin_z, cos_z, sin_a, cos_a = cond_vec.tolist()

            # Plot
            fig, axs = plt.subplots(1, 6, figsize=(24, 4))
            fig.suptitle(
                f"Condition {idx+1}: "
                f"E={energy:.3f}, sin_z={sin_z:.3f}, cos_z={cos_z:.3f}, "
                f"sin_a={sin_a:.3f}, cos_a={cos_a:.3f}",
                y=1.05,
            )

            axs[0].plot(gt_R_x, label="GT")
            axs[0].plot(R_x_mean, label="Gen Mean", linestyle="--")
            axs[0].set_title("Particle Density vs x")

            axs[1].plot(gt_R_y, label="GT")
            axs[1].plot(R_y_mean, label="Gen Mean", linestyle="--")
            axs[1].set_title("Particle Density vs y")

            axs[2].plot(gt_G_x, label="GT")
            axs[2].plot(G_x_mean, label="Gen Mean", linestyle="--")
            axs[2].set_title("Average arrival time vs x")

            axs[3].plot(gt_G_y, label="GT")
            axs[3].plot(G_y_mean, label="Gen Mean", linestyle="--")
            axs[3].set_title("Average arrival time vs y")

            axs[4].plot(gt_B_x, label="GT")
            axs[4].plot(B_x_mean, label="Gen Mean", linestyle="--")
            axs[4].set_title("Average arrival time vs x")

            axs[5].plot(gt_B_y, label="GT")
            axs[5].plot(B_y_mean, label="Gen Mean", linestyle="--")
            axs[5].set_title("Average Kinetic Energy vs y")

            axs[0].legend(loc="upper right")

            plt.tight_layout()
            out_name = os.path.join(self.output_dir, f"condition_{idx+1}.png")
            print(f"Saving plot: {out_name}")
            plt.savefig(out_name, dpi=dpi, bbox_inches="tight")
            plt.close()

    def run_full_pipeline(
        self, 
        num_samples: int = 1000, 
        num_conditions: int = 10,
        chunk_size: int = 200,
        plot_dpi: int = 300
    ):
        """
        Run the complete generation pipeline.

        Args:
            num_samples: Number of samples to generate per condition
            num_conditions: Number of test conditions to process
            chunk_size: Batch size for generation
            plot_dpi: DPI for saved plots
        """
        print("=" * 60)
        print("Starting full generation pipeline")
        print("=" * 60)

        self.load_model()
        self.setup_data()
        self.extract_test_samples(num_conditions=num_conditions)
        self.generate_samples(num_samples=num_samples, chunk_size=chunk_size)
        self.save_results()
        self.plot_results(dpi=plot_dpi)

        print("=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

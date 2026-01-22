# train_planes_single.py - Single Head (Diffusion Only) - STANDARDIZED VERSION
import os
from typing import Dict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

from diffusion import (
    GaussianDiffusionTrainer, 
    GaussianDiffusionSampler,
    DDIMSamplerPlanes,
    AutoregressivePlaneGenerator
)
from unet import UNetPlanes, count_parameters

print("Number of CPU cores available:", os.cpu_count())
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


class PlaneDataset(Dataset):
    """
    Dataset for loading preprocessed plane data from step3_preprocessing.py
    
    Expected data format:
    - histograms: (N, 24, 3, H, W) - STANDARDIZED (zero mean, unit variance)
    - p_energy: (N,) - normalized primary energy
    - sin_zenith, cos_zenith: (N,) - zenith angle
    - sin_azimuth, cos_azimuth: (N,) - azimuth angle
    - class_id: (N,) - integer class labels (0, 1, 2, ...)
    """
    def __init__(self, chunk_dir, split='train'):
        """
        Args:
            chunk_dir: path to directory containing train/val/test subdirectories
            split: 'train', 'val', or 'test'
        """
        self.split = split
        self.chunk_dir = os.path.join(chunk_dir, split)
        
        # Read chunk index
        index_file = os.path.join(self.chunk_dir, 'index.txt')
        with open(index_file, 'r') as f:
            self.chunk_files = [line.strip() for line in f]
        
        # Load all chunks
        print(f"Loading {split} data from {len(self.chunk_files)} chunks...")
        self.load_all_chunks()
        
        print(f"Loaded {len(self)} samples from {split} split")
        print(f"Histograms shape: {self.histograms.shape}")
        print(f"Data range: [{self.histograms.min():.3f}, {self.histograms.max():.3f}]")
        print(f"Data mean: {self.histograms.mean():.3f}, std: {self.histograms.std():.3f}")

    def load_all_chunks(self):
        """Load all chunks into memory"""
        all_hists = []
        all_p_energy = []
        all_sin_zenith = []
        all_cos_zenith = []
        all_sin_azimuth = []
        all_cos_azimuth = []
        all_class_id = []
        
        for chunk_file in tqdm(self.chunk_files, desc=f"Loading {self.split} chunks"):
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            data = torch.load(chunk_path, map_location='cpu')
            
            all_hists.append(data['histograms'])
            all_p_energy.append(data['p_energy'])
            all_sin_zenith.append(data['sin_zenith'])
            all_cos_zenith.append(data['cos_zenith'])
            all_sin_azimuth.append(data['sin_azimuth'])
            all_cos_azimuth.append(data['cos_azimuth'])
            all_class_id.append(data['class_id'])
        
        self.histograms = torch.cat(all_hists, dim=0)  # (N, 24, 3, H, W) - STANDARDIZED
        self.p_energy = torch.cat(all_p_energy, dim=0)
        self.sin_zenith = torch.cat(all_sin_zenith, dim=0)
        self.cos_zenith = torch.cat(all_cos_zenith, dim=0)
        self.sin_azimuth = torch.cat(all_sin_azimuth, dim=0)
        self.cos_azimuth = torch.cat(all_cos_azimuth, dim=0)
        self.class_id = torch.cat(all_class_id, dim=0)

    def __len__(self):
        return len(self.histograms)

    def __getitem__(self, idx):
        """Returns data for training autoregressive plane model"""
        planes = self.histograms[idx]  # (24, 3, H, W)
        
        return {
            'planes': planes,
            'p_energy': self.p_energy[idx],
            'sin_zenith': self.sin_zenith[idx],
            'cos_zenith': self.cos_zenith[idx],
            'sin_azimuth': self.sin_azimuth[idx],
            'cos_azimuth': self.cos_azimuth[idx],
            'class_id': self.class_id[idx] + 1
        }


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def load_standardization_stats(data_dir):
    """Load standardization statistics from preprocessing"""
    stats_path = os.path.join(data_dir, "..", "standardization_stats_train_only.pt")
    if not os.path.exists(stats_path):
        # Try alternative path
        stats_path = os.path.join(data_dir, "standardization_stats_train_only.pt")
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Standardization stats not found. Expected at: {stats_path}\n"
            "Make sure you ran step3_preprocessing.py first."
        )
    
    stats = torch.load(stats_path, map_location='cpu')
    print(f"\nLoaded standardization stats from: {stats_path}")
    print(f"  Mean shape: {stats['mean'].shape}")
    print(f"  Std shape: {stats['std'].shape}")
    
    return stats['mean'], stats['std']


def denormalize_standardized(standardized, mean, std):
    """
    Convert from standardized (zero mean, unit variance) back to original scale
    
    Args:
        standardized: (B, 24, 3, H, W) or (24, 3, H, W) - standardized data
        mean: (24, 3) - per-plane, per-channel means
        std: (24, 3) - per-plane, per-channel standard deviations
    
    Returns:
        denormalized: same shape as input, in original scale
    """
    # Add batch dimension handling
    if standardized.dim() == 4:
        # (24, 3, H, W)
        return standardized * std[:, :, None, None] + mean[:, :, None, None]
    elif standardized.dim() == 5:
        # (B, 24, 3, H, W)
        return standardized * std[None, :, :, None, None] + mean[None, :, :, None, None]
    else:
        raise ValueError(f"Unexpected dimension: {standardized.dim()}")


def normalize_for_visualization(data, percentile_clip=99.5):
    """
    Normalize unbounded data to [0, 1] for visualization
    
    Args:
        data: tensor of any shape
        percentile_clip: clip extreme values at this percentile (default 99.5%)
    
    Returns:
        normalized: same shape, values in [0, 1]
    """
    # Clip extreme outliers using percentiles
    data_flat = data.flatten()
    lower = torch.quantile(data_flat, (100 - percentile_clip) / 100)
    upper = torch.quantile(data_flat, percentile_clip / 100)
    
    data_clipped = torch.clamp(data, lower, upper)
    
    # Normalize to [0, 1]
    data_min = data_clipped.amin()
    data_max = data_clipped.amax()
    
    if data_max - data_min < 1e-8:
        return torch.zeros_like(data)
    
    return (data_clipped - data_min) / (data_max - data_min)


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    # Load dataset
    dataset = PlaneDataset(
        chunk_dir=modelConfig["data_dir"],
        split='train'
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=modelConfig["batch_size"], 
        shuffle=True,
        num_workers=modelConfig.get("num_workers", 4),
        drop_last=True,
        pin_memory=True
    )
    
    val_dataset = PlaneDataset(
        chunk_dir=modelConfig["data_dir"],
        split='val'
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=False,
        num_workers=modelConfig.get("num_workers", 4),
        drop_last=False,
        pin_memory=True
    )
    
    # Get number of unique classes
    num_classes = len(torch.unique(dataset.class_id))
    unique_classes = torch.unique(dataset.class_id).tolist()
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class IDs: {unique_classes}")
    
    # Verify CFG null class doesn't overlap with real classes
    if modelConfig.get("use_cfg", False):
        cfg_null_class = modelConfig.get("cfg_null_class", 0)
        if cfg_null_class in unique_classes:
            print(f"WARNING: CFG null class {cfg_null_class} overlaps with real classes!")
            print(f"Consider using a different null class ID.")
    
    # Create model
    net_model = UNetPlanes(
        T=modelConfig["T"],
        num_classes=num_classes,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    
    print(f"Total trainable parameters: {count_parameters(net_model):,}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net_model = torch.nn.DataParallel(net_model)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        net_model.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=modelConfig.get("weight_decay", 1e-4)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=modelConfig["epoch"], 
        eta_min=0, 
        last_epoch=-1
    )
    
    # Initialize EMA
    ema = EMA(net_model, decay=modelConfig.get("ema_decay", 0.9999))
    
    # Create trainer
    trainer = GaussianDiffusionTrainer(
        net_model, 
        modelConfig["beta_1"], 
        modelConfig["beta_T"], 
        modelConfig["T"]
    ).to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if modelConfig.get("resume_checkpoint"):
        ckpt_path = os.path.join(
            modelConfig["save_weight_dir"],
            modelConfig["resume_checkpoint"]
        )
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            net_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'ema_shadow' in checkpoint:
                ema.shadow = checkpoint['ema_shadow']
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for e in range(start_epoch, modelConfig["epoch"]):
        net_model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                optimizer.zero_grad()
                
                # Get batch data (already standardized)
                all_planes = batch['planes'].to(device)  # (B, 24, 3, H, W)
                p_energy = batch['p_energy'].to(device)
                sin_zenith = batch['sin_zenith'].to(device)
                cos_zenith = batch['cos_zenith'].to(device)
                sin_azimuth = batch['sin_azimuth'].to(device)
                cos_azimuth = batch['cos_azimuth'].to(device)
                class_id = batch['class_id'].to(device)
                
                B = all_planes.shape[0]
                
                # Randomly select a plane to train on (0-23)
                plane_indices = torch.randint(0, 24, (B,), device=device)
                
                # Get target plane and past plane
                target_planes = []
                past_planes = []
                
                for i, plane_idx in enumerate(plane_indices):
                    target_planes.append(all_planes[i, plane_idx])
                    
                    if plane_idx > 0:
                        past_planes.append(all_planes[i, plane_idx - 1])
                    else:
                        past_planes.append(torch.zeros_like(all_planes[i, 0]))
                
                target_plane = torch.stack(target_planes)  # (B, 3, H, W)
                past_plane = torch.stack(past_planes)  # (B, 3, H, W)
                
                # Apply classifier-free guidance training to ALL conditions
                if modelConfig.get("use_cfg", False):
                    mask = torch.rand(B, device=device) < modelConfig.get("cfg_drop_prob", 0.1)
                    
                    p_energy_masked = p_energy.clone()
                    p_energy_masked[mask] = 0.0
                    
                    class_id_masked = class_id.clone()
                    class_id_masked[mask] = modelConfig.get("cfg_null_class", 0)
                    
                    sin_zenith_masked = sin_zenith.clone()
                    sin_zenith_masked[mask] = 0.0
                    
                    cos_zenith_masked = cos_zenith.clone()
                    cos_zenith_masked[mask] = 0.0
                    
                    sin_azimuth_masked = sin_azimuth.clone()
                    sin_azimuth_masked[mask] = 0.0
                    
                    cos_azimuth_masked = cos_azimuth.clone()
                    cos_azimuth_masked[mask] = 0.0
                else:
                    p_energy_masked = p_energy
                    class_id_masked = class_id
                    sin_zenith_masked = sin_zenith
                    cos_zenith_masked = cos_zenith
                    sin_azimuth_masked = sin_azimuth
                    cos_azimuth_masked = cos_azimuth
                
                # Forward pass through diffusion trainer
                loss = trainer(
                    target_plane,
                    p_energy_masked,
                    class_id_masked,
                    sin_zenith_masked,
                    cos_zenith_masked,
                    sin_azimuth_masked,
                    cos_azimuth_masked,
                    plane_indices,
                    past_plane
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                
                optimizer.step()
                ema.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "avg_loss": total_loss / num_batches,
                    "LR": optimizer.param_groups[0]["lr"],
                })
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        scheduler.step()
        
        # Validation loop
        if (e + 1) % modelConfig.get("val_interval", 5) == 0:
            net_model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                    all_planes = batch['planes'].to(device)
                    p_energy = batch['p_energy'].to(device)
                    sin_zenith = batch['sin_zenith'].to(device)
                    cos_zenith = batch['cos_zenith'].to(device)
                    sin_azimuth = batch['sin_azimuth'].to(device)
                    cos_azimuth = batch['cos_azimuth'].to(device)
                    class_id = batch['class_id'].to(device)
                    
                    B = all_planes.shape[0]
                    plane_indices = torch.randint(0, 24, (B,), device=device)
                    
                    target_planes = []
                    past_planes = []
                    for i, plane_idx in enumerate(plane_indices):
                        target_planes.append(all_planes[i, plane_idx])
                        if plane_idx > 0:
                            past_planes.append(all_planes[i, plane_idx - 1])
                        else:
                            past_planes.append(torch.zeros_like(all_planes[i, 0]))
                    
                    target_plane = torch.stack(target_planes)
                    past_plane = torch.stack(past_planes)
                    
                    loss = trainer(
                        target_plane, p_energy, class_id,
                        sin_zenith, cos_zenith, sin_azimuth, cos_azimuth,
                        plane_indices, past_plane
                    )
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            print(f"\nEpoch {e+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Enhanced checkpoint saving
        if (e + 1) % modelConfig.get("save_interval", 10) == 0:
            checkpoint_path = os.path.join(
                modelConfig["save_weight_dir"], 
                f'ckpt_{e + 1}.pt'
            )
            
            torch.save({
                'epoch': e + 1,
                'model_state_dict': net_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'ema_shadow': ema.shadow,
            }, checkpoint_path)
            
            print(f"\nSaved checkpoint: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(modelConfig["save_weight_dir"], "training_history.npz")
    np.savez(history_path, train_losses=train_losses, val_losses=val_losses)
    
    print("\nTraining completed!")


def eval(modelConfig: Dict):
    """Generate samples autoregressively for all 24 planes"""
    device = torch.device(modelConfig["device"])
    
    # Load standardization stats
    mean, std = load_standardization_stats(modelConfig["data_dir"])
    mean = mean.to(device)
    std = std.to(device)
    
    print(f"\nStandardization stats loaded:")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")
    
    # Load dataset
    dataset = PlaneDataset(
        chunk_dir=modelConfig["data_dir"],
        split='test'
    )
    
    num_classes = len(torch.unique(dataset.class_id))
    
    # Create model
    model = UNetPlanes(
        T=modelConfig["T"],
        num_classes=num_classes,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Load checkpoint
    ckpt_path = os.path.join(
        modelConfig["save_weight_dir"], 
        modelConfig["test_load_weight"]
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle both old and new checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Handle DataParallel models
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {ckpt_path}")
    
    # Apply EMA weights if available
    if 'ema_shadow' in ckpt and modelConfig.get("use_ema", True):
        print("Using EMA weights for evaluation")
        for name, param in model.named_parameters():
            if name in ckpt['ema_shadow']:
                param.data = ckpt['ema_shadow'][name]
    
    model.eval()
    
    # Create sampler
    cfg_weight = modelConfig.get("cfg_weight", 0.0)
    
    if modelConfig.get("use_ddim", True):
        sampler = DDIMSamplerPlanes(
            model,
            beta_1=modelConfig["beta_1"],
            beta_T=modelConfig["beta_T"],
            T=modelConfig["T"],
            eta=modelConfig.get("ddim_eta", 0.0),
            ddim_steps=modelConfig.get("ddim_steps", 50),
            w=cfg_weight
        ).to(device)
    else:
        sampler = GaussianDiffusionSampler(
            model,
            beta_1=modelConfig["beta_1"],
            beta_T=modelConfig["beta_T"],
            T=modelConfig["T"],
            w=cfg_weight
        ).to(device)
    
    # Create autoregressive generator
    ar_generator = AutoregressivePlaneGenerator(sampler)
    
    # Sample generation
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    
    batch_size = modelConfig.get("eval_batch_size", 4)
    unique_classes = torch.unique(dataset.class_id).tolist()
    
    print("\nGenerating samples...")
    
    with torch.no_grad():
        for class_id in unique_classes[:min(3, len(unique_classes))]:
            class_mask = dataset.class_id == class_id
            class_indices = torch.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Get a few examples
            sample_idx = class_indices[:batch_size]
            
            p_energy = dataset.p_energy[sample_idx].to(device)
            sin_zenith = dataset.sin_zenith[sample_idx].to(device)
            cos_zenith = dataset.cos_zenith[sample_idx].to(device)
            sin_azimuth = dataset.sin_azimuth[sample_idx].to(device)
            cos_azimuth = dataset.cos_azimuth[sample_idx].to(device)
            class_id_tensor = dataset.class_id[sample_idx].to(device)
            
            # Get real data for comparison (already standardized)
            real_planes_std = dataset.histograms[sample_idx].to(device)  # (B, 24, 3, H, W)
            
            # Generate all 24 planes (will be in standardized space)
            generated_planes_std = ar_generator.generate_all_planes(
                batch_size=len(sample_idx),
                p_energy=p_energy,
                class_id=class_id_tensor,
                sin_zenith=sin_zenith,
                cos_zenith=cos_zenith,
                sin_azimuth=sin_azimuth,
                cos_azimuth=cos_azimuth,
                device=device,
                img_size=modelConfig["img_size"]
            )  # (B, 24, 3, H, W) - standardized
            
            # Denormalize both generated and real data
            generated_planes = denormalize_standardized(generated_planes_std, mean, std)
            real_planes = denormalize_standardized(real_planes_std, mean, std)
            
            print(f"\nClass {class_id} statistics:")
            print(f"  Generated (standardized): [{generated_planes_std.min():.3f}, {generated_planes_std.max():.3f}]")
            print(f"  Generated (denormalized): [{generated_planes.min():.3f}, {generated_planes.max():.3f}]")
            print(f"  Real (denormalized): [{real_planes.min():.3f}, {real_planes.max():.3f}]")
            
            # Save visualizations
            for sample_i in range(len(sample_idx)):
                gen_sample_planes = generated_planes[sample_i]  # (24, 3, H, W)
                real_sample_planes = real_planes[sample_i]
                
                # Create comparison visualizations for each channel
                for channel_idx, channel_name in enumerate(['density', 'energy', 'time']):
                    gen_channel = gen_sample_planes[:, channel_idx:channel_idx+1, :, :]  # (24, 1, H, W)
                    real_channel = real_sample_planes[:, channel_idx:channel_idx+1, :, :]
                    
                    # Normalize for visualization (handles unbounded data)
                    gen_vis = normalize_for_visualization(gen_channel, percentile_clip=99.5)
                    real_vis = normalize_for_visualization(real_channel, percentile_clip=99.5)
                    
                    # Create comparison grid: generated on top, real on bottom
                    combined = torch.cat([gen_vis, real_vis], dim=0)  # (48, 1, H, W)
                    grid = make_grid(combined, nrow=6, padding=2, normalize=False)
                    
                    save_path = os.path.join(
                        modelConfig["sampled_dir"],
                        f"class_{class_id}_sample_{sample_i}_{channel_name}_comparison.png"
                    )
                    save_image(grid, save_path)
                
                # Also save generated only
                density_planes = gen_sample_planes[:, 0:1, :, :]
                energy_planes = gen_sample_planes[:, 1:2, :, :]
                time_planes = gen_sample_planes[:, 2:3, :, :]
                
                # Normalize each channel separately
                density_vis = normalize_for_visualization(density_planes)
                energy_vis = normalize_for_visualization(energy_planes)
                time_vis = normalize_for_visualization(time_planes)
                
                save_image(make_grid(density_vis, nrow=6, padding=2), 
                          os.path.join(modelConfig["sampled_dir"],
                                      f"class_{class_id}_sample_{sample_i}_density.png"))
                save_image(make_grid(energy_vis, nrow=6, padding=2),
                          os.path.join(modelConfig["sampled_dir"],
                                      f"class_{class_id}_sample_{sample_i}_energy.png"))
                save_image(make_grid(time_vis, nrow=6, padding=2),
                          os.path.join(modelConfig["sampled_dir"],
                                      f"class_{class_id}_sample_{sample_i}_time.png"))
            
            print(f"Generated samples for class {class_id}")
    
    print(f"\nSamples saved to {modelConfig['sampled_dir']}")


if __name__ == "__main__":
    modelConfig = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "/path/to/preprocessed/step3/output",  # Update this path
        "save_weight_dir": "./checkpoints",
        "sampled_dir": "./samples",
        "test_load_weight": "ckpt_100.pt",
        "resume_checkpoint": None,
        
        # Model parameters
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "img_size": 64,
        
        # Diffusion parameters
        "beta_1": 1e-4,
        "beta_T": 0.02,
        
        # Training parameters
        "batch_size": 16,
        "epoch": 100,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "num_workers": 4,
        "save_interval": 10,
        "val_interval": 5,
        
        # Classifier-free guidance
        "use_cfg": True,
        "cfg_drop_prob": 0.1,
        "cfg_weight": 0.0,
        "cfg_null_class": 0,  # Make sure this doesn't overlap with real classes
        
        # EMA
        "ema_decay": 0.9999,
        "use_ema": True,
        
        # Evaluation parameters
        "use_ddim": True,
        "ddim_eta": 0.0,
        "ddim_steps": 50,
        "eval_batch_size": 4,
    }
    
    # Train or evaluate
    MODE = "train"  # or "eval"
    
    if MODE == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
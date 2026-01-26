import os
import argparse
from typing import Dict, Optional
from collections import OrderedDict
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from fnn import FNNPlanes, FNNPlanesLarge, count_parameters


class PlaneDataset(Dataset):
    """
    Memory-efficient dataset with LRU caching and optional pre-warming
    Precomputes bounding boxes from plane histograms during chunk loading
    Applies global normalization to bbox coordinates
    """
    def __init__(self, chunk_dir, split='train', cache_size=40, prewarm_cache=True,
                 bbox_mean=None, bbox_std=None):
        self.split = split
        self.chunk_dir = os.path.join(chunk_dir, split)
        self.cache_size = cache_size
        self.bbox_mean = bbox_mean  # Global mean (scalar)
        self.bbox_std = bbox_std    # Global std (scalar)

        # Read chunk index
        index_file = os.path.join(self.chunk_dir, 'index.txt')
        with open(index_file, 'r') as f:
            self.chunk_files = [line.strip() for line in f]

        # Build index
        print(f"Building index for {split} data from {len(self.chunk_files)} chunks...")
        self.chunk_index = []
        self.chunk_sizes = []

        for chunk_idx, chunk_file in enumerate(tqdm(self.chunk_files, desc=f"Indexing {split}")):
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            data = torch.load(chunk_path, map_location='cpu', weights_only=False)
            # Use bbox_ranges to determine chunk size (histograms not needed)
            chunk_size = len(data['bbox_ranges']) if 'bbox_ranges' in data else len(data.get('histograms', []))
            self.chunk_sizes.append(chunk_size)

            for sample_idx in range(chunk_size):
                self.chunk_index.append((chunk_idx, sample_idx))

            del data

        # LRU cache for chunks (stores precomputed bboxes)
        self.chunk_cache = OrderedDict()
        self.cache_lock = threading.Lock()

        print(f"Loaded index for {len(self)} samples from {split} split")
        print(f"Total chunks: {len(self.chunk_files)}, Cache size: {cache_size}")

        # Pre-warm cache
        if prewarm_cache and cache_size > 0:
            self._prewarm_cache()

    def _prewarm_cache(self):
        """Pre-load chunks into cache"""
        chunks_to_load = min(self.cache_size, len(self.chunk_files))
        print(f"Pre-warming cache with {chunks_to_load} chunks...")

        for chunk_idx in tqdm(range(chunks_to_load), desc="Loading chunks"):
            self._load_chunk(chunk_idx)

        print(f"Cache pre-warmed with {len(self.chunk_cache)} chunks")

    def _load_chunk_from_disk(self, chunk_idx):
        """Load chunk from disk and use pre-computed bbox_ranges with optional normalization"""
        chunk_file = self.chunk_files[chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_file)
        data = torch.load(chunk_path, map_location='cpu', weights_only=False)

        # Use pre-computed bbox_ranges from preprocessing (already in format: xmin, xmax, ymin, ymax)
        bboxes = data['bbox_ranges']  # (N, 24, 4)

        # Apply global normalization if statistics are provided
        if self.bbox_mean is not None and self.bbox_std is not None:
            bboxes = (bboxes - self.bbox_mean) / self.bbox_std

        data['bboxes'] = bboxes

        # Remove histograms and bbox_ranges to save memory (we only need normalized bboxes)
        if 'histograms' in data:
            del data['histograms']
        if 'bbox_ranges' in data:
            del data['bbox_ranges']

        return data

    def _load_chunk(self, chunk_idx):
        """Load chunk with LRU caching"""
        with self.cache_lock:
            if chunk_idx in self.chunk_cache:
                self.chunk_cache.move_to_end(chunk_idx)
                return self.chunk_cache[chunk_idx]

            data = self._load_chunk_from_disk(chunk_idx)
            self.chunk_cache[chunk_idx] = data

            while len(self.chunk_cache) > self.cache_size:
                self.chunk_cache.popitem(last=False)

            return data

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx):
        chunk_idx, sample_idx = self.chunk_index[idx]
        data = self._load_chunk(chunk_idx)

        # Bboxes are precomputed - just fetch them
        return {
            'bboxes': data['bboxes'][sample_idx],
            'p_energy': data['p_energy'][sample_idx],
            'sin_zenith': data['sin_zenith'][sample_idx],
            'cos_zenith': data['cos_zenith'][sample_idx],
            'sin_azimuth': data['sin_azimuth'][sample_idx],
            'cos_azimuth': data['cos_azimuth'][sample_idx],
            'class_id': data['class_id'][sample_idx] + 1
        }

    def get_sample_class_ids(self, max_chunks=3):
        """Get class IDs from a sample of chunks for validation (fast)"""
        all_class_ids = []
        chunks_to_check = min(max_chunks, len(self.chunk_files))
        for chunk_idx in range(chunks_to_check):
            data = self._load_chunk(chunk_idx)
            all_class_ids.append(data['class_id'])
        return torch.cat(all_class_ids)

    def compute_global_bbox_stats(self, max_samples=10000):
        """Compute global mean and std for all bbox coordinates from pre-computed bbox_ranges"""
        print(f"Computing global bbox statistics from training data (up to {max_samples} samples)...")
        all_bboxes = []
        samples_collected = 0

        for chunk_idx in tqdm(range(len(self.chunk_files)), desc="Loading chunks for stats"):
            chunk_file = self.chunk_files[chunk_idx]
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            data = torch.load(chunk_path, map_location='cpu', weights_only=False)

            # Use pre-computed bbox_ranges (already xmin, xmax, ymin, ymax per plane)
            bboxes = data['bbox_ranges']  # (N, 24, 4)
            all_bboxes.append(bboxes)
            samples_collected += len(bboxes)

            del data

            if samples_collected >= max_samples:
                break

        # Concatenate all bboxes and flatten to compute global statistics
        all_bboxes = torch.cat(all_bboxes, dim=0)  # (N, 24, 4)
        all_coords = all_bboxes.flatten()  # Flatten all coordinates to a single vector

        bbox_mean = all_coords.mean().item()
        bbox_std = all_coords.std().item()

        print(f"Global bbox statistics computed from {len(all_bboxes)} samples:")
        print(f"  Mean: {bbox_mean:.6f}")
        print(f"  Std:  {bbox_std:.6f}")

        return bbox_mean, bbox_std


class PlaneFNNModule(pl.LightningModule):
    """PyTorch Lightning module for FNN-based bbox regression"""

    def __init__(
        self,
        num_classes: int = 3,
        channel: int = 64,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_large_model: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        loss_type: str = 'mse',  # 'mse', 'l1', or 'huber'
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        if use_large_model:
            self.model = FNNPlanesLarge(
                num_classes=num_classes,
                ch=channel,
                num_res_blocks=num_res_blocks,
                dropout=dropout
            )
        else:
            self.model = FNNPlanes(
                num_classes=num_classes,
                ch=channel,
                num_res_blocks=num_res_blocks,
                dropout=dropout
            )

        self.model_ema = None
        print(f"Model parameters: {count_parameters(self.model):,}")

        # Set up loss function
        if loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'huber':
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _create_ema_model(self):
        ema_model = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ema_model[name] = param.data.clone()
        return ema_model

    def _update_ema(self):
        if not self.hparams.use_ema:
            return

        if self.model_ema is None:
            self.model_ema = self._create_ema_model()
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.model_ema:
                    if self.model_ema[name].device != param.device:
                        self.model_ema[name] = self.model_ema[name].to(param.device)

                    self.model_ema[name].mul_(self.hparams.ema_decay).add_(
                        param.data, alpha=1 - self.hparams.ema_decay
                    )

    def forward(self, batch):
        all_bboxes = batch['bboxes']  # (B, 24, 4)
        p_energy = batch['p_energy']
        sin_zenith = batch['sin_zenith']
        cos_zenith = batch['cos_zenith']
        sin_azimuth = batch['sin_azimuth']
        cos_azimuth = batch['cos_azimuth']
        class_id = batch['class_id']

        B = all_bboxes.shape[0]

        # Flatten target to (B, 96) for all 24 planes at once
        target = all_bboxes.view(B, -1)  # (B, 96)

        # Forward pass - predict bboxes directly
        pred = self.model(
            p_energy,
            class_id,
            sin_zenith,
            cos_zenith,
            sin_azimuth,
            cos_azimuth
        )  # (B, 96)

        # Compute loss
        loss = self.loss_fn(pred, target, reduction='mean')

        return loss, pred, target

    def training_step(self, batch, batch_idx):
        loss, pred, target = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)

        # Log MAE for interpretability
        with torch.no_grad():
            mae = F.l1_loss(pred, target, reduction='mean')
            self.log('train_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log MAE for interpretability
        with torch.no_grad():
            mae = F.l1_loss(pred, target, reduction='mean')
            self.log('val_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.use_ema and self.model_ema is not None:
            checkpoint['model_ema'] = self.model_ema

    def on_load_checkpoint(self, checkpoint):
        if self.hparams.use_ema and 'model_ema' in checkpoint:
            self.model_ema = checkpoint['model_ema']
            if self.model_ema is not None:
                device = next(self.model.parameters()).device
                for name in self.model_ema:
                    self.model_ema[name] = self.model_ema[name].to(device)


class PlaneDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module with global normalization"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        cache_size: int = 40,
        prefetch_factor: int = 4,
        prewarm_cache: bool = True,
        compute_global_stats: bool = True,
        max_samples_for_stats: int = 10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.prefetch_factor = prefetch_factor
        self.prewarm_cache = prewarm_cache
        self.compute_global_stats = compute_global_stats
        self.max_samples_for_stats = max_samples_for_stats
        self.bbox_mean = None
        self.bbox_std = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # First, create train dataset WITHOUT normalization to compute stats
            temp_train_dataset = PlaneDataset(
                self.data_dir,
                split='train',
                cache_size=self.cache_size,
                prewarm_cache=False,  # Don't prewarm yet
                bbox_mean=None,
                bbox_std=None
            )

            # Compute global bbox statistics from training data
            if self.compute_global_stats:
                print("\n" + "="*60)
                print("COMPUTING GLOBAL NORMALIZATION STATISTICS FROM TRAINING DATA")
                print("="*60)
                self.bbox_mean, self.bbox_std = temp_train_dataset.compute_global_bbox_stats(
                    max_samples=self.max_samples_for_stats
                )
                print(f"Global bbox normalization: mean={self.bbox_mean:.6f}, std={self.bbox_std:.6f}")
                print("="*60 + "\n")

                # Save statistics to file
                stats_file = os.path.join(self.data_dir, 'global_bbox_stats.pt')
                torch.save({
                    'bbox_mean': self.bbox_mean,
                    'bbox_std': self.bbox_std,
                    'max_samples': self.max_samples_for_stats,
                    'notes': 'Global normalization statistics for all bbox coordinates (xmin, xmax, ymin, ymax across all planes)'
                }, stats_file)
                print(f"Saved global bbox statistics to: {stats_file}\n")

            del temp_train_dataset

            # Now create datasets WITH normalization
            self.train_dataset = PlaneDataset(
                self.data_dir,
                split='train',
                cache_size=self.cache_size,
                prewarm_cache=self.prewarm_cache,
                bbox_mean=self.bbox_mean,
                bbox_std=self.bbox_std
            )
            self.val_dataset = PlaneDataset(
                self.data_dir,
                split='val',
                cache_size=max(5, self.cache_size // 4),
                prewarm_cache=False,  # Don't prewarm validation
                bbox_mean=self.bbox_mean,
                bbox_std=self.bbox_std
            )

            # Verify class IDs (sample check - fast)
            print("\nVerifying class IDs (sampling a few chunks)...")
            train_classes_raw = torch.unique(self.train_dataset.get_sample_class_ids(max_chunks=3)).tolist()
            val_classes_raw = torch.unique(self.val_dataset.get_sample_class_ids(max_chunks=3)).tolist()

            train_classes = sorted([c + 1 for c in train_classes_raw])
            val_classes = sorted([c + 1 for c in val_classes_raw])

            print(f"Train classes (raw, sampled): {sorted(train_classes_raw)} -> shifted: {train_classes}")
            print(f"Val classes   (raw, sampled): {sorted(val_classes_raw)} -> shifted: {val_classes}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(2, self.num_workers // 2),
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train FNN Bbox Regression Model')

    # Paths
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_weight_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None)

    # Model
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_large_model', action='store_true', default=False,
                       help='Use the larger FNN model variant')

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cache_size', type=int, default=40)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--prewarm_cache', action='store_true', default=True)
    parser.add_argument('--save_interval', type=int, default=30)
    parser.add_argument('--val_check_interval', type=float, default=1.0)

    # Loss function
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1', 'huber'],
                       help='Loss function type')

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', action='store_false', dest='use_ema')
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # Hardware
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])

    return parser.parse_args()


def train(args):
    """Main training function"""

    data_module = PlaneDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        prefetch_factor=args.prefetch_factor,
        prewarm_cache=args.prewarm_cache,
        compute_global_stats=True,
        max_samples_for_stats=10000
    )

    model = PlaneFNNModule(
        num_classes=3,
        channel=args.channel,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_large_model=args.use_large_model,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        loss_type=args.loss_type,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_weight_dir,
        filename='epoch_{epoch:03d}-val_loss_{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs=args.save_interval
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='plane_fnn'
    )

    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()

    # Set matmul precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # DDP strategy with optimizations
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
    else:
        strategy = 'auto'

    trainer = pl.Trainer(
        max_epochs=args.epoch,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=num_gpus if torch.cuda.is_available() else 1,
        strategy=strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=args.grad_clip,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=50,
        precision=args.precision
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_checkpoint
    )

    print("\nTraining completed!")
    print(f"Checkpoints: {args.save_weight_dir}")
    print(f"Logs: {args.log_dir}")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Training Configuration (FNN):")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("=" * 60)

    train(args)

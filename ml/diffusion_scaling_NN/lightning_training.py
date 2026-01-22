import os
import argparse
from typing import Dict, Optional
from collections import OrderedDict
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from diffusion import GaussianDiffusionTrainer
from unet import UNetPlanes, count_parameters


class PlaneDataset(Dataset):
    """
    Memory-efficient dataset with LRU caching and optional pre-warming
    """
    def __init__(self, chunk_dir, split='train', cache_size=40, prewarm_cache=True):
        self.split = split
        self.chunk_dir = os.path.join(chunk_dir, split)
        self.cache_size = cache_size
        
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
            chunk_size = len(data['histograms'])
            self.chunk_sizes.append(chunk_size)
            
            for sample_idx in range(chunk_size):
                self.chunk_index.append((chunk_idx, sample_idx))
            
            del data
        
        # LRU cache for chunks
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
        """Load chunk from disk"""
        chunk_file = self.chunk_files[chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_file)
        return torch.load(chunk_path, map_location='cpu', weights_only=False)
    
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
        
        return {
            'planes': data['histograms'][sample_idx],
            'p_energy': data['p_energy'][sample_idx],
            'sin_zenith': data['sin_zenith'][sample_idx],
            'cos_zenith': data['cos_zenith'][sample_idx],
            'sin_azimuth': data['sin_azimuth'][sample_idx],
            'cos_azimuth': data['cos_azimuth'][sample_idx],
            'class_id': data['class_id'][sample_idx] + 1
        }
    
    def get_all_class_ids(self):
        """Get all class IDs for validation"""
        all_class_ids = []
        for chunk_idx in range(len(self.chunk_files)):
            data = self._load_chunk(chunk_idx)
            all_class_ids.append(data['class_id'])
        return torch.cat(all_class_ids)


class PlaneDiffusionModule(pl.LightningModule):
    """PyTorch Lightning module for plane diffusion"""
    
    def __init__(
        self,
        num_classes: int = 3,
        T: int = 1000,
        channel: int = 64,
        channel_mult: list = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_cfg: bool = True,
        cfg_drop_prob: float = 0.1,
        cfg_null_class: int = 0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = UNetPlanes(
            T=T,
            num_classes=num_classes,
            ch=channel,
            ch_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        # Create diffusion trainer
        self.diffusion_trainer = GaussianDiffusionTrainer(
            self.model,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T
        )
        
        self.model_ema = None
        print(f"Model parameters: {count_parameters(self.model):,}")
    
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
        all_planes = batch['planes']
        p_energy = batch['p_energy']
        sin_zenith = batch['sin_zenith']
        cos_zenith = batch['cos_zenith']
        sin_azimuth = batch['sin_azimuth']
        cos_azimuth = batch['cos_azimuth']
        class_id = batch['class_id']
        
        B = all_planes.shape[0]
        device = all_planes.device
        
        # Randomly select plane
        plane_indices = torch.randint(0, 24, (B,), device=device)
        
        # Get target and past planes
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
        
        # Classifier-free guidance
        if self.hparams.use_cfg:
            mask = torch.rand(B, device=device) < self.hparams.cfg_drop_prob
            
            p_energy_masked = p_energy.clone()
            p_energy_masked[mask] = 0.0
            
            class_id_masked = class_id.clone()
            class_id_masked[mask] = self.hparams.cfg_null_class
            
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
        
        # Forward pass
        loss = self.diffusion_trainer(
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
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
    """PyTorch Lightning data module"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        cache_size: int = 40,
        prefetch_factor: int = 4,
        prewarm_cache: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.prefetch_factor = prefetch_factor
        self.prewarm_cache = prewarm_cache
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PlaneDataset(
                self.data_dir, 
                split='train', 
                cache_size=self.cache_size,
                prewarm_cache=self.prewarm_cache
            )
            self.val_dataset = PlaneDataset(
                self.data_dir, 
                split='val', 
                cache_size=max(5, self.cache_size // 4),
                prewarm_cache=False  # Don't prewarm validation
            )
            
            # Verify class IDs
            print("\nVerifying class IDs...")
            train_classes_raw = torch.unique(self.train_dataset.get_all_class_ids()).tolist()
            val_classes_raw = torch.unique(self.val_dataset.get_all_class_ids()).tolist()

            train_classes = sorted([c + 1 for c in train_classes_raw])
            val_classes = sorted([c + 1 for c in val_classes_raw])

            print(f"Train classes (raw): {sorted(train_classes_raw)} -> shifted: {train_classes}")
            print(f"Val classes   (raw): {sorted(val_classes_raw)} -> shifted: {val_classes}")

            assert set(train_classes) == {1, 2, 3}, f"Expected [1,2,3], got {train_classes}"
    
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
    parser = argparse.ArgumentParser(description='Train Plane Diffusion Model')
    
    # Paths
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_weight_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    
    # Model
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Diffusion
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cache_size', type=int, default=40)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--prewarm_cache', action='store_true', default=True)
    parser.add_argument('--save_interval', type=int, default=30)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    
    # CFG
    parser.add_argument('--use_cfg', action='store_true', default=True)
    parser.add_argument('--no_cfg', action='store_false', dest='use_cfg')
    parser.add_argument('--cfg_drop_prob', type=float, default=0.1)
    parser.add_argument('--cfg_null_class', type=int, default=0)
    
    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', action='store_false', dest='use_ema')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    
    # Hardware
    parser.add_argument('--num_gpus', type=int, default=None)

    parser.add_argument('--precision', type=int, default=16, choices=[16, 32])
    
    return parser.parse_args()


def train(args):
    """Main training function"""
    
    data_module = PlaneDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        prefetch_factor=args.prefetch_factor,
        prewarm_cache=args.prewarm_cache
    )
    
    model = PlaneDiffusionModule(
        num_classes=3,
        T=args.T,
        channel=args.channel,
        channel_mult=args.channel_mult,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_cfg=args.use_cfg,
        cfg_drop_prob=args.cfg_drop_prob,
        cfg_null_class=args.cfg_null_class,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay
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
        name='plane_diffusion'
    )
    
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    
    # Set matmul precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    
    # DDP strategy with optimizations
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # Fixes stride mismatch warning
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
    print("Training Configuration:")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("=" * 60)
    
    train(args)
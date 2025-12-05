import sys
# sys.path.append('/n/home04/hhanif/tambo_optimization')
sys.path.append('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization')

import os
import glob
from typing import Dict, Tuple, Any

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


from models.FlowCondition import OTCondFlowMatcher  
from models.ModelConditionFlow import UNet

try:
    # PL >= 1.6-ish
    from pytorch_lightning.callbacks import TQDMProgressBar
except ImportError:
    # PL 2.x layout
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
try:
    # Lightning 2.x
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    # Older PL
    from pytorch_lightning.utilities.distributed import rank_zero_only

@rank_zero_only
def rank0_print(*args, **kwargs):
    print(*args, **kwargs)


class CustomTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.pos = 0
        bar.leave = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.pos = 0
        bar.leave = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.pos = 0
        bar.leave = True
        return bar


# ============================================================
# 1. LOAD & BUILD DATAFRAME FROM NPZ FILES
# ============================================================

import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import os

MAX_TENSORS = 24

# Class mapping for PDG directories
PDG_CLASS_MAP = {
    "pdg_-11": 0,
    "pdg_11": 0,
    "pdg_111": 1,
    "pdg_-211": 2,
    "pdg_211": 2,
}


def npz_to_run_rows(npz_path: str, primary_pdg: str, class_id: int) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)

    tensors = data["tensors"]
    energy = data["energy"]
    sin_zenith = data["sin_zenith"]
    cos_zenith = data["cos_zenith"]
    sin_azimuth = data["sin_azimuth"]
    cos_azimuth = data["cos_azimuth"]
    run_ids = data["run_ids"]
    bins = int(data["bins"])

    N = len(run_ids)
    print(f"Loaded {N} tensors from {os.path.basename(npz_path)} with shape (3, {bins}, {bins})")

    assert (
        len(energy) == len(sin_zenith) == len(cos_zenith) ==
        len(sin_azimuth) == len(cos_azimuth) == tensors.shape[0] == N
    ), "Metadata arrays must all have the same length as tensors."

    by_run = defaultdict(list)
    for idx, run_id in enumerate(run_ids):
        by_run[str(run_id)].append(idx)

    rows = []
    for run_id, idx_list in by_run.items():
        idx_list = sorted(idx_list)
        first_idx = idx_list[0]

        row = {
            "run_id": run_id,
            "energy": float(energy[first_idx]),
            "sin_zenith": float(sin_zenith[first_idx]),
            "cos_zenith": float(cos_zenith[first_idx]),
            "sin_azimuth": float(sin_azimuth[first_idx]),
            "cos_azimuth": float(cos_azimuth[first_idx]),
            "primary_pdg": primary_pdg,
            "class_id": class_id,
        }

        for j in range(MAX_TENSORS):
            col_name = f"tensor{j+1}"
            if j < len(idx_list):
                row[col_name] = tensors[idx_list[j]]
            else:
                row[col_name] = None

        rows.append(row)

    return pd.DataFrame(rows)


# if __name__ == "__main__":
if True:
    base_dir = (
        "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/"
        "tambo_simulation_nov_25/pre_processed_npz"
    )

    # Find all NPZ files
    pattern = os.path.join(base_dir, "*", "combined_pdg_*.npz")
    npz_files = sorted(glob.glob(pattern))

    print(f"Found {len(npz_files)} NPZ files.")

    all_dfs = []

    for npz_path in npz_files:
        # Extract primary directory name  
        primary_pdg = os.path.basename(os.path.dirname(npz_path))

        # Map to class id
        class_id = PDG_CLASS_MAP.get(primary_pdg, -1)

        df = npz_to_run_rows(npz_path, primary_pdg, class_id)
        df["source_file"] = npz_path

        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print("Final combined DataFrame shape:", full_df.shape)


tensor_cols = [f"tensor{i}" for i in range(1, 25)]

# Replacement tensor of shape (3,16,16)
zero_tensor = np.zeros((3, 16, 16))

# Replace None values in each tensor column
for col in tensor_cols:
    full_df[col] = full_df[col].apply(lambda x: zero_tensor if x is None else x)



allowed_pdgs = ["pdg_211", "pdg_-211", "pdg_11", "pdg_-11", "pdg_111"]

dfs = []
for pdg in allowed_pdgs:
    subset = full_df[full_df["primary_pdg"] == pdg].sample(n=23000, random_state=42)
    dfs.append(subset)

df_all = pd.concat(dfs, ignore_index=True)

print(df_all.shape)   




# ============================================================
# 2. CLEAN FEATURES & SCALE
# ============================================================

cols = ["energy", "sin_zenith", "cos_zenith", "sin_azimuth", "cos_azimuth"]
df_cleaned = df_all.copy()

for col in cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype(float)


def to_scalar(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


df_cleaned["energy"] = df_cleaned["energy"].map(to_scalar)
df_cleaned["energy"] = pd.to_numeric(df_cleaned["energy"], errors="coerce").astype(float)
df_cleaned["energy"] = np.log(df_cleaned["energy"])

# Scale the continuous columns
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])


# ============================================================
# 3. DATASET & DATAMODULE
# ============================================================

class TamboDataset(Dataset):
    """
    Dataset wrapping:
      'rgb_32', 'energy', 'sin_zenith', 'cos_zenith', 'sin_azimuth', 'cos_azimuth'
    """

    def __init__(self, df: pd.DataFrame):
        self.rgb_32      = df["rgb_32"].values
        self.energy      = df["energy"].values
        self.sin_zenith  = df["sin_zenith"].values
        self.cos_zenith  = df["cos_zenith"].values
        self.sin_azimuth = df["sin_azimuth"].values
        self.cos_azimuth = df["cos_azimuth"].values

    def __len__(self):
        return len(self.rgb_32)

    def __getitem__(self, idx):
        # Image: (H, W, 3) -> (3, H, W)
        img_np = self.rgb_32[idx]
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)

        # scalar features
        energy      = self.energy[idx]
        sin_zenith  = self.sin_zenith[idx]
        cos_zenith  = self.cos_zenith[idx]
        sin_azimuth = self.sin_azimuth[idx]
        cos_azimuth = self.cos_azimuth[idx]

        return img, (
            torch.tensor(energy,      dtype=torch.float32),
            torch.tensor(sin_zenith,  dtype=torch.float32),
            torch.tensor(cos_zenith,  dtype=torch.float32),
            torch.tensor(sin_azimuth, dtype=torch.float32),
            torch.tensor(cos_azimuth, dtype=torch.float32),
        )


class TamboDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        dataset = TamboDataset(self.df)

        n = len(dataset)
        train_len = int(n * self.train_ratio)
        val_len   = int(n * self.val_ratio)
        test_len  = n - train_len - val_len

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

        img0, cond0 = self.train_set[0]
        rank0_print("Sample image shape:", img0.shape)
        rank0_print("Total samples:", len(dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ============================================================
# 4. LIGHTNING MODULE FOR CONDITIONAL DIFFUSION
# ============================================================


class TamboFlowLitModel(pl.LightningModule):
    def __init__(
        self,
        ch: int = 128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks: int = 3,
        dropout: float = 0.15,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        p_uncond: float = 0.1,
        t_max_scheduler: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Continuous-time UNet: no T needed
        self.model = UNet(
            ch=ch,
            ch_mult=list(ch_mult),
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        self.flow = OTCondFlowMatcher()

    def _split_cond(self, cond):
        energy, sin_z, cos_z, sin_a, cos_a = cond
        return energy, sin_z, cos_z, sin_a, cos_a

    def _null_condition(self, energy, sin_z, cos_z, sin_a, cos_a):
        p_uncond = self.hparams.p_uncond
        bs = energy.shape[0]
        mask = (torch.rand(bs, device=energy.device) < p_uncond)

        # Clone before modifying to avoid in-place ops on graph inputs
        energy = energy.clone()
        sin_z  = sin_z.clone()
        cos_z  = cos_z.clone()
        sin_a  = sin_a.clone()
        cos_a  = cos_a.clone()

        energy[mask] = 0.0
        sin_z[mask]  = 0.0
        cos_z[mask]  = 0.0
        sin_a[mask]  = 0.0
        cos_a[mask]  = 0.0

        return energy, sin_z, cos_z, sin_a, cos_a


    def training_step(self, batch, batch_idx):
        img, cond = batch
        energy, sin_z, cos_z, sin_a, cos_a = self._split_cond(cond)

        # classifier-free null-conditioning
        energy, sin_z, cos_z, sin_a, cos_a = self._null_condition(
            energy, sin_z, cos_z, sin_a, cos_a
        )

        # FlowMatching loss:
        # cfm_loss sends (x_t, t) into self.model(x_t, t, energy, sin_z, cos_z, sin_a, cos_a)
        loss = self.flow.cfm_loss(
            self.model,   # network
            img,          # x_1
            None,         # x_0 (sampled as N(0, I) inside cfm_loss if None)
            energy,
            sin_z,
            cos_z,
            sin_a,
            cos_a,
        )


        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, cond = batch
        energy, sin_z, cos_z, sin_a, cos_a = self._split_cond(cond)

        loss = self.flow.cfm_loss(
            self.model,   # network
            img,          # x_1
            None,         # x_0 (sampled as N(0, I) inside cfm_loss if None)
            energy,
            sin_z,
            cos_z,
            sin_a,
            cos_a,
        )


        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max_scheduler,
            eta_min=0.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }


# ============================================================
# 5. TRAINING ENTRY POINT (MULTI-GPU VIA LIGHTNING)
# ============================================================

def main():
    pl.seed_everything(42)

    # Data
    data_module = TamboDataModule(
        df=df_cleaned,
        batch_size=64,
        train_ratio=0.85,
        val_ratio=0.10,
        test_ratio=0.05,
        num_workers=4,
        seed=42,
    )

    # Model
    model = TamboFlowLitModel(
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=3,
        dropout=0.15,
        learning_rate=1e-4,
        weight_decay=1e-4,
        p_uncond=0.1,
        t_max_scheduler=10000,
    )



    ckpt_dir = "/n/home04/hhanif/tambo_ckpts/flow/"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="ckpt_{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=400,
        save_weights_only=False,
    )

    progress_bar = CustomTQDMProgressBar()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        max_epochs=2000,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    print("Start training Conditional DDPMS (Lightning multi-GPU)...")
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

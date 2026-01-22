# model_bbox.py - Bounding Box Regression Architecture - BBOX OUTPUT
import math
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.timembedding(t)

class ContinuousEmbedding(nn.Module):
    """Embedding for continuous scalar values"""
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, embed_dim),
            Swish(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return self.fc(x)

class ClassEmbedding(nn.Module):
    """Embedding for discrete class labels"""
    def __init__(self, num_classes, d_model, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_classes + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        return self.embedding(x)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        return self.c1(x) + self.c2(x)

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        x = self.t(x)
        return self.c(x)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 3, 1).view(B, H * W, C)
        k = self.proj_k(h).view(B, C, H * W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)
        v = self.proj_v(h).permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.proj(h)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),  # FIXED: Added min()
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(32, out_ch), out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb, cemb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None] + self.cond_proj(cemb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)

class UNetPlanes(nn.Module):
    """
    MLP-based model for bounding box regression with diffusion:
    - Predicts noise for ALL 24 bounding boxes at once (24×4 = 96 values)
    - No autoregression needed

    Conditioned on: p_energy, class_id, sin_zenith, cos_zenith, sin_azimuth, cos_azimuth
    """
    def __init__(self, T, num_classes, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        hidden_dim = ch * 8  # MLP hidden dimension
        output_dim = 24 * 4  # 24 planes × 4 coordinates

        # Time embedding for diffusion
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # Class embedding (discrete)
        self.class_embedding = ClassEmbedding(num_classes, ch, tdim)

        # Continuous embeddings for physics parameters
        self.p_energy_embedding = ContinuousEmbedding(tdim)
        self.sin_zenith_embedding = ContinuousEmbedding(tdim)
        self.cos_zenith_embedding = ContinuousEmbedding(tdim)
        self.sin_azimuth_embedding = ContinuousEmbedding(tdim)
        self.cos_azimuth_embedding = ContinuousEmbedding(tdim)

        # MLP for bbox processing
        # Input: 96 (all 24 planes × 4 bbox coords, noisy)
        self.bbox_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim * 2),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Combine bbox features with conditioning
        # hidden_dim + 2*tdim (time + all other conditions)
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + 2 * tdim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Output head: predict noise for all 24 bboxes (96 values)
        self.diffusion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth):
        """
        Args:
            x: (B, 96) - noisy bounding boxes for all 24 planes (flattened)
            t: (B,) - diffusion timestep
            p_energy: (B,) - primary energy (normalized)
            class_id: (B,) - particle class ID
            sin_zenith, cos_zenith: (B,) - zenith angle
            sin_azimuth, cos_azimuth: (B,) - azimuth angle

        Returns:
            noise_pred: (B, 96) - predicted noise for all bounding boxes
        """
        # Time embedding
        temb = self.time_embedding(t)  # (B, tdim)

        # Combine all condition embeddings
        cemb = (
            self.class_embedding(class_id) +
            self.p_energy_embedding(p_energy) +
            self.sin_zenith_embedding(sin_zenith) +
            self.cos_zenith_embedding(cos_zenith) +
            self.sin_azimuth_embedding(sin_azimuth) +
            self.cos_azimuth_embedding(cos_azimuth)
        )  # (B, tdim)

        # Encode bbox
        h = self.bbox_encoder(x)  # (B, hidden_dim)

        # Combine with conditioning
        h = torch.cat([h, temb, cemb], dim=1)  # (B, hidden_dim + 2*tdim)
        h = self.combiner(h)  # (B, hidden_dim)

        # Predict noise for all bboxes
        noise_pred = self.diffusion_head(h)  # (B, 96)

        return noise_pred


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    batch_size = 8
    model = UNetPlanes(
        T=1000,
        num_classes=5,  # number of particle types
        ch=64,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        dropout=0.1
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, 96)  # noisy bboxes for all 24 planes (24×4=96)
    t = torch.randint(1000, size=[batch_size])
    p_energy = torch.rand(batch_size)
    class_id = torch.randint(5, size=[batch_size])
    sin_zenith = torch.rand(batch_size)
    cos_zenith = torch.rand(batch_size)
    sin_azimuth = torch.rand(batch_size)
    cos_azimuth = torch.rand(batch_size)

    # Test forward pass
    noise_pred = model(
        x, t, p_energy, class_id, sin_zenith, cos_zenith,
        sin_azimuth, cos_azimuth
    )
    print(f"Noise prediction shape: {noise_pred.shape}")  # (8, 96)
    assert noise_pred.shape == (batch_size, 96), f"Expected ({batch_size}, 96), got {noise_pred.shape}"
    print("✓ Model test passed!")

# model_planes_single.py - Single Head Architecture (Diffusion Only) - FIXED
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

class PlaneIndexEmbedding(nn.Module):
    """Embedding for plane index (0-23)"""
    def __init__(self, d_model, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=24, embedding_dim=d_model),
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
    UNet for plane-based diffusion with single head:
    - Predicts noise for plane content (diffusion)
    
    Conditioned on: p_energy, class_id, sin_zenith, cos_zenith, sin_azimuth, cos_azimuth, 
                    plane_index, past_plane
    """
    def __init__(self, T, num_classes, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        
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
        
        # Plane index embedding (0-23)
        self.plane_embedding = PlaneIndexEmbedding(ch, tdim)
        
        # Input: 3 channels (current noisy plane) + 3 channels (past plane) = 6 channels
        self.head = nn.Conv2d(6, ch, 3, padding=1)
        
        # Downsampling blocks
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        
        # Middle blocks
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        
        # Upsampling blocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
        
        # Diffusion head for plane content
        # Output: 3 channels (density, energy, time) noise prediction
        self.diffusion_head = nn.Sequential(
            nn.GroupNorm(min(32, now_ch), now_ch),  # FIXED: Added min()
            Swish(),
            nn.Conv2d(now_ch, 3, 3, padding=1)
        )

    def forward(self, x, t, p_energy, class_id, sin_zenith, cos_zenith, 
                sin_azimuth, cos_azimuth, plane_idx, past_plane):
        """
        Args:
            x: (B, 3, H, W) - noisy current plane
            t: (B,) - diffusion timestep
            p_energy: (B,) - primary energy (normalized)
            class_id: (B,) - particle class ID
            sin_zenith, cos_zenith: (B,) - zenith angle
            sin_azimuth, cos_azimuth: (B,) - azimuth angle
            plane_idx: (B,) - plane index (0-23)
            past_plane: (B, 3, H, W) - previous plane (or zeros for first plane)
        
        Returns:
            noise_pred: (B, 3, H, W) - predicted noise
        """
        # Time embedding
        temb = self.time_embedding(t)
        
        # Combine all condition embeddings
        cemb = (
            self.class_embedding(class_id) +
            self.p_energy_embedding(p_energy) +
            self.sin_zenith_embedding(sin_zenith) +
            self.cos_zenith_embedding(cos_zenith) +
            self.sin_azimuth_embedding(sin_azimuth) +
            self.cos_azimuth_embedding(cos_azimuth) +
            self.plane_embedding(plane_idx)
        )
        
        # Concatenate current noisy plane with past plane
        h = torch.cat([x, past_plane], dim=1)  # (B, 6, H, W)
        h = self.head(h)
        
        # Downsampling
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        
        # Middle (bottleneck)
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        
        assert len(hs) == 0
        
        # Predict noise for plane content
        noise_pred = self.diffusion_head(h)
        
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
    x = torch.randn(batch_size, 3, 64, 64)  # noisy plane
    t = torch.randint(1000, size=[batch_size])
    p_energy = torch.rand(batch_size)
    class_id = torch.randint(5, size=[batch_size])
    sin_zenith = torch.rand(batch_size)
    cos_zenith = torch.rand(batch_size)
    sin_azimuth = torch.rand(batch_size)
    cos_azimuth = torch.rand(batch_size)
    plane_idx = torch.randint(24, size=[batch_size])
    past_plane = torch.randn(batch_size, 3, 64, 64)
    
    # Test forward pass
    noise_pred = model(
        x, t, p_energy, class_id, sin_zenith, cos_zenith, 
        sin_azimuth, cos_azimuth, plane_idx, past_plane
    )
    print(f"Noise prediction shape: {noise_pred.shape}")  # (8, 3, 64, 64)
    print("✓ Model test passed!")

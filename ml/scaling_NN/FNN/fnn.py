# fnn.py - Feedforward Neural Network for Bounding Box Regression
"""
FNN model for direct bounding box prediction.
Unlike the diffusion model, this directly regresses the bounding boxes
from conditioning features without iterative denoising.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class FNNPlanes(nn.Module):
    """
    Feedforward Neural Network for direct bbox range prediction:
    - Directly predicts bbox ranges for ALL 24 planes at once (24×4 = 96 values)
    - Each plane has 4 values: xmin, xmax, ymin, ymax
    - No diffusion/denoising - direct regression from conditioning features

    Conditioned on: p_energy, class_id, sin_zenith, cos_zenith, sin_azimuth, cos_azimuth
    """
    def __init__(self, num_classes, ch=64, num_res_blocks=4, dropout=0.1):
        super().__init__()
        embed_dim = ch * 4  # Embedding dimension
        hidden_dim = ch * 8  # MLP hidden dimension
        output_dim = 24 * 4  # 24 planes × 4 coordinates

        # Class embedding (discrete)
        self.class_embedding = ClassEmbedding(num_classes, ch, embed_dim)

        # Continuous embeddings for physics parameters
        self.p_energy_embedding = ContinuousEmbedding(embed_dim)
        self.sin_zenith_embedding = ContinuousEmbedding(embed_dim)
        self.cos_zenith_embedding = ContinuousEmbedding(embed_dim)
        self.sin_azimuth_embedding = ContinuousEmbedding(embed_dim)
        self.cos_azimuth_embedding = ContinuousEmbedding(embed_dim)

        # Input projection from combined embeddings
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Residual blocks for deep processing
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        # Additional MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Output head: predict bbox ranges for all 24 planes (96 values)
        # Output: 24 planes × 4 coords (xmin, xmax, ymin, ymax)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth):
        """
        Args:
            p_energy: (B,) - primary energy (normalized)
            class_id: (B,) - particle class ID
            sin_zenith, cos_zenith: (B,) - zenith angle
            sin_azimuth, cos_azimuth: (B,) - azimuth angle

        Returns:
            bbox_pred: (B, 96) - predicted bbox ranges for all 24 planes
                                 Format: [plane0_xmin, plane0_xmax, plane0_ymin, plane0_ymax, ...]
        """
        # Combine all condition embeddings by addition
        h = (
            self.class_embedding(class_id) +
            self.p_energy_embedding(p_energy) +
            self.sin_zenith_embedding(sin_zenith) +
            self.cos_zenith_embedding(cos_zenith) +
            self.sin_azimuth_embedding(sin_azimuth) +
            self.cos_azimuth_embedding(cos_azimuth)
        )  # (B, embed_dim)

        # Project to hidden dimension
        h = self.input_proj(h)  # (B, hidden_dim)

        # Apply residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)  # (B, hidden_dim)

        # Additional MLP processing
        h = self.mlp(h)  # (B, hidden_dim)

        # Output bbox predictions
        bbox_pred = self.output_head(h)  # (B, 96)

        return bbox_pred


class FNNPlanesLarge(nn.Module):
    """
    Larger FNN variant with more capacity for complex patterns.
    Uses concatenation instead of addition for embeddings to preserve information.
    """
    def __init__(self, num_classes, ch=64, num_res_blocks=6, dropout=0.1):
        super().__init__()
        embed_dim = ch * 2  # Individual embedding dimension
        combined_dim = embed_dim * 6  # 6 conditioning inputs concatenated
        hidden_dim = ch * 8  # MLP hidden dimension
        output_dim = 24 * 4  # 24 planes × 4 coordinates

        # Class embedding (discrete)
        self.class_embedding = ClassEmbedding(num_classes, ch, embed_dim)

        # Continuous embeddings for physics parameters
        self.p_energy_embedding = ContinuousEmbedding(embed_dim)
        self.sin_zenith_embedding = ContinuousEmbedding(embed_dim)
        self.cos_zenith_embedding = ContinuousEmbedding(embed_dim)
        self.sin_azimuth_embedding = ContinuousEmbedding(embed_dim)
        self.cos_azimuth_embedding = ContinuousEmbedding(embed_dim)

        # Input projection from concatenated embeddings
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Residual blocks for deep processing
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        # Additional MLP layers with layer norm
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
            nn.Dropout(dropout),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, p_energy, class_id, sin_zenith, cos_zenith,
                sin_azimuth, cos_azimuth):
        """
        Args:
            p_energy: (B,) - primary energy (normalized)
            class_id: (B,) - particle class ID
            sin_zenith, cos_zenith: (B,) - zenith angle
            sin_azimuth, cos_azimuth: (B,) - azimuth angle

        Returns:
            bbox_pred: (B, 96) - predicted bbox ranges for all 24 planes
        """
        # Get all embeddings
        class_emb = self.class_embedding(class_id)
        energy_emb = self.p_energy_embedding(p_energy)
        sin_z_emb = self.sin_zenith_embedding(sin_zenith)
        cos_z_emb = self.cos_zenith_embedding(cos_zenith)
        sin_a_emb = self.sin_azimuth_embedding(sin_azimuth)
        cos_a_emb = self.cos_azimuth_embedding(cos_azimuth)

        # Concatenate all embeddings
        h = torch.cat([
            class_emb, energy_emb, sin_z_emb, cos_z_emb, sin_a_emb, cos_a_emb
        ], dim=-1)  # (B, combined_dim)

        # Project to hidden dimension
        h = self.input_proj(h)  # (B, hidden_dim)

        # Apply residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)

        # Additional MLP processing
        h = self.mlp(h)

        # Output bbox predictions
        bbox_pred = self.output_head(h)  # (B, 96)

        return bbox_pred


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    batch_size = 8

    # Test standard FNN
    print("Testing FNNPlanes (standard):")
    model = FNNPlanes(
        num_classes=5,
        ch=64,
        num_res_blocks=4,
        dropout=0.1
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Test inputs
    p_energy = torch.rand(batch_size)
    class_id = torch.randint(5, size=[batch_size])
    sin_zenith = torch.rand(batch_size)
    cos_zenith = torch.rand(batch_size)
    sin_azimuth = torch.rand(batch_size)
    cos_azimuth = torch.rand(batch_size)

    # Test forward pass
    bbox_pred = model(p_energy, class_id, sin_zenith, cos_zenith,
                      sin_azimuth, cos_azimuth)
    print(f"Output shape: {bbox_pred.shape}")  # (8, 96)
    assert bbox_pred.shape == (batch_size, 96), f"Expected ({batch_size}, 96), got {bbox_pred.shape}"
    print("✓ FNNPlanes test passed!")

    # Test large FNN
    print("\nTesting FNNPlanesLarge:")
    model_large = FNNPlanesLarge(
        num_classes=5,
        ch=64,
        num_res_blocks=6,
        dropout=0.1
    )
    print(f"Model parameters: {count_parameters(model_large):,}")

    bbox_pred_large = model_large(p_energy, class_id, sin_zenith, cos_zenith,
                                   sin_azimuth, cos_azimuth)
    print(f"Output shape: {bbox_pred_large.shape}")
    assert bbox_pred_large.shape == (batch_size, 96)
    print("✓ FNNPlanesLarge test passed!")

    print("\nNote: Output is 96 values = 24 planes × 4 coords (xmin, xmax, ymin, ymax)")

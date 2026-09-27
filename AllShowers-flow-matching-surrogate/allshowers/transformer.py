import torch
import torch.nn as nn
from torch import Tensor

from allshowers.attention import (
    BLOCK_SIZE,
    AttentionMask,
    attend,
    block_sparse_available,
    compute_mask,
)

__all__ = ["FlexEncoderLayer", "Transformer", "compute_mask", "ATTENTION_BACKENDS"]

ATTENTION_BACKENDS = ("flex", "block_sparse")

# head dims the Block-Sparse-Attention CUDA kernels are compiled for
_BSA_HEAD_DIMS = (32, 64, 128)


class FlexEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        num_head: int = 4,
        dim_feedforward: int = 2048,
        activation: str | torch.nn.Module = "relu",
        dropout: float = 0.0,
        attn_backend: str = "flex",
        attn_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if dim_embedding % num_head != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by num_head ({num_head})."
            )
        super().__init__()

        self.num_head = num_head
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_head
        self.attn_backend = attn_backend
        self.attn_dtype = attn_dtype

        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation]()
        else:
            activation_module = activation
        del activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.key_query_value = nn.Linear(dim_embedding, dim_embedding * 3)
        self.feedforward = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            activation_module,
            nn.Linear(dim_feedforward, dim_embedding),
            self.dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def multihead_attention(
        self,
        x: Tensor,
        mask: AttentionMask,
    ) -> Tensor:
        key_query_value: Tensor = self.key_query_value(x)
        key_query_value = key_query_value.view(
            key_query_value.shape[0],  # batch
            key_query_value.shape[1],  # sequence
            self.num_head,
            3,
            self.dim_head,
        )

        if self.attn_backend == "flex":
            # -> [3, batch, head, sequence, dim_head]
            key_query_value = key_query_value.permute(3, 0, 2, 1, 4).contiguous()
            key, query, value = key_query_value
            x = attend(query, key, value, mask, backend="flex")
            x = x.permute(0, 2, 1, 3).contiguous()
        else:
            # -> [3, batch, sequence, head, dim_head]
            key_query_value = key_query_value.permute(3, 0, 1, 2, 4).contiguous()
            key, query, value = key_query_value
            x = attend(
                query,
                key,
                value,
                mask,
                backend=self.attn_backend,
                dtype=self.attn_dtype,
            )

        x = x.reshape(x.shape[0], x.shape[1], self.dim_embedding)
        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        mask: AttentionMask,
    ) -> Tensor:
        x = x + self.multihead_attention(self.layer_norm1(x), mask=mask)
        x = x + self.feedforward(self.layer_norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim_inputs: tuple[int, ...],
        dim_embedding: int,
        num_head: int,
        num_blocks: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        num_points_cond: int = 0,
        identity_init: bool = False,
        activation: str | torch.nn.Module = "relu",
        num_layer_cond: int = -1,
        num_particles: int = 1,
        dropout: float = 0.0,
        attn_backend: str = "flex",
        attn_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        if attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attn_backend must be one of {ATTENTION_BACKENDS}, got {attn_backend!r}"
            )
        dim_head = dim_embedding // num_head
        if attn_backend == "block_sparse":
            if not block_sparse_available():
                raise ImportError(
                    "attn_backend='block_sparse' but the block_sparse_attn package "
                    "could not be imported."
                )
            if dim_head not in _BSA_HEAD_DIMS:
                raise ValueError(
                    f"Block-Sparse-Attention only supports head dims {_BSA_HEAD_DIMS}, "
                    f"got dim_embedding/num_head = {dim_head}."
                )

        self.num_layer_cond = num_layer_cond
        self.num_head = num_head
        self.attn_backend = attn_backend
        self.attn_dtype = getattr(torch, attn_dtype)

        self.embedding = nn.Linear(dim_inputs[0], dim_embedding)
        self.layer_embedding = nn.Embedding(num_layers, dim_embedding)
        self.cond_embedding = nn.Linear(sum(dim_inputs[1:]), dim_embedding)
        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation.lower()]()
        else:
            activation_module = activation
        del activation
        if num_points_cond > 0:
            self.num_points_embedding = nn.Sequential(
                nn.Linear(num_layers, num_points_cond),
                activation_module,
                nn.Linear(num_points_cond, dim_embedding),
            )
        else:
            self.num_points_embedding = None
        if num_particles > 1:
            self.particle_embedding = nn.Embedding(num_particles, dim_embedding)
        else:
            self.particle_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                FlexEncoderLayer(
                    dim_embedding,
                    num_head,
                    dim_feedforward=dim_feedforward,
                    activation=activation_module,
                    dropout=dropout,
                    attn_backend=attn_backend,
                    attn_dtype=self.attn_dtype,
                )
                for _ in range(num_blocks)
            ]
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.head = nn.Linear(dim_embedding, dim_inputs[0])
        if identity_init:
            with torch.no_grad():
                self.head.weight.fill_(0.0)
                self.head.bias.fill_(0.0)

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor,
        num_points: Tensor,
        layer: Tensor,
        block_mask: AttentionMask,
        label: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        x += self.layer_embedding(layer.squeeze())
        cond = torch.cat([t, cond], dim=1)
        cond = self.cond_embedding(cond).unsqueeze(1)
        x += cond
        if label is not None and self.particle_embedding is not None:
            x += self.particle_embedding(label).unsqueeze(1)
        if self.num_points_embedding is not None:
            num_points = self.num_points_embedding(
                num_points.to(torch.get_default_dtype())
            )
            x += num_points.unsqueeze(1)
        for block in self.transformer_blocks:
            x = block(x, mask=block_mask)
        return self.head(x)
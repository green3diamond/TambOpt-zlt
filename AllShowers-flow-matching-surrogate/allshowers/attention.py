"""Pluggable attention backends.

Two backends are supported:

* ``"flex"``          -- ``torch.nn.attention.flex_attention`` (the original path).
* ``"block_sparse"``  -- the MIT-HAN-Lab Block-Sparse-Attention kernels
                         (https://github.com/mit-han-lab/Block-Sparse-Attention).

Both expose the same two entry points:

    mask = compute_mask(padding_mask, layer, num_layer_cond, backend, num_head)
    out  = attend(q, k, v, mask, backend)

IMPORTANT SEMANTIC DIFFERENCE
-----------------------------
``flex_attention`` evaluates ``mask_mod`` *element-wise* inside partial blocks, so
the layer window is applied exactly.  Block-Sparse-Attention decides per 128x128
block: a block is either fully computed or fully skipped.  We therefore include a
block whenever *any* (query, key) pair inside it is inside the layer window, which
makes the effective window a superset of the flex one (it is widened by up to one
block at the boundaries).  Padding, by contrast, is handled exactly, because the
block-sparse path removes padded tokens from the sequence (varlen / "unpad")
instead of masking them.

Consequence: a model trained with one backend is not bit-identical under the
other.  Treat the backend as a model hyper-parameter, not a runtime switch, unless
``num_layer_cond < 0`` (dense), where the two agree up to numerics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

__all__ = [
    "BLOCK_SIZE",
    "AttentionMask",
    "BlockSparseMask",
    "attend",
    "block_sparse_available",
    "compute_mask",
]

# Block-Sparse-Attention hardcodes m_block_dim = n_block_dim = 128.
BLOCK_SIZE = 128

# Sentinel used when reducing layer indices over empty block slots.  Large enough
# to never fall inside a window, small enough that int32 differences cannot
# overflow.
_SENTINEL = 1 << 20

_create_block_mask = torch.compile(create_block_mask)

try:  # optional dependency -- only needed for backend="block_sparse"
    from block_sparse_attn import block_sparse_attn_func

    _BSA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the CUDA build being present
    block_sparse_attn_func = None
    _BSA_AVAILABLE = False


def block_sparse_available() -> bool:
    """Whether the compiled ``block_sparse_attn`` extension can be imported."""
    return _BSA_AVAILABLE


@dataclass
class BlockSparseMask:
    """Everything ``block_sparse_attn_func`` needs, precomputed once per batch.

    Attributes:
        index: flat indices (into ``B * S``) of the non-padded tokens, in
            sequence-major order.  Used to unpad q/k/v and to scatter the output
            back into the padded layout.
        cu_seqlens: ``[B + 1]`` int32 prefix sum of the per-shower point counts.
        max_seqlen: longest unpadded sequence in the batch.
        head_mask_type: ``[num_head]`` int32.  All ones => every head uses the
            block mask.  All zeros => dense attention.
        streaming_info: ``[2 * num_head]`` int32, unused here but the kernel
            expects a real tensor.
        base_blockmask: ``[B, num_head, n_blocks, n_blocks]`` bool, or ``None``
            for the dense case.
        batch_size / sequence_length: shape of the padded layout.
    """

    index: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    head_mask_type: Tensor
    streaming_info: Tensor
    base_blockmask: Tensor | None
    batch_size: int
    sequence_length: int


AttentionMask = BlockMask | BlockSparseMask


# --------------------------------------------------------------------------- #
# mask construction
# --------------------------------------------------------------------------- #


def _window_bounds(num_layer_cond: int) -> tuple[int, int]:
    """Inclusive bounds on ``layer_q - layer_kv``; matches the flex mask_mod."""
    lower = -1 * (num_layer_cond + 1) // 2
    upper = num_layer_cond // 2
    return lower, upper


def compute_mask_flex(
    padding_mask: Tensor,
    layer: Tensor,
    num_layer_cond: int = -1,
) -> BlockMask:
    """Original flex_attention block mask (unchanged behaviour)."""
    padding_mask = padding_mask.flatten(1)
    layer = layer.flatten(1)
    if num_layer_cond < 0:

        def mask_fn(b, h, q_idx, kv_idx):
            return padding_mask[b, q_idx] & padding_mask[b, kv_idx]

    else:
        lower, upper = _window_bounds(num_layer_cond)

        def mask_fn(b, h, q_idx, kv_idx):
            lower_bound = layer[b, q_idx] - layer[b, kv_idx] >= lower
            upper_bound = layer[b, q_idx] - layer[b, kv_idx] <= upper
            not_padding = padding_mask[b, q_idx] & padding_mask[b, kv_idx]
            return lower_bound & upper_bound & not_padding

    sequence_length = padding_mask.shape[1]
    batch_size = padding_mask.shape[0]
    return _create_block_mask(
        mask_mod=mask_fn,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=str(padding_mask.device),
    )


@torch.no_grad()
@torch.compiler.disable
def compute_mask_block_sparse(
    padding_mask: Tensor,
    layer: Tensor,
    num_layer_cond: int = -1,
    num_head: int = 1,
) -> BlockSparseMask:
    """Build the varlen metadata + 128x128 block mask for Block-Sparse-Attention.

    Args:
        padding_mask: ``[B, S, 1]`` (or ``[B, S]``) bool, True for real points.
        layer: ``[B, S, 1]`` (or ``[B, S]``) integer layer index per point.
        num_layer_cond: same meaning as in the flex path; negative => dense.
        num_head: number of query heads.
    """
    padding_mask = padding_mask.flatten(1)
    layer = layer.flatten(1)
    device = padding_mask.device
    batch_size, sequence_length = padding_mask.shape

    # --- varlen bookkeeping -------------------------------------------------
    seqlens = padding_mask.sum(dim=1)
    max_seqlen = int(seqlens.max().item()) if batch_size > 0 else 0
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0).to(torch.int32)
    # Row-major nonzero => tokens are already grouped by sequence, which is
    # exactly the layout cu_seqlens describes.
    index = padding_mask.reshape(-1).nonzero(as_tuple=True)[0]

    streaming_info = torch.zeros(2 * num_head, dtype=torch.int32, device=device)

    if num_layer_cond < 0:
        # mask_type 0 => dense attention for every head.  Padding is already
        # handled by unpadding, so no block mask is needed at all.
        return BlockSparseMask(
            index=index,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            head_mask_type=torch.zeros(num_head, dtype=torch.int32, device=device),
            streaming_info=streaming_info,
            base_blockmask=None,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

    lower, upper = _window_bounds(num_layer_cond)
    num_blocks = (max_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_len = num_blocks * BLOCK_SIZE

    # --- per-block layer range, in the *unpadded* token ordering -------------
    # position of each real token inside its own sequence
    position = torch.cumsum(padding_mask.to(torch.int64), dim=1) - 1
    batch_index = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand_as(padding_mask)
    )
    layer_i32 = layer.to(torch.int32)

    packed_min = torch.full(
        (batch_size, padded_len), _SENTINEL, dtype=torch.int32, device=device
    )
    packed_max = torch.full(
        (batch_size, padded_len), -_SENTINEL, dtype=torch.int32, device=device
    )
    b_sel = batch_index[padding_mask]
    p_sel = position[padding_mask]
    l_sel = layer_i32[padding_mask]
    packed_min[b_sel, p_sel] = l_sel
    packed_max[b_sel, p_sel] = l_sel

    layer_min = packed_min.view(batch_size, num_blocks, BLOCK_SIZE).amin(dim=-1)
    layer_max = packed_max.view(batch_size, num_blocks, BLOCK_SIZE).amax(dim=-1)

    # A (query block i, key block j) pair is kept iff the achievable range of
    # (layer_q - layer_k) intersects [lower, upper].  Empty blocks keep the
    # sentinels and are dropped automatically.
    keep = (layer_min.unsqueeze(2) - layer_max.unsqueeze(1) <= upper) & (
        layer_max.unsqueeze(2) - layer_min.unsqueeze(1) >= lower
    )  # [B, n_blocks, n_blocks]

    base_blockmask = (
        keep.unsqueeze(1).expand(batch_size, num_head, num_blocks, num_blocks)
        .contiguous()
    )

    return BlockSparseMask(
        index=index,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        head_mask_type=torch.ones(num_head, dtype=torch.int32, device=device),
        streaming_info=streaming_info,
        base_blockmask=base_blockmask,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )


def compute_mask(
    padding_mask: Tensor,
    layer: Tensor,
    num_layer_cond: int = -1,
    backend: str = "flex",
    num_head: int = 1,
) -> AttentionMask:
    """Dispatch to the backend-specific mask builder."""
    if backend == "flex":
        return compute_mask_flex(padding_mask, layer, num_layer_cond)
    if backend == "block_sparse":
        if not _BSA_AVAILABLE:
            raise ImportError(
                "backend='block_sparse' requires the block_sparse_attn package. "
                "Install it from https://github.com/mit-han-lab/Block-Sparse-Attention"
            )
        return compute_mask_block_sparse(
            padding_mask, layer, num_layer_cond, num_head
        )
    raise ValueError(f"Unknown attention backend: {backend!r}")


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #


def flex_attend(query: Tensor, key: Tensor, value: Tensor, mask: BlockMask) -> Tensor:
    """q/k/v: ``[B, num_head, S, dim_head]`` -> ``[B, num_head, S, dim_head]``."""
    return flex_attention(query=query, key=key, value=value, block_mask=mask)  # type: ignore


@torch.compiler.disable
def block_sparse_attend(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: BlockSparseMask,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """q/k/v: ``[B, S, num_head, dim_head]`` -> ``[B, S, num_head, dim_head]``.

    The kernel is varlen and fp16/bf16 only, so this unpads, casts, calls the
    kernel, casts back and scatters into the padded layout.  Padded slots are
    left at zero.
    """
    if block_sparse_attn_func is None:  # pragma: no cover
        raise ImportError("block_sparse_attn is not installed")

    batch_size, sequence_length, num_head, dim_head = query.shape
    out_dtype = query.dtype

    flat_shape = (batch_size * sequence_length, num_head, dim_head)
    q_unpad = query.reshape(flat_shape).index_select(0, mask.index).to(dtype)
    k_unpad = key.reshape(flat_shape).index_select(0, mask.index).to(dtype)
    v_unpad = value.reshape(flat_shape).index_select(0, mask.index).to(dtype)

    out_unpad = block_sparse_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        mask.cu_seqlens,
        mask.cu_seqlens,
        mask.head_mask_type,
        mask.streaming_info,
        mask.base_blockmask,
        mask.max_seqlen,
        mask.max_seqlen,
        dropout_p,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )

    out = query.new_zeros(flat_shape)
    out = out.index_copy(0, mask.index, out_unpad.to(out_dtype))
    return out.view(batch_size, sequence_length, num_head, dim_head)


def attend(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: AttentionMask,
    backend: str = "flex",
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dispatch.  Note the two backends expect different q/k/v layouts:

    * flex          -> ``[B, num_head, S, dim_head]``
    * block_sparse  -> ``[B, S, num_head, dim_head]``
    """
    if backend == "flex":
        if not isinstance(mask, BlockMask):
            raise TypeError("backend='flex' requires a BlockMask")
        return flex_attend(query, key, value, mask)
    if backend == "block_sparse":
        if not isinstance(mask, BlockSparseMask):
            raise TypeError("backend='block_sparse' requires a BlockSparseMask")
        return block_sparse_attend(query, key, value, mask, dropout_p, dtype)
    raise ValueError(f"Unknown attention backend: {backend!r}")
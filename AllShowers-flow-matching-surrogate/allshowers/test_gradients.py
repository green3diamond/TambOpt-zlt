#!/usr/bin/env python3
"""Test BACKWARD correctness of the attention backends.

The earlier debug script only checked forward outputs, and skipped the windowed
case entirely (the two masks legitimately differ there, so there was no
reference).  That left the backward pass untested -- which is the prime suspect
for a model that trains correctly for ~18 epochs and then collapses to a null
predictor.

Strategy: build a naive fp32 masked attention that takes the SAME effective
mask each backend uses, so gradients are comparable even in the windowed case.

    python test_gradients.py
    python test_gradients.py --num-layer-cond 4 --dim-head 32 --iters 200

Reading the output:
  * forward matching but backward not  -> gradient plumbing bug (index_copy /
    index_select / kernel bwd)
  * both matching on one step but the
    accumulation test drifting          -> bf16 bias accumulating over steps
  * effective-mask row differing         -> block rounding changes the model
"""

import os
import argparse
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention as _flex
_flex_compiled = torch.compile(_flex)

# Resolve `allshowers` to this copy (TambOpt-zlt/AllShowers-flow-matching-surrogate), ahead of
# the editable install that still points at ~/AllShowers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allshowers.attention import (
    BLOCK_SIZE,
    BlockSparseMask,
    attend,
    block_sparse_available,
    compute_mask,
)

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bool_mask: torch.Tensor,
) -> torch.Tensor:
    """[B, S, H, D] with an explicit [B, S, S] boolean mask, all in fp32.

    bool_mask[b, q, k] is True where query q may attend to key k.
    """
    q = query.permute(0, 2, 1, 3).float()
    k = key.permute(0, 2, 1, 3).float()
    v = value.permute(0, 2, 1, 3).float()

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~bool_mask[:, None], float("-inf"))
    # Rows with no permitted keys would give NaN; zero them instead.
    all_masked = ~bool_mask.any(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.matmul(weights, v)
    out = out.permute(0, 2, 1, 3)  # [B, S, H, D]
    # all_masked is [B, S]; broadcast over heads and dim_head
    return out.masked_fill(all_masked[:, :, None, None], 0.0)

def flex_effective_mask(
    layer: torch.Tensor, mask: torch.Tensor, num_layer_cond: int
) -> torch.Tensor:
    """Exact element-wise mask that flex_attention applies -> [B, S, S] bool."""
    layer_flat = layer.flatten(1)
    mask_flat = mask.flatten(1)
    not_padding = mask_flat[:, :, None] & mask_flat[:, None, :]
    if num_layer_cond < 0:
        return not_padding
    lower = -1 * (num_layer_cond + 1) // 2
    upper = num_layer_cond // 2
    delta = layer_flat[:, :, None] - layer_flat[:, None, :]
    return (delta >= lower) & (delta <= upper) & not_padding


def block_sparse_effective_mask(
    layer: torch.Tensor, mask: torch.Tensor, num_layer_cond: int, num_head: int
) -> torch.Tensor:
    """Mask block-sparse actually applies, expanded back to the padded grid.

    Block granularity means this is a SUPERSET of the flex mask: a whole
    128x128 block is kept if any pair inside it is in-window.
    """
    attn_mask = compute_mask(mask, layer, num_layer_cond, "block_sparse", num_head)
    assert isinstance(attn_mask, BlockSparseMask)

    batch_size, sequence_length = mask.shape[0], mask.shape[1]
    mask_flat = mask.flatten(1)
    not_padding = mask_flat[:, :, None] & mask_flat[:, None, :]

    if attn_mask.base_blockmask is None:
        return not_padding

    # Map padded token index -> position within its unpadded sequence -> block.
    position = torch.cumsum(mask_flat.to(torch.int64), dim=1) - 1
    block_index = (position // BLOCK_SIZE).clamp(min=0)

    per_batch = attn_mask.base_blockmask[:, 0]  # [B, n_blk, n_blk]
    batch_arange = torch.arange(batch_size, device=mask.device)[:, None, None]
    q_block = block_index[:, :, None].expand(-1, -1, sequence_length)
    k_block = block_index[:, None, :].expand(-1, sequence_length, -1)
    keep = per_batch[batch_arange, q_block, k_block]
    return keep & not_padding


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def make_batch(
    batch_size: int,
    max_points: int,
    num_layers: int,
    num_head: int,
    dim_head: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer-sorted ragged showers.  Returns (layer, mask, qkv)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    depth = torch.arange(num_layers, dtype=torch.float64)
    profile = (depth + 0.5) ** 2 * torch.exp(-(depth + 0.5) / (num_layers / 8))
    profile /= profile.sum()

    layer = torch.zeros(batch_size, max_points, 1, dtype=torch.int64)
    mask = torch.zeros(batch_size, max_points, 1, dtype=torch.bool)
    for b in range(batch_size):
        scale = 0.6 + 0.8 * torch.rand(1, generator=generator).item()
        total = min(int(max_points * 0.8 * scale), max_points)
        counts = torch.round(profile * total).to(torch.int64).clamp(min=0)
        values = torch.repeat_interleave(
            torch.arange(num_layers, dtype=torch.int64), counts
        )[:max_points]
        layer[b, : len(values), 0] = values
        mask[b, : len(values), 0] = True

    torch.manual_seed(seed)
    qkv = torch.randn(
        3, batch_size, max_points, num_head, dim_head, device=DEVICE
    )
    return layer.to(DEVICE), mask.to(DEVICE), qkv


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    scale = reference.pow(2).mean().sqrt().item()
    if scale == 0:
        return float("nan")
    return (reference - candidate).abs().mean().item() / scale


def report(name: str, value: float, tolerance: float) -> bool:
    ok = value < tolerance
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} rel {value:.3e}")
    return ok


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


def test_backend(
    backend: str,
    layer: torch.Tensor,
    mask: torch.Tensor,
    qkv: torch.Tensor,
    num_layer_cond: int,
    num_head: int,
    dtype: torch.dtype,
) -> None:
    """Compare forward AND backward against a naive fp32 reference that uses
    the same effective mask this backend applies."""
    print(f"\n--- {backend} ---")

    if backend == "flex":
        bool_mask = flex_effective_mask(layer, mask, num_layer_cond)
    else:
        bool_mask = block_sparse_effective_mask(layer, mask, num_layer_cond, num_head)

    density = bool_mask.float().mean().item()
    print(f"  effective mask density: {100 * density:.2f}%")

    attn_mask = compute_mask(mask, layer, num_layer_cond, backend, num_head)
    keep = mask.squeeze(-1)

    # --- candidate ---------------------------------------------------------
    candidate_qkv = qkv.clone().requires_grad_(True)
    query, key, value = candidate_qkv
    if backend == "flex":
        out = attend(
            query.permute(0, 2, 1, 3).to(dtype),
            key.permute(0, 2, 1, 3).to(dtype),
            value.permute(0, 2, 1, 3).to(dtype),
            attn_mask,
            backend="flex",
        ).permute(0, 2, 1, 3).float()
    else:
        out = attend(query, key, value, attn_mask, backend=backend, dtype=dtype)

    # A mask-weighted scalar objective, so padded slots contribute nothing --
    # this mirrors how the real loss is reduced.
    weights = torch.randn_like(out)
    objective = (out * weights * keep[..., None, None]).sum()
    objective.backward()
    candidate_grad = candidate_qkv.grad.clone()

    # --- reference ---------------------------------------------------------
    reference_qkv = qkv.clone().requires_grad_(True)
    q_ref, k_ref, v_ref = reference_qkv
    out_ref = naive_attention(q_ref, k_ref, v_ref, bool_mask)
    objective_ref = (out_ref * weights * keep[..., None, None]).sum()
    objective_ref.backward()
    reference_grad = reference_qkv.grad.clone()

    # --- compare -----------------------------------------------------------
    tolerance = 5e-2 if dtype != torch.float32 else 1e-4
    report("forward", relative_error(out_ref[keep], out[keep]), tolerance)

    names = ["dQ", "dK", "dV"]
    for i, name in enumerate(names):
        error = relative_error(reference_grad[i][keep], candidate_grad[i][keep])
        report(f"backward {name}", error, tolerance)

    # Gradient that leaks into padded slots is a scatter bug: those positions
    # do not participate in the objective at all.
    padded_grad = candidate_grad[:, ~keep]
    if padded_grad.numel():
        magnitude = padded_grad.abs().max().item()
        ok = magnitude < 1e-6
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {'no gradient in padded slots':46s} "
            f"max {magnitude:.3e}"
        )


def test_accumulation(
    layer: torch.Tensor,
    mask: torch.Tensor,
    qkv: torch.Tensor,
    num_layer_cond: int,
    num_head: int,
    dtype: torch.dtype,
    iterations: int,
) -> None:
    """Does bf16 gradient bias accumulate over many steps?

    Sums the gradient over `iterations` different objectives and compares the
    running total against fp32.  A per-step error that is random cancels; a
    systematic bias grows.
    """
    print(f"\n--- accumulation over {iterations} steps (block_sparse) ---")
    if not block_sparse_available():
        print("  skipped: block_sparse not installed")
        return

    bool_mask = block_sparse_effective_mask(layer, mask, num_layer_cond, num_head)
    attn_mask = compute_mask(mask, layer, num_layer_cond, "block_sparse", num_head)
    keep = mask.squeeze(-1)

    total_candidate = torch.zeros_like(qkv)
    total_reference = torch.zeros_like(qkv)

    for step in range(iterations):
        torch.manual_seed(10_000 + step)
        weights = torch.randn(
            qkv.shape[1], qkv.shape[2], qkv.shape[3], qkv.shape[4], device=DEVICE
        )

        candidate_qkv = qkv.clone().requires_grad_(True)
        query, key, value = candidate_qkv
        out = attend(query, key, value, attn_mask, backend="block_sparse", dtype=dtype)
        (out * weights * keep[..., None, None]).sum().backward()
        total_candidate += candidate_qkv.grad

        reference_qkv = qkv.clone().requires_grad_(True)
        q_ref, k_ref, v_ref = reference_qkv
        out_ref = naive_attention(q_ref, k_ref, v_ref, bool_mask)
        (out_ref * weights * keep[..., None, None]).sum().backward()
        total_reference += reference_qkv.grad

        if (step + 1) % max(iterations // 4, 1) == 0:
            error = relative_error(total_reference, total_candidate)
            print(f"  after {step + 1:4d} steps: rel {error:.3e}")

    print(
        "  -> if this GROWS with steps, bf16 bias is systematic and will "
        "corrupt training;\n     if it SHRINKS or is flat, per-step error is "
        "random noise and harmless."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-layer-cond", type=int, default=4)
    parser.add_argument("--num-head", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32, choices=[32, 64, 128])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    dtype = getattr(torch, args.dtype)
    torch.set_float32_matmul_precision("high")

    layer, mask, qkv = make_batch(
        args.batch_size,
        args.max_points,
        args.num_layers,
        args.num_head,
        args.dim_head,
    )

    print(f"gpu            : {torch.cuda.get_device_name(0)}")
    print(f"batch          : {tuple(mask.shape[:2])}")
    print(f"heads x dim    : {args.num_head} x {args.dim_head}")
    print(f"num_layer_cond : {args.num_layer_cond}")
    print(f"dtype          : {args.dtype}")
    print(f"block-sparse   : {'available' if block_sparse_available() else 'MISSING'}")

    for num_layer_cond in (-1, args.num_layer_cond):
        label = "dense" if num_layer_cond < 0 else f"window +/-{num_layer_cond // 2}"
        print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
        test_backend(
            "flex", layer, mask, qkv, num_layer_cond, args.num_head, dtype
        )
        if block_sparse_available():
            test_backend(
                "block_sparse", layer, mask, qkv, num_layer_cond, args.num_head, dtype
            )

    if args.iters > 0:
        test_accumulation(
            layer,
            mask,
            qkv,
            args.num_layer_cond,
            args.num_head,
            dtype,
            args.iters,
        )


if __name__ == "__main__":
    main()
"""Masked-path parity test for the block_sparse backend.

The dense test (test_backend_parity.py) cleared the kernel's arithmetic.  This
one exercises the part it could not: base_blockmask.

Two independent checks:

  CHECK 1 -- does the kernel apply the mask it was given?
      Builds an fp64 reference using the SAME base_blockmask that
      compute_mask_block_sparse produced, expanded to element level.  If the
      kernel honours the mask, agreement should look exactly like the dense
      case (~2e-4 mean_abs in bf16, unbiased).  A large error here means the
      kernel is reading the mask differently than we are writing it -- e.g.
      query/key axes transposed.

  CHECK 2 -- is the block mask the window we think it is?
      Compares the block-rounded mask against the exact element-wise flex
      window.  Reports how many extra pairs the block mask admits, and -- more
      importantly -- whether it DROPS any pair flex would keep.  Dropping is a
      bug; admitting extra is the documented widening.

Note _window_bounds is asymmetric: for num_layer_cond=4, Python floors
-1 * 5 // 2 to -3, giving lower=-3, upper=2.  So a transposed blockmask does
not cancel out -- it shifts the window.  CHECK 1 catches that.

Usage:
    python test_masked_parity.py --num-layer-cond 4
    python test_masked_parity.py --num-layer-cond 8 --dim-head 32
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Resolve `allshowers` to this copy (TambOpt-zlt/AllShowers-flow-matching-surrogate), ahead of
# the editable install that still points at ~/AllShowers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allshowers.attention import (
    BLOCK_SIZE,
    attend,
    block_sparse_available,
    compute_mask,
    compute_mask_block_sparse,
    _window_bounds,
)


def make_shower_batch(batch_size, sequence_length, num_layers, device, generator):
    """Layer indices that look like a real shower: sorted, uneven occupancy."""
    seqlens = torch.randint(
        sequence_length // 2, sequence_length, (batch_size,),
        device=device, generator=generator,
    )
    position = torch.arange(sequence_length, device=device)[None, :]
    padding_mask = position < seqlens[:, None]

    # Uneven per-layer counts, then sorted so layers are contiguous in sequence
    # order -- which is what the real data loader produces.
    weights = torch.rand(batch_size, num_layers, device=device, generator=generator)
    weights = weights / weights.sum(dim=1, keepdim=True)
    layer = torch.zeros(
        batch_size, sequence_length, dtype=torch.int64, device=device
    )
    for b in range(batch_size):
        counts = (weights[b] * int(seqlens[b])).long()
        counts[-1] += int(seqlens[b]) - int(counts.sum())
        idx = torch.repeat_interleave(
            torch.arange(num_layers, device=device), counts
        )
        layer[b, : idx.numel()] = idx
    layer = layer * padding_mask
    return padding_mask, layer


def elementwise_from_blockmask(mask, padding_mask):
    """Expand base_blockmask [B,H,nb,nb] to an element mask [B,S,S] (padded coords).

    Uses the same within-sequence position that compute_mask_block_sparse used,
    so this is exactly the mask the kernel was handed.
    """
    batch_size, sequence_length = padding_mask.shape
    position = torch.cumsum(padding_mask.to(torch.int64), dim=1) - 1
    block_of = (position // BLOCK_SIZE).clamp(min=0)  # [B, S]

    keep = mask.base_blockmask[:, 0]  # [B, nb, nb]; all heads identical here
    num_blocks = keep.shape[-1]
    qb = block_of.clamp(max=num_blocks - 1)
    element = keep[
        torch.arange(batch_size, device=keep.device)[:, None, None],
        qb[:, :, None],
        qb[:, None, :],
    ]
    valid = padding_mask[:, :, None] & padding_mask[:, None, :]
    return element & valid


def exact_window_mask(padding_mask, layer, num_layer_cond):
    """The element-wise window flex applies, for comparison."""
    lower, upper = _window_bounds(num_layer_cond)
    delta = layer[:, :, None] - layer[:, None, :]
    inside = (delta >= lower) & (delta <= upper)
    valid = padding_mask[:, :, None] & padding_mask[:, None, :]
    return inside & valid


def reference_attention(query, key, value, element_mask):
    """fp64 attention under an explicit [B,S,S] element mask.  q/k/v: [B,H,S,D]."""
    q, k, v = query.double(), key.double(), value.double()
    scale = 1.0 / (q.shape[-1] ** 0.5)
    logits = (q @ k.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~element_mask[:, None], float("-inf"))
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    return weights @ v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-head", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32, choices=[32, 64, 128])
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-layer-cond", type=int, default=4)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not block_sparse_available():
        raise SystemExit("block_sparse_attn is not importable in this environment")

    device = torch.device(args.device)
    attn_dtype = getattr(torch, args.dtype)
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    lower, upper = _window_bounds(args.num_layer_cond)
    print(
        f"\nmasked parity, num_layer_cond={args.num_layer_cond} "
        f"(window on layer_q - layer_kv: [{lower}, {upper}]), "
        f"D={args.dim_head} H={args.num_head} S={args.seq_len}\n"
    )

    stats = {n: {"signed": [], "abs": [], "cos": []} for n in ["out", "dq", "dk", "dv"]}
    extra_frac, dropped_total = [], 0

    for _ in range(args.trials):
        padding_mask, layer = make_shower_batch(
            args.batch_size, args.seq_len, args.num_layers, device, generator
        )
        shape = (args.batch_size, args.num_head, args.seq_len, args.dim_head)
        q = torch.randn(shape, device=device, generator=generator)
        k = torch.randn(shape, device=device, generator=generator)
        v = torch.randn(shape, device=device, generator=generator)
        real = padding_mask[:, None, :, None]
        grad_out = torch.randn_like(q) * real

        mask = compute_mask_block_sparse(
            padding_mask[:, :, None], layer[:, :, None],
            args.num_layer_cond, args.num_head,
        )

        # ---- CHECK 2: block mask vs exact flex window ----
        block_element = elementwise_from_blockmask(mask, padding_mask)
        exact = exact_window_mask(padding_mask, layer, args.num_layer_cond)
        dropped = (exact & ~block_element).sum().item()
        dropped_total += dropped
        extra_frac.append(
            (block_element & ~exact).sum().item() / max(exact.sum().item(), 1)
        )

        # ---- CHECK 1: kernel output vs fp64 under the SAME block mask ----
        qd = q.double().requires_grad_(True)
        kd = k.double().requires_grad_(True)
        vd = v.double().requires_grad_(True)
        ref_out = reference_attention(qd, kd, vd, block_element)
        ref_out.backward(grad_out.double())
        ref = {"out": ref_out.detach(), "dq": qd.grad, "dk": kd.grad, "dv": vd.grad}

        qi = q.transpose(1, 2).contiguous().requires_grad_(True)
        ki = k.transpose(1, 2).contiguous().requires_grad_(True)
        vi = v.transpose(1, 2).contiguous().requires_grad_(True)
        out = attend(qi, ki, vi, mask, backend="block_sparse", dtype=attn_dtype)
        out.backward(grad_out.transpose(1, 2).contiguous())
        got = {
            "out": out.detach().transpose(1, 2),
            "dq": qi.grad.transpose(1, 2),
            "dk": ki.grad.transpose(1, 2),
            "dv": vi.grad.transpose(1, 2),
        }

        for name in stats:
            a = (got[name].double() * real)
            b = (ref[name].double() * real)
            err = (a - b)[real.expand_as(a)]
            stats[name]["signed"].append(err.mean())
            stats[name]["abs"].append(err.abs().mean())
            stats[name]["cos"].append(
                torch.nn.functional.cosine_similarity(
                    a.flatten(), b.flatten(), dim=0
                )
            )

    print("CHECK 1 -- kernel output vs fp64 under the kernel's own block mask")
    print("  (compare these to the dense run: ~2e-4 abs, cos ~0.99999)")
    for name, s in stats.items():
        abs_err = torch.stack(s["abs"]).mean().item()
        signed = torch.stack(s["signed"]).mean().item()
        cos = torch.stack(s["cos"]).mean().item()
        flag = "" if cos > 0.999 else "   <<< MASK NOT APPLIED AS WRITTEN"
        print(
            f"  {name:<4s} mean_signed={signed:+.3e}  mean_abs={abs_err:.3e}  "
            f"cos={cos:.6f}{flag}"
        )

    print("\nCHECK 2 -- block mask vs exact flex window")
    print(
        f"  extra pairs admitted: {100 * sum(extra_frac) / len(extra_frac):.1f}% "
        f"of the exact window (expected: documented widening)"
    )
    print(
        f"  pairs DROPPED that flex keeps: {dropped_total}  "
        f"({'OK' if dropped_total == 0 else 'BUG -- mask is not a superset'})"
    )
    print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Compare flex_attention against Block-Sparse-Attention on shower-like inputs.

Both backends are run in the SAME dtype so the comparison is fair -- an earlier
version of this script ran flex in fp32 and block-sparse in bf16, which made the
"speedup" mostly a dtype artifact and made the error check meaningless.

    python bench_attention.py --max-points 4096  --num-layers 24 --batch-size 64
    python bench_attention.py --max-points 25088 --num-layers 24 --batch-size 8

Add --config /path/to/conf.yaml to use a real batch from your data loader instead
of the synthetic profile.  Add --fp32-flex to additionally time flex in fp32,
which is what training currently does.
"""

import os
import argparse
import sys

import torch

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


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def synthetic_batch(
    batch_size: int,
    max_points: int,
    num_layers: int,
    device: torch.device,
    fill: float = 0.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer-sorted showers with a gamma-ish longitudinal profile.

    Returns (layer [B, S, 1] int64, mask [B, S, 1] bool).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    depth = torch.arange(num_layers, dtype=torch.float64)
    profile = (depth + 0.5) ** 2 * torch.exp(-(depth + 0.5) / (num_layers / 8))
    profile /= profile.sum()

    layer = torch.zeros(batch_size, max_points, 1, dtype=torch.int64)
    mask = torch.zeros(batch_size, max_points, 1, dtype=torch.bool)

    for b in range(batch_size):
        scale = 0.6 + 0.8 * torch.rand(1, generator=generator).item()
        total = min(int(max_points * fill * scale), max_points)
        counts = torch.round(profile * total).to(torch.int64).clamp(min=0)
        values = torch.repeat_interleave(
            torch.arange(num_layers, dtype=torch.int64), counts
        )[:max_points]
        layer[b, : len(values), 0] = values
        mask[b, : len(values), 0] = True

    return layer.to(device), mask.to(device)


def real_batch(
    config_path: str, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    import yaml

    from allshowers import data_sets

    with open(config_path) as f:
        conf = yaml.safe_load(f)
    conf["data"]["val_len"] = batch_size
    if "num_layers" in conf["model"]:
        conf["data"]["num_layers"] = conf["model"]["num_layers"]
    _, val_loader, _ = data_sets.get_data_loaders(conf["data"], batch_size)
    batch = next(iter(val_loader))
    return batch["layer"].to(device), batch["mask"].to(device)


# --------------------------------------------------------------------------- #
# mask statistics
# --------------------------------------------------------------------------- #


def mask_stats(attn_mask, mask: torch.Tensor, backend: str) -> dict:
    """Report kept blocks both as a fraction and as an absolute count.

    The fractions are NOT directly comparable between backends: flex works on the
    padded [S, S] grid while block-sparse works on the shorter unpadded grid.  The
    absolute block count is the comparable quantity, so report both.
    """
    if backend == "block_sparse":
        assert isinstance(attn_mask, BlockSparseMask)
        if attn_mask.base_blockmask is None:
            num_blocks = (attn_mask.max_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
            fraction = 1.0
            kept = float(attn_mask.batch_size * num_blocks * num_blocks)
        else:
            per_head = attn_mask.base_blockmask[:, 0]  # [B, n_blk, n_blk]
            fraction = per_head.float().mean().item()
            kept = float(per_head.sum().item())
        grid = "unpadded"
    else:
        fraction = 1.0 - attn_mask.sparsity() / 100.0
        batch_size, sequence_length = mask.shape[0], mask.shape[1]
        num_blocks = (sequence_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        kept = fraction * batch_size * num_blocks * num_blocks
        grid = "padded"

    return {"fraction": fraction, "kept": kept, "grid": grid}


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def time_call(fn, iterations: int = 20, warmup: int = 5) -> float:
    """Median wall time per call, in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def run_backend(
    backend: str,
    layer: torch.Tensor,
    mask: torch.Tensor,
    num_layer_cond: int,
    num_head: int,
    dim_head: int,
    dtype: torch.dtype,
    flex_attention_compiled,
) -> dict:
    """Time one backend.

    q/k/v are generated in fp32 and cast to ``dtype`` inside the timed region for
    BOTH backends, so the comparison is apples to apples.  Pass
    ``dtype=torch.float32`` to time the fp32 path.
    """
    batch_size, sequence_length = mask.shape[0], mask.shape[1]
    device = mask.device

    attn_mask = compute_mask(
        padding_mask=mask,
        layer=layer,
        num_layer_cond=num_layer_cond,
        backend=backend,
        num_head=num_head,
    )
    stats = mask_stats(attn_mask, mask, backend)

    qkv_shape = (batch_size, sequence_length, num_head, dim_head)
    torch.manual_seed(1234)          # same inputs for every backend
    qkv = torch.randn(3, *qkv_shape, device=device, dtype=torch.float32)
    qkv.requires_grad_(True)

    def forward():
        query, key, value = qkv
        if backend == "flex":
            # [B, S, H, D] -> [B, H, S, D], cast to match the block-sparse path
            query = query.permute(0, 2, 1, 3).to(dtype)
            key = key.permute(0, 2, 1, 3).to(dtype)
            value = value.permute(0, 2, 1, 3).to(dtype)
            out = flex_attention_compiled(
                query=query, key=key, value=value, block_mask=attn_mask
            )
            return out.permute(0, 2, 1, 3).to(torch.float32)
        return attend(query, key, value, attn_mask, backend=backend, dtype=dtype)

    def forward_backward():
        out = forward()
        out.sum().backward()
        qkv.grad = None

    fwd_ms = time_call(forward)
    fwd_bwd_ms = time_call(forward_backward)

    with torch.no_grad():
        reference = forward()

    return {
        **stats,
        "fwd_ms": fwd_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "out": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="conf.yaml for a real batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-layer-cond", type=int, default=4)
    parser.add_argument("--num-head", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--fp32-flex",
        action="store_true",
        help="also time flex in fp32 (what training does today), for reference",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.set_float32_matmul_precision("high")

    if args.config:
        layer, mask = real_batch(args.config, args.batch_size, device)
    else:
        layer, mask = synthetic_batch(
            args.batch_size, args.max_points, args.num_layers, device
        )

    occupancy = mask.float().mean().item()
    print(f"gpu            : {torch.cuda.get_device_name(0)}")
    print(f"batch          : {tuple(mask.shape[:2])}")
    print(f"occupancy      : {100 * occupancy:.1f}% of padded slots are real points")
    print(f"num_layer_cond : {args.num_layer_cond}")
    print(f"heads x dim    : {args.num_head} x {args.dim_head}")
    print(f"compare dtype  : {args.dtype} (both backends)")
    print(
        f"block-sparse   : "
        f"{'available' if block_sparse_available() else 'NOT INSTALLED'}"
    )
    print()
    print(
        "note: kept-block % uses different grids per backend (flex = padded,\n"
        "      block-sparse = unpadded); compare absolute block counts instead."
    )
    print()

    from torch.nn.attention.flex_attention import flex_attention

    flex_compiled = torch.compile(flex_attention)

    backends = ["flex"] + (["block_sparse"] if block_sparse_available() else [])

    for num_layer_cond in (-1, args.num_layer_cond):
        label = "dense" if num_layer_cond < 0 else f"window +/-{num_layer_cond // 2}"
        print(f"--- {label} ---")
        results = {}
        for backend in backends:
            r = run_backend(
                backend,
                layer,
                mask,
                num_layer_cond,
                args.num_head,
                args.dim_head,
                dtype,
                flex_compiled,
            )
            results[backend] = r
            print(
                f"  {backend:13s} "
                f"kept {100 * r['fraction']:5.1f}% of {r['grid']:8s} grid "
                f"({r['kept']:9.0f} blocks)  "
                f"fwd {r['fwd_ms']:7.2f} ms   fwd+bwd {r['fwd_bwd_ms']:7.2f} ms"
            )

        if args.fp32_flex:
            r = run_backend(
                "flex",
                layer,
                mask,
                num_layer_cond,
                args.num_head,
                args.dim_head,
                torch.float32,
                flex_compiled,
            )
            print(
                f"  {'flex (fp32)':13s} "
                f"{'':36s}  "
                f"fwd {r['fwd_ms']:7.2f} ms   fwd+bwd {r['fwd_bwd_ms']:7.2f} ms"
            )

        if len(backends) == 2:
            flex_r, bsa_r = results["flex"], results["block_sparse"]
            print(
                f"  block ratio: {bsa_r['kept'] / max(flex_r['kept'], 1):.2f}x "
                f"the blocks flex computes"
            )
            print(
                f"  speedup: fwd {flex_r['fwd_ms'] / bsa_r['fwd_ms']:.2f}x   "
                f"fwd+bwd {flex_r['fwd_bwd_ms'] / bsa_r['fwd_bwd_ms']:.2f}x"
            )
            print(
                f"  bwd/fwd ratio: flex {flex_r['fwd_bwd_ms'] / flex_r['fwd_ms']:.1f}x"
                f"   block_sparse {bsa_r['fwd_bwd_ms'] / bsa_r['fwd_ms']:.1f}x"
            )

            if num_layer_cond < 0:
                # Only meaningful in the dense case: with a window the two masks
                # genuinely differ (block rounding), so a mismatch is expected.
                keep = mask.squeeze(-1)
                flex_out = flex_r["out"][keep]
                bsa_out = bsa_r["out"][keep]
                difference = (flex_out - bsa_out).abs()
                scale = flex_out.pow(2).mean().sqrt().item()
                print(
                    f"  |flex - block_sparse| / rms: "
                    f"max {difference.max().item() / scale:.2e}  "
                    f"mean {difference.mean().item() / scale:.2e}  "
                    f"(output rms {scale:.3e})"
                )
                print(
                    "    -> mean should be ~1e-2 in bf16; much larger means the "
                    "unpad/layout plumbing is wrong."
                )
        print()


if __name__ == "__main__":
    main()
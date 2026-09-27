#!/usr/bin/env python3
"""Isolate the block-sparse plumbing bug by bisecting from the simplest case up.

Each test adds one layer of complexity.  The first FAIL tells you where the bug
is.  Run:  python debug_block_sparse.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# Resolve `allshowers` to this copy (TambOpt-zlt/AllShowers-flow-matching-surrogate), ahead of
# the editable install that still points at ~/AllShowers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allshowers.attention import BlockSparseMask, block_sparse_attend, compute_mask

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def report(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    difference = (reference - candidate).abs()
    scale = reference.pow(2).mean().sqrt().item()
    relative = difference.mean().item() / scale
    ok = relative < 5e-2
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:52s} "
        f"rel {relative:.2e}  (rms {scale:.3e})"
    )
    return ok


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B, S, H, D] reference in fp32.  mask is [B, S] bool over keys."""
    q = query.permute(0, 2, 1, 3).float()
    k = key.permute(0, 2, 1, 3).float()
    v = value.permute(0, 2, 1, 3).float()
    attn_mask = None
    if mask is not None:
        attn_mask = mask[:, None, None, :]  # [B, 1, 1, S]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.permute(0, 2, 1, 3)


def test_1_raw_kernel_no_padding() -> bool:
    """Call the kernel directly, one full sequence, nothing to unpad."""
    from block_sparse_attn import block_sparse_attn_func

    torch.manual_seed(0)
    seqlen, num_head, dim_head = 256, 4, 64
    q, k, v = (
        torch.randn(seqlen, num_head, dim_head, device=DEVICE, dtype=DTYPE)
        for _ in range(3)
    )
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    hmt = torch.zeros(num_head, dtype=torch.int32, device=DEVICE)
    si = torch.zeros(2 * num_head, dtype=torch.int32, device=DEVICE)

    out = block_sparse_attn_func(
        q, k, v, cu, cu, hmt, si, None, seqlen, seqlen, 0.0
    )
    reference = naive_attention(
        q.unsqueeze(0).float(), k.unsqueeze(0).float(), v.unsqueeze(0).float()
    )[0]
    return report("raw kernel, 1 seq, no padding", reference, out.float())


def test_2_raw_kernel_two_seqs() -> bool:
    """Two equal-length sequences packed together -- exercises cu_seqlens."""
    from block_sparse_attn import block_sparse_attn_func

    torch.manual_seed(0)
    seqlen, num_head, dim_head = 256, 4, 64
    q, k, v = (
        torch.randn(2, seqlen, num_head, dim_head, device=DEVICE, dtype=DTYPE)
        for _ in range(3)
    )
    cu = torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device=DEVICE)
    hmt = torch.zeros(num_head, dtype=torch.int32, device=DEVICE)
    si = torch.zeros(2 * num_head, dtype=torch.int32, device=DEVICE)

    out = block_sparse_attn_func(
        q.reshape(-1, num_head, dim_head),
        k.reshape(-1, num_head, dim_head),
        v.reshape(-1, num_head, dim_head),
        cu, cu, hmt, si, None, seqlen, seqlen, 0.0,
    )
    out = out.view(2, seqlen, num_head, dim_head)
    reference = naive_attention(q.float(), k.float(), v.float())
    return report("raw kernel, 2 seqs, cu_seqlens", reference, out.float())


def test_3_attend_no_padding() -> bool:
    """Our attend() wrapper, but every token is real -- isolates the wrapper."""
    torch.manual_seed(0)
    batch, seqlen, num_head, dim_head = 4, 256, 4, 64
    q, k, v = (
        torch.randn(batch, seqlen, num_head, dim_head, device=DEVICE) for _ in range(3)
    )
    layer = torch.zeros(batch, seqlen, 1, dtype=torch.int64, device=DEVICE)
    mask = torch.ones(batch, seqlen, 1, dtype=torch.bool, device=DEVICE)

    attn_mask = compute_mask(mask, layer, -1, "block_sparse", num_head)
    out = block_sparse_attend(q, k, v, attn_mask, dtype=DTYPE)
    reference = naive_attention(q, k, v)
    return report("attend(), no padding", reference, out)


def test_4_attend_with_padding() -> bool:
    """Our attend() with ragged sequences -- exercises unpad + scatter."""
    torch.manual_seed(0)
    batch, seqlen, num_head, dim_head = 4, 512, 4, 64
    q, k, v = (
        torch.randn(batch, seqlen, num_head, dim_head, device=DEVICE) for _ in range(3)
    )
    lengths = [512, 400, 300, 137]
    mask = torch.zeros(batch, seqlen, 1, dtype=torch.bool, device=DEVICE)
    for b, length in enumerate(lengths):
        mask[b, :length, 0] = True
    layer = torch.zeros(batch, seqlen, 1, dtype=torch.int64, device=DEVICE)

    attn_mask = compute_mask(mask, layer, -1, "block_sparse", num_head)
    out = block_sparse_attend(q, k, v, attn_mask, dtype=DTYPE)
    reference = naive_attention(q, k, v, mask=mask.squeeze(-1))

    keep = mask.squeeze(-1)
    return report("attend(), ragged padding", reference[keep], out[keep])


def test_5_index_roundtrip() -> bool:
    """Does the unpad index + scatter alone round-trip the identity?"""
    torch.manual_seed(0)
    batch, seqlen, num_head, dim_head = 4, 512, 4, 64
    x = torch.randn(batch, seqlen, num_head, dim_head, device=DEVICE)
    lengths = [512, 400, 300, 137]
    mask = torch.zeros(batch, seqlen, 1, dtype=torch.bool, device=DEVICE)
    for b, length in enumerate(lengths):
        mask[b, :length, 0] = True
    layer = torch.zeros(batch, seqlen, 1, dtype=torch.int64, device=DEVICE)

    attn_mask = compute_mask(mask, layer, -1, "block_sparse", num_head)
    assert isinstance(attn_mask, BlockSparseMask)

    flat = (batch * seqlen, num_head, dim_head)
    packed = x.reshape(flat).index_select(0, attn_mask.index)
    restored = x.new_zeros(flat).index_copy(0, attn_mask.index, packed)
    restored = restored.view(batch, seqlen, num_head, dim_head)

    keep = mask.squeeze(-1)
    print(
        f"  cu_seqlens = {attn_mask.cu_seqlens.tolist()}  "
        f"(expected prefix sum of {lengths})"
    )
    print(f"  max_seqlen = {attn_mask.max_seqlen}  (expected {max(lengths)})")
    print(f"  index.numel = {attn_mask.index.numel()}  (expected {sum(lengths)})")
    return report("unpad/scatter round-trip (identity)", x[keep], restored[keep])


def main() -> None:
    print("Bisecting block-sparse plumbing.  First FAIL localises the bug.\n")
    tests = [
        test_1_raw_kernel_no_padding,
        test_2_raw_kernel_two_seqs,
        test_5_index_roundtrip,
        test_3_attend_no_padding,
        test_4_attend_with_padding,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"  [ERROR] {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    print("Interpretation:")
    print("  1 fails -> kernel/build problem, or softmax_scale mismatch")
    print("  2 fails -> cu_seqlens semantics differ from what we assume")
    print("  5 fails -> unpad index ordering is wrong")
    print("  3 fails -> wrapper bug (layout, dtype, or q/k/v order)")
    print("  4 fails -> unpad+scatter interaction with ragged lengths")


if __name__ == "__main__":
    main()
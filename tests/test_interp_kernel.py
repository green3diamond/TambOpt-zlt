"""Unit test for the interpolated (adaptive-bandwidth) detector kernel. CPU only,
no checkpoints.

`adaptive_bandwidths` widens each point's Gaussian to its k-th nearest in-plane
neighbour distance (floored at sigma); the ring kernel (test_shower_frame.py)
area-normalises it. Pinned here:

  - a dense cloud (spacing << sigma) is left at sigma;
  - two sparse points get their separation as width;
  - widths are searched per plane and padding rows are untouched;
  - the vectorised search matches the per-shower reference loop.

    python tests/test_interp_kernel.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from modules.showers.kernel import adaptive_bandwidths

SIGMA = 10.0
INF = float("inf")       # the algorithm uncapped; runs may cap widths (constants.KERNEL_INTERP_MAX)


def _cloud(xy, energy=1.0, layer=0.0, time=1e-7):
    """(1, P, 5) cloud from (P, 2) transverse positions, all in one plane."""
    xy = torch.as_tensor(xy, dtype=torch.float32)
    c = torch.zeros((1, xy.shape[0], 5))
    c[0, :, :2] = xy
    c[0, :, 2] = layer
    c[0, :, 3] = energy
    c[0, :, 4] = time
    return c


def test_dense_cloud_unchanged():
    g = torch.arange(0, 60, 2.0)                       # 2 m spacing << sigma
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    cloud = _cloud(xy)
    h = adaptive_bandwidths(cloud, k=8, sigma_min=SIGMA, sigma_max=INF)
    assert torch.all(h == SIGMA), f"dense cloud got h up to {h.max():.1f}"


def test_sparse_points_get_their_separation():
    cloud = _cloud([[-30.0, 0.0], [30.0, 0.0]])
    h = adaptive_bandwidths(cloud, k=1, sigma_min=SIGMA, sigma_max=INF)
    assert torch.allclose(h, torch.full_like(h, 60.0)), h


def test_padding_and_planes_are_respected():
    # Two planes far apart in layer; a padding row (energy 0) sits between.
    c = torch.zeros((1, 5, 5))
    c[0, :, :2] = torch.tensor([[0, 0], [50, 0], [0, 0], [500, 0], [25, 0]], dtype=torch.float32)
    c[0, :, 2] = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32)
    c[0, :, 3] = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)
    h = adaptive_bandwidths(c, k=1, sigma_min=SIGMA, sigma_max=INF)
    assert torch.allclose(h[0, :2], torch.tensor([50.0, 50.0])), h      # own plane only
    assert torch.allclose(h[0, 2:4], torch.tensor([500.0, 500.0])), h
    assert float(h[0, 4]) == SIGMA                                        # padding untouched


def _bandwidths_reference(samples, k, sigma_min):
    """The original per-shower, per-plane loop, kept as the parity reference."""
    B, P, _ = samples.shape
    h = torch.full((B, P), float(sigma_min))
    layer, real = samples[..., 2].round().long(), samples[..., 3] > 0
    for b in range(B):
        for L in torch.unique(layer[b][real[b]]):
            idx = torch.nonzero(real[b] & (layer[b] == L)).squeeze(-1)
            if idx.numel() < 2:
                continue
            kk = min(int(k), idx.numel() - 1)
            pts = samples[b, idx, :2].double()
            r = torch.cdist(pts, pts).topk(kk + 1, dim=1, largest=False).values[:, kk]
            h[b, idx] = r.float().clamp(min=float(sigma_min))
    return h


def test_vectorised_bandwidths_match_reference():
    torch.manual_seed(1)
    B, P = 5, 300
    c = torch.zeros((B, P, 5))
    c[..., :2] = torch.rand(B, P, 2) * 800 + 3000          # far from origin, like ENU
    c[..., 2] = torch.randint(0, 6, (B, P)).float()        # several planes
    c[..., 3] = torch.rand(B, P) + 0.1
    for b, n_real in enumerate([300, 250, 7, 1, 0]):       # padding, tiny, lone, empty
        c[b, n_real:, 3] = 0.0
    c[1, 40, 3] = 0.0                                      # an interior zeroed point
    ref = _bandwidths_reference(c, k=4, sigma_min=SIGMA)
    for max_pairs in (1 << 27, 5000):                      # one block, then heavy chunking
        got = adaptive_bandwidths(c, k=4, sigma_min=SIGMA, max_pairs=max_pairs, sigma_max=INF)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-3), (got - ref).abs().max()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} passed")

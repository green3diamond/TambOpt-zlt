"""Unit test for the interpolated (adaptive-bandwidth) detector kernel. CPU only,
no checkpoints.

`adaptive_bandwidths` widens each point's Gaussian to its k-th nearest in-plane
neighbour distance (floored at sigma), and `GetCounts_planeaware(bandwidth=...)`
area-normalises it as (sigma/h)^2 * exp(-d^2/2h^2). The properties that make
that a safe opt-in are pinned here:

  - bandwidth=None is the fixed kernel, and h == sigma everywhere reproduces it;
  - a dense cloud (spacing << sigma) is left unchanged;
  - a detector between two sparse points reads more than the fixed kernel gives;
  - each point's deposit is conserved: over a fine detector grid the summed
    reading is the same with and without interpolation.

    python tests/test_interp_kernel.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from modules.showers.kernel import GetCounts_planeaware, adaptive_bandwidths

SIGMA = 10.0


def _cloud(xy, energy=1.0, layer=0.0, time=1e-7):
    """(1, P, 5) cloud from (P, 2) transverse positions, all in one plane."""
    xy = torch.as_tensor(xy, dtype=torch.float32)
    c = torch.zeros((1, xy.shape[0], 5))
    c[0, :, :2] = xy
    c[0, :, 2] = layer
    c[0, :, 3] = energy
    c[0, :, 4] = time
    return c


def _read(cloud, det_xy, bandwidth=None, hit_min=1e-3):
    det_xy = torch.as_tensor(det_xy, dtype=torch.float32)
    z = torch.zeros(det_xy.shape[0])
    return GetCounts_planeaware(cloud, det_xy[:, 0], det_xy[:, 1], z, None, None, None,
                                sigma=SIGMA, hit_deposit_min=hit_min, bandwidth=bandwidth)


def test_sigma_bandwidth_matches_fixed_kernel():
    cloud = _cloud(torch.rand(50, 2) * 200)
    det = torch.rand(20, 2) * 200
    E0, T0 = _read(cloud, det)
    E1, T1 = _read(cloud, det, bandwidth=torch.full((1, 50), SIGMA))
    assert torch.allclose(E0, E1, rtol=1e-6, atol=0), (E0 - E1).abs().max()
    assert torch.equal(T0, T1)


def test_dense_cloud_unchanged():
    g = torch.arange(0, 60, 2.0)                       # 2 m spacing << sigma
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    cloud = _cloud(xy)
    h = adaptive_bandwidths(cloud, k=8, sigma_min=SIGMA)
    assert torch.all(h == SIGMA), f"dense cloud got h up to {h.max():.1f}"
    det = torch.rand(30, 2) * 60
    assert torch.allclose(_read(cloud, det)[0], _read(cloud, det, bandwidth=h)[0], rtol=1e-6)


def test_detector_between_sparse_points_reads_more():
    cloud = _cloud([[-30.0, 0.0], [30.0, 0.0]])
    h = adaptive_bandwidths(cloud, k=1, sigma_min=SIGMA)
    assert torch.allclose(h, torch.full_like(h, 60.0)), h
    E_raw, _ = _read(cloud, [[0.0, 0.0]])
    E_int, _ = _read(cloud, [[0.0, 0.0]], bandwidth=h)
    expect_raw = 2 * math.exp(-30.0 ** 2 / (2 * SIGMA ** 2))
    expect_int = 2 * (SIGMA / 60.0) ** 2 * math.exp(-30.0 ** 2 / (2 * 60.0 ** 2))
    assert abs(float(E_raw) - expect_raw) < 1e-6 * max(1.0, expect_raw)
    assert abs(float(E_int) - expect_int) < 1e-5, (float(E_int), expect_int)
    assert float(E_int) > 2 * float(E_raw)


def test_deposit_is_conserved_over_a_fine_grid():
    torch.manual_seed(0)
    xy = torch.rand(40, 2) * 600                        # sparse: ~95 m spacing
    e = torch.rand(40) + 0.5
    cloud = _cloud(xy)
    cloud[0, :, 3] = e
    h = adaptive_bandwidths(cloud, k=4, sigma_min=SIGMA)
    assert float(h.max()) > 3 * SIGMA                   # interpolation actually kicks in
    pad, step = 4 * float(h.max()), 4.0
    g = torch.arange(-pad, 600 + pad, step)
    det = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    total_raw = float(_read(cloud, det)[0].sum()) * step ** 2
    total_int = float(_read(cloud, det, bandwidth=h)[0].sum()) * step ** 2
    expect = 2 * math.pi * SIGMA ** 2 * float(e.sum())
    assert abs(total_raw / expect - 1) < 0.01, total_raw / expect
    assert abs(total_int / expect - 1) < 0.01, total_int / expect


def test_padding_and_planes_are_respected():
    # Two planes far apart in layer; a padding row (energy 0) sits between.
    c = torch.zeros((1, 5, 5))
    c[0, :, :2] = torch.tensor([[0, 0], [50, 0], [0, 0], [500, 0], [25, 0]], dtype=torch.float32)
    c[0, :, 2] = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32)
    c[0, :, 3] = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)
    h = adaptive_bandwidths(c, k=1, sigma_min=SIGMA)
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
        got = adaptive_bandwidths(c, k=4, sigma_min=SIGMA, max_pairs=max_pairs)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-3), (got - ref).abs().max()


def test_mean_time_is_deposit_weighted():
    # Two points, energies 1 and 3, times 10 and 30 ns, both at the detector.
    c = _cloud([[0.0, 0.0], [0.0, 0.0]])
    c[0, :, 3] = torch.tensor([1.0, 3.0])
    c[0, :, 4] = torch.tensor([10e-9, 30e-9])
    det = torch.tensor([[0.0, 0.0], [500.0, 0.0]])            # second detector sees nothing
    z = torch.zeros(2)
    E, T = GetCounts_planeaware(c, det[:, 0], det[:, 1], z, None, None, None,
                                sigma=SIGMA, t_mode="mean")
    assert abs(float(T[0, 0]) - 25e-9) < 1e-12, float(T[0, 0])  # (1*10 + 3*30)/4
    assert float(T[0, 1]) == 0.0                                # dark -> 0 sentinel
    _, T_lead = GetCounts_planeaware(c, det[:, 0], det[:, 1], z, None, None, None, sigma=SIGMA)
    assert abs(float(T_lead[0, 0]) - 10e-9) < 1e-12              # default unchanged


def test_mean_energy_mode():
    # Dense cluster at the detector: weight >> 1, so E is the mean point energy,
    # independent of how many points there are. Far detector: weight << 1, so E
    # falls back to the sum and reads ~0. T and the default E are unchanged.
    g = torch.arange(-4, 5, 1.0)
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    c = _cloud(xy, energy=2.0)
    c = torch.cat([c, torch.zeros(1, 20, 5)], dim=1)            # padding rows at the origin
    det = torch.tensor([[0.0, 0.0], [500.0, 0.0]])
    z = torch.zeros(2)
    E_sum, T_sum = GetCounts_planeaware(c, det[:, 0], det[:, 1], z, None, None, None, sigma=SIGMA)
    E_mean, T_mean = GetCounts_planeaware(c, det[:, 0], det[:, 1], z, None, None, None,
                                          sigma=SIGMA, e_mode="mean")
    assert abs(float(E_mean[0, 0]) - 2.0) < 1e-4, float(E_mean[0, 0])   # padding not counted
    assert float(E_sum[0, 0]) > 50.0                                     # the sum counts points
    assert float(E_mean[0, 1]) == float(E_sum[0, 1]) < 1e-6              # far: sum fallback
    assert torch.equal(T_sum, T_mean)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} passed")

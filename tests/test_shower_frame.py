"""Unit tests for the shower-frame kernel (run 18). CPU only, no checkpoints.

  - (B, n_det) per-shower detector coordinates match the shared (n_det,) path;
  - for an exactly horizontal, West-going shower without circular t2 or time
    interpolation, the shower frame reproduces the global frame with the layer fix;
  - t1 is C8's horizontal transverse coordinate, exactly, for an inclined shower;
  - time interpolation carries a point's time along the axis at c;
  - circular t2 is deterministic in the event ids and a within-layer permutation of t1;
  - KERNEL_FRAME == "shower" without frames raises instead of mislabelling.

    python tests/test_shower_frame.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import modules.data.dataset_builder as DB                      # noqa: E402
from modules.showers.kernel import GetCounts_planeaware, adaptive_bandwidths, C_LIGHT  # noqa: E402

SIGMA = 10.0


class FlatSurface:
    """Up = const, the SurfaceUpMap call signature (north, east) -> up."""

    def __init__(self, up=0.0):
        self.up = up

    def __call__(self, n, e):
        return torch.full_like(n, self.up)


def _native_cloud(B, P, gen, layers=6, spread=80.0, d=None):
    """Native AllShowers-like clouds: x, y = horizontal offsets from the decay, i.e.
    the plane centre's horizontal position (l + 1) * 500 * d_h plus an in-plane
    offset (a along x_hat, b along y_hat, C8's plane axes)."""
    d = torch.tensor([-1.0, 0.0, 0.0]) if d is None else d / d.norm()
    th, ph = math.acos(float(d[2])), math.atan2(float(d[1]), float(d[0]))
    xh = torch.tensor([math.cos(ph) * math.cos(th), math.sin(ph) * math.cos(th), -math.sin(th)])
    yh = torch.tensor([math.sin(ph), -math.cos(ph), 0.0])
    lay = torch.randint(0, layers, (B, P), generator=gen).float()
    a = torch.randn(B, P, generator=gen) * spread
    b = torch.randn(B, P, generator=gen) * spread
    pos3 = ((lay + 1) * 500).unsqueeze(-1) * d + a.unsqueeze(-1) * xh + b.unsqueeze(-1) * yh
    c = torch.zeros(B, P, 5)
    c[..., 0], c[..., 1], c[..., 2] = pos3[..., 0], pos3[..., 1], lay
    c[..., 3] = torch.rand(B, P, generator=gen) + 0.1
    c[..., 4] = (lay + 1) * 500 / C_LIGHT + torch.rand(B, P, generator=gen) * 1e-8
    return c, a, b


def test_per_shower_detectors_match_shared():
    gen = torch.Generator().manual_seed(0)
    cloud = torch.zeros(3, 40, 5)
    cloud[..., :2] = torch.rand(3, 40, 2, generator=gen) * 200
    cloud[..., 2] = torch.randint(0, 3, (3, 40), generator=gen).float()
    cloud[..., 3] = 1.0
    cloud[..., 4] = torch.rand(3, 40, generator=gen) * 1e-6
    x, y, z = torch.rand(20) * 200, torch.rand(20) * 200, torch.rand(20) * 2
    bw = adaptive_bandwidths(cloud, 8, SIGMA)
    E0, T0 = GetCounts_planeaware(cloud, x, y, z, None, None, None, sigma=SIGMA, bandwidth=bw)
    E1, T1 = GetCounts_planeaware(cloud, x.expand(3, -1), y.expand(3, -1), z.expand(3, -1),
                                  None, None, None, sigma=SIGMA, bandwidth=bw)
    assert torch.allclose(E0, E1) and torch.equal(T0, T1)


def test_shower_frame_equals_global_for_horizontal_west_shower():
    gen = torch.Generator().manual_seed(1)
    B, P = 2, 300
    raw, _, _ = _native_cloud(B, P, gen)
    # Decay at East = EAST_ENTRY, so the global East slabs coincide with the C8 planes.
    pos = torch.tensor([[1500.0, 30.0, 200.0], [1500.0, -50.0, 150.0]])
    dirs = torch.tensor([[-1.0, 0.0, 0.0]] * B)
    surf = FlatSurface(up=180.0)
    e_det = 1500.0 - torch.rand(60, generator=gen) * 3000
    n_det = (torch.rand(60, generator=gen) - 0.5) * 300

    g = raw.clone()
    DB.place_clouds_enu(g, pos, dirs, east_entry=1500.0, layer_east_dx=500.0, layer_offset=1.0)
    old = DB.KERNEL_FRAME
    DB.KERNEL_FRAME = "global"
    try:
        Eg, Tg = DB.compute_labels_batch(g, e_det, n_det, surf, interp_k=8)
    finally:
        DB.KERNEL_FRAME = old

    sfc = raw.clone()
    frames = DB.place_clouds_shower(sfc, pos, dirs, torch.arange(B), 0, circular=False)
    Es, Ts = DB.compute_labels_batch(sfc, e_det, n_det, surf, interp_k=8, frames=frames, t_interp=False)
    assert torch.allclose(Eg, Es, rtol=1e-4, atol=1e-6), (Eg - Es).abs().max()
    assert torch.allclose(Tg, Ts, rtol=0, atol=1e-12), (Tg - Ts).abs().max()


def test_t1_is_c8_transverse_for_inclined_shower():
    gen = torch.Generator().manual_seed(2)
    d = torch.tensor([0.3, 0.8, -0.35])
    raw, a, b = _native_cloud(1, 200, gen, d=d)
    c = raw.clone()
    DB.place_clouds_shower(c, torch.zeros(1, 3), d.view(1, 3), torch.arange(1), 0, circular=False)
    assert torch.allclose(c[0, :, 0], -b[0], atol=1e-2), (c[0, :, 0] + b[0]).abs().max()
    assert torch.equal(c[0, :, 2], raw[0, :, 2] + 1)


def test_time_interpolation_between_planes():
    cloud = torch.zeros(1, 1, 5)
    cloud[0, 0] = torch.tensor([0.0, 0.0, 3.0, 1.0, 5e-6])
    x = torch.zeros(1)
    for zd, t_exp in ((3.0, 5e-6), (3.5, 5e-6 + 250 / C_LIGHT), (2.5, 5e-6 - 250 / C_LIGHT)):
        _, T = GetCounts_planeaware(cloud, x, x, torch.tensor([zd]), None, None, None,
                                    sigma=SIGMA, t_interp=True, layer_dx=500.0)
        assert abs(float(T[0, 0]) - t_exp) < 1e-12, (zd, float(T[0, 0]), t_exp)
    _, T = GetCounts_planeaware(cloud, x, x, torch.tensor([3.5]), None, None, None, sigma=SIGMA)
    assert abs(float(T[0, 0]) - 5e-6) < 1e-12           # plane time without interpolation


def test_circular_t2_permutation_and_determinism():
    gen = torch.Generator().manual_seed(3)
    raw, _, _ = _native_cloud(4, 500, gen)
    raw[1, 450:] = 0.0                                   # padding rows
    pos, dirs = torch.zeros(4, 3), torch.tensor([[-0.8, 0.6, 0.0]] * 4)
    c1, c2, c3 = raw.clone(), raw.clone(), raw.clone()
    DB.place_clouds_shower(c1, pos, dirs, torch.arange(4), 1)
    DB.place_clouds_shower(c2, pos, dirs, torch.arange(4), 1)
    DB.place_clouds_shower(c3, pos, dirs, torch.arange(4) + 7, 1)
    assert torch.equal(c1, c2)
    assert not torch.equal(c1[..., 1], c3[..., 1])
    assert torch.all(c1[1, 450:, :2] == 0)
    for i in range(4):
        real = c1[i, :, 3] > 0
        for L in c1[i, real, 2].unique():
            m = real & (c1[i, :, 2] == L)
            assert torch.equal(c1[i, m, 0].sort().values, c1[i, m, 1].sort().values)


def test_shower_frame_without_frames_raises():
    cloud = torch.zeros(1, 4, 5)
    cloud[..., 3] = 1.0
    old = DB.KERNEL_FRAME
    DB.KERNEL_FRAME = "shower"
    try:
        DB.compute_labels_batch(cloud, torch.zeros(2), torch.zeros(2), FlatSurface())
    except ValueError:
        return
    finally:
        DB.KERNEL_FRAME = old
    raise AssertionError("expected ValueError without frames")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} passed")

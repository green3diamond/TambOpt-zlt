"""Relative-time recon input (modules/surrogates/recon.py: reltime_features, DeepSetsRecon(input_mode)).

    python tests/test_recon_reltime.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from modules.surrogates.recon import DeepSetsRecon, build_recon_from_ckpt, reltime_features  # noqa: E402


def raw(B=4, n=10, frac_lit=0.7, frac_timed=0.4, seed=0):
    """lit = E > 0; timed = T > 0, a subset of lit (as the kernel: a time needs one point over threshold)."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(B, n, 2, generator=g) * 6000 - 3000
    t_us = 5 + 20 * torch.rand(B, n, generator=g)
    lit = torch.rand(B, n, generator=g) < frac_lit
    timed = lit & (torch.rand(B, n, generator=g) < frac_timed / frac_lit)
    lit[:, 0] = timed[:, 0] = True
    E = torch.where(lit, 0.05 + torch.rand(B, n, generator=g) * 3, torch.zeros(B, n))
    T = torch.where(timed, torch.log1p(t_us), torch.zeros(B, n))
    return torch.cat([xy, E[..., None], T[..., None]], -1).double(), t_us.double(), lit, timed


def test_features_by_hand():
    x, t_us, lit, timed = raw()
    f = reltime_features(x)
    assert f.shape == x.shape[:2] + (9,)
    for b in range(x.shape[0]):
        k = int(torch.where(timed[b], t_us[b], torch.full_like(t_us[b], 1e30)).argmin())
        assert torch.allclose(f[b, :, :2], x[b, :, :2] - x[b, k, :2])
        assert torch.allclose(f[b, :, 3], (t_us[b] - t_us[b, k]) * 1e3 * timed[b], atol=1e-6)
        assert torch.equal(f[b, :, 4], lit[b].double()) and torch.equal(f[b, :, 5], timed[b].double())
        assert torch.allclose(f[b, :, 6:8], x[b, k, :2].expand(x.shape[1], 2))
        assert torch.allclose(f[b, :, 8], torch.log1p(t_us[b, k]).expand(x.shape[1]))
        assert (f[b, ~timed[b], 3] == 0).all() and (f[b, timed[b], 3] >= 0).all()
        assert ((f[b, :, 4] - f[b, :, 5]) >= 0).all()          # timed implies lit


def test_time_shift_only_moves_t0():
    x, t_us, lit, timed = raw()
    x2 = x.clone()
    x2[..., 3] = torch.where(timed, torch.log1p(t_us + 3.7), x[..., 3])     # +3.7 us on every timed detector
    f, f2 = reltime_features(x), reltime_features(x2)
    assert torch.allclose(f[..., :8], f2[..., :8], atol=1e-6)
    assert not torch.allclose(f[..., 8], f2[..., 8])


def test_position_shift_keeps_relative():
    x = raw()[0]
    x2 = x.clone()
    x2[..., :2] += torch.tensor([123.0, -456.0], dtype=x.dtype)
    f, f2 = reltime_features(x), reltime_features(x2)
    assert torch.allclose(f[..., :6], f2[..., :6], atol=1e-6)
    assert torch.allclose(f2[..., 6:8] - f[..., 6:8], torch.tensor([123.0, -456.0], dtype=x.dtype).expand_as(f[..., 6:8]))


def test_lit_without_time_and_dark():
    x = raw(B=3)[0]
    x[1, :, 3] = 0.0                                   # row 1: energy but no time anywhere
    x[2, :, 2:] = 0.0                                  # row 2: nothing at all
    f = reltime_features(x)
    assert torch.isfinite(f).all()
    k = int(x[1, :, 2].argmax())                       # reference = brightest lit detector
    assert torch.allclose(f[1, :, 6:8], x[1, k, :2].expand(x.shape[1], 2))
    assert (f[1, :, 3] == 0).all() and (f[1, :, 5] == 0).all() and (f[1, :, 8] == 0).all()
    assert (f[1, :, 4] == (x[1, :, 2] > 0).double()).all()
    assert (f[2, :, 3:6] == 0).all() and torch.allclose(f[2, :, 6:8], x[2, :, :2].mean(0).expand(x.shape[1], 2))
    m = DeepSetsRecon(n_det=10, hidden=32, context=16, n_enc=2, n_dec=2, pool="maxmean", input_mode="reltime").double()
    assert torch.isfinite(m(x)).all()


def test_gradient_wrt_positions():
    x = raw()[0]
    x.requires_grad_(True)
    m = DeepSetsRecon(n_det=10, hidden=32, context=16, n_enc=2, n_dec=2, pool="maxmean", input_mode="reltime").double()
    m(x).sum().backward()
    g = x.grad[..., :2]
    assert torch.isfinite(g).all() and g.norm() > 0


def test_raw_mode_and_ckpt_roundtrip():
    torch.manual_seed(0)
    old = DeepSetsRecon(n_det=10, hidden=32, context=16, n_enc=2, n_dec=2, pool="maxmean")
    x = raw()[0].float()
    ck = dict(state_dict=old.state_dict(), input_features=4, num_detectors=10,
              input_mean=torch.zeros(4), input_std=torch.ones(4), target_mean=torch.zeros(4), target_std=torch.ones(4),
              config=dict(model_type="deepsets", hidden=32, context=16, n_enc=2, n_dec=2, pool="maxmean"))
    m = build_recon_from_ckpt(ck, 10)                          # no input_mode key -> raw
    assert m.input_mode == "raw" and torch.allclose(m(x), old.eval()(x))
    new = DeepSetsRecon(n_det=10, hidden=32, context=16, n_enc=2, n_dec=2, pool="maxmean", input_mode="reltime")
    ck2 = dict(ck, state_dict=new.state_dict(), input_mean=torch.zeros(9), input_std=torch.ones(9),
               config=dict(ck["config"], input_mode="reltime"))
    m2 = build_recon_from_ckpt(ck2, 10)
    assert m2.input_mode == "reltime" and torch.allclose(m2(x), new.eval()(x))


def test_one_degree_is_visible():
    """The point of the change: a 1-degree tilt over 1 km moves dt by ~58 ns (linear), vs ~0.005 in log1p(us)."""
    x = raw(B=1, n=2, frac_lit=1.0, frac_timed=1.0)[0]
    x[0, :, :2] = torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=x.dtype)
    dt_ns = 1000.0 * math.sin(math.radians(1.0)) / 0.299792458
    x[0, :, 3] = torch.log1p(torch.tensor([10.0, 10.0 + dt_ns / 1e3], dtype=x.dtype))
    f = reltime_features(x)
    assert abs(float(f[0, 1, 3]) - dt_ns) < 1e-6 and float(x[0, 1, 3] - x[0, 0, 3]) < 0.01


if __name__ == "__main__":
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn(); n += 1; print(f"ok  {k}")
    print(f"{n} passed")

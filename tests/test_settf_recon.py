"""SetTransformerRecon: permutation invariance, dark slots, gradients in x/y, checkpoint round trip."""
import math

import torch

from modules.surrogates.recon import SetTransformerRecon, build_recon_from_ckpt


def _input(B=4, n=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(B, n, 2, generator=g) - 0.5) * 6000
    hit = torch.rand(B, n, generator=g) < 0.2
    E = torch.where(hit, torch.rand(B, n, generator=g) * 5 + 0.2, torch.rand(B, n, generator=g) * 1e-4)
    T = torch.where(hit, torch.log1p(10 + torch.rand(B, n, generator=g) * 5), torch.zeros(B, n))
    return torch.cat([xy, E[..., None], T[..., None]], -1)


def _model(seed=0):
    torch.manual_seed(seed)
    return SetTransformerRecon(n_blocks=2).eval()


def test_permutation_invariant():
    m, x = _model(), _input()
    p = torch.randperm(x.shape[1])
    assert torch.allclose(m(x), m(x[:, p]), atol=1e-4)


def test_dark_slots_ignored():
    m, x = _model(), _input(n=60)
    dark = torch.zeros(x.shape[0], 40, 4)
    dark[..., :2] = x[:, :40, :2]
    y1, y2 = m(x), m(torch.cat([x, dark], 1))
    # reltime_features still broadcasts the array-wide reference; the attention and pool must ignore dark slots.
    assert (y1 - y2).abs().max() < 5e-2


def test_grad_xy():
    m, x = _model(), _input()
    x.requires_grad_(True)
    m(x).sum().backward()
    g = x.grad[..., :2]
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_ckpt_round_trip():
    m, x = _model(), _input()
    cfg = dict(model_type="settf", hidden=256, context=128, n_enc=4, n_dec=4, pool="maxmean", d_model=128,
               n_heads=4, n_blocks=2, pair_hidden=32, hit_e_min=0.1, hit_tau=0.5, input_mode="reltime")
    ck = dict(config=cfg, num_detectors=100, input_features=4, state_dict=m.state_dict(),
              input_mean=m.in_mean, input_std=m.in_std, target_mean=m.out_mean, target_std=m.out_std)
    assert torch.allclose(build_recon_from_ckpt(ck, 100)(x), m(x), atol=1e-6)

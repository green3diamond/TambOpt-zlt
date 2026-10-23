"""Unit tests for the counting noise (modules/optimize/counting.py) and the per-row axis distance
(modules/optimize/background.py). CPU only, no checkpoints.

  - E' is unbiased: the mean over many draws equals E, per species, with and without r;
  - P(N = 0) = exp(-mu), and T is zeroed exactly where nothing was counted;
  - mode="expected" is the identity; the same seed gives the same draw;
  - photons scale with eps_gamma; r=None uses the band values;
  - straight_through: forward = counted E, gradient = that of E;
  - axis_distance_rows matches axis_distance for a layout shared by every row.

    python tests/test_counting.py
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from modules.optimize.background import axis_distance, axis_distance_rows, particles_per_E  # noqa: E402
from modules.optimize.counting import count_species, expected_counts, p_hit, KEY  # noqa: E402

SP = ("electron", "muon", "photon")


def _preds(E, t_us=5.0):
    """Per-species (B, nd, 2) labels [log1p E, log1p(t * 1e6)] from one (B, nd) E for every species."""
    T = torch.full_like(E, math.log1p(t_us))
    return [torch.stack([torch.log1p(E), T], -1) for _ in SP]


def _gen(seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return g


def test_unbiased_per_species():
    E = torch.full((200_000, 2), 30.0)
    for r in (None, torch.full_like(E, 300.0)):
        out, _ = count_species(_preds(E), r, _gen(), eps_gamma=0.02)
        for nm, o in zip(SP, out):
            m = float(torch.expm1(o[..., 0]).mean())
            mu = float(expected_counts([o for o in _preds(E)], r, 0.02)[SP.index(nm)].mean())
            tol = 4.0 / math.sqrt(mu * E.numel()) + 1e-3                        # 4 sigma of the relative mean
            assert abs(m - 30.0) / 30.0 < tol, (nm, r is None, m, mu)


def test_p_zero_and_time_zeroing():
    E = torch.full((100_000, 1), 50.0)
    preds = _preds(E)
    mu = expected_counts(preds, None, 0.02)
    out, n = count_species(preds, None, _gen(1), eps_gamma=0.02)
    for k, o in enumerate(out):
        zero = torch.expm1(o[..., 0]) == 0
        assert torch.equal(zero, o[..., 1] == 0), SP[k]                      # T zeroed exactly where N = 0
        p0 = float(zero.float().mean())
        assert abs(p0 - math.exp(-float(mu[k][0, 0]))) < 0.01, (SP[k], p0, float(mu[k][0, 0]))
    assert torch.equal(n > 0, sum((torch.expm1(o[..., 0]) > 0).float() for o in out) > 0)
    ph = p_hit(preds, None, 0.02)
    assert abs(float(ph[0, 0]) - (1 - math.exp(-sum(float(m[0, 0]) for m in mu)))) < 1e-6


def test_expected_mode_and_seed():
    E = torch.rand(64, 7) * 100
    preds = _preds(E)
    same, n = count_species(preds, None, None, mode="expected")
    assert n is None and all(torch.equal(a, b) for a, b in zip(same, preds))
    a, _ = count_species(preds, None, _gen(5))
    b, _ = count_species(preds, None, _gen(5))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_photon_eps_and_band():
    E = torch.full((3, 4), 10.0)
    mu1 = expected_counts(_preds(E), None, 0.02)[2]
    mu2 = expected_counts(_preds(E), None, 0.04)[2]
    assert torch.allclose(mu2, 2 * mu1)
    for nm, mu in zip(SP, expected_counts(_preds(E), None, 1.0)):
        assert torch.allclose(mu, E * particles_per_E(KEY[nm]))


def test_straight_through():
    E = torch.full((50, 3), 20.0, requires_grad=True)
    preds = [torch.stack([torch.log1p(E), torch.zeros_like(E)], -1) for _ in SP]
    out, _ = count_species(preds, None, _gen(2), straight_through=True)
    Ec = torch.expm1(out[0][..., 0])
    Ec.sum().backward()
    assert torch.allclose(E.grad, torch.ones_like(E), atol=1e-4)            # dE'/dE = 1
    n = (Ec * particles_per_E("e")).detach()
    assert torch.allclose(n, n.round(), atol=1e-3)                          # forward value = an integer count


def test_axis_distance_rows_matches_shared():
    torch.manual_seed(0)
    B, nd = 6, 9
    p = torch.randn(B, 3) * 1000
    d = torch.nn.functional.normalize(torch.randn(B, 3) + torch.tensor([3.0, 0, 0]), dim=1)
    e1 = torch.nn.functional.normalize(torch.stack([-d[:, 1], d[:, 0], torch.zeros(B)], 1), dim=1)
    frames = torch.cat([p, d, e1, torch.cross(d, e1, dim=1)], 1)
    e, n = torch.randn(nd) * 2000, torch.randn(nd) * 2000

    def surface(nn_, ee):                                                     # any smooth Up(N, E)
        return 0.1 * nn_ - 0.05 * ee + 30.0

    r1 = axis_distance(e, n, surface, frames)
    r2 = axis_distance_rows(torch.stack([e, n], -1)[None].expand(B, -1, -1), surface, frames)
    assert torch.allclose(r1, r2, rtol=1e-4, atol=1e-2), (r1 - r2).abs().max()


def test_cosmic_background_independent_of_r():
    """Default background (TAMBOPT_BG_UNITS=cosmic): each muon's own random KE, mean ~3.85 GeV; the amplitude does not
    depend on the detector's distance from the shower axis."""
    import math
    from modules.optimize import background as bg
    if bg.BG_UNITS != "cosmic":
        return
    ke = bg.sample_bg_muon_ke((200_000,), torch.Generator().manual_seed(1))
    assert ke.min() >= bg.KE_MIN - 1e-6 and ke.max() <= bg.KE_MAX + 1e-3
    assert abs(float(ke.clamp(max=100).mean()) - 3.6) < 0.4                 # tail clipped for a stable mean
    assert abs(bg.mean_bg_muon_ke() - 3.85) < 0.05
    E = torch.zeros(4000, 50); t = torch.zeros(4000, 50); timed = torch.zeros(4000, 50, dtype=torch.bool)
    out = []
    for r in (30.0, 3000.0):
        g = torch.Generator().manual_seed(2)
        En, _, Nb, _ = bg.add_background(E, t, timed, t + 5.0, 1e6, g, r_axis=torch.full_like(E, r))
        out.append(En)
        per_muon = En.sum() / Nb.sum()
        assert abs(float(per_muon) / (bg.mean_bg_muon_ke() * 2 * math.pi * 100 / 2) - 1) < 0.15
    assert torch.equal(out[0], out[1])                                     # same draws -> same E at any r


def test_trigger_prob_matches_sampling():
    """trigger_prob (exact Poisson-binomial) = the rate of >= k detectors with a Poisson-drawn particle."""
    from modules.optimize.counting import trigger_prob
    g = torch.Generator().manual_seed(0)
    mu = torch.rand(64, 40, generator=g) ** 3 * 2.0                      # mostly small, some > 1
    for k in (1, 3, 5):
        exact = trigger_prob(mu, k)
        draws = torch.poisson(mu.expand(4000, -1, -1), generator=g)        # (draws, B, nd)
        mc = ((draws > 0).sum(-1) >= k).float().mean(0)
        assert torch.allclose(exact, mc, atol=0.03), (k, (exact - mc).abs().max())
    assert torch.allclose(trigger_prob(torch.zeros(2, 10)), torch.zeros(2))


def test_trigger_prob_gradient():
    from modules.optimize.counting import trigger_prob
    mu = torch.full((1, 10), 0.3, requires_grad=True)
    trigger_prob(mu, 3).sum().backward()
    assert (mu.grad > 0).all()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} passed")

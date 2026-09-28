"""Unit test for the lit/dark (hurdle) Stage-2 surrogate. CPU only, no checkpoints.

    python tests/test_hurdle_surrogate.py
"""
import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from modules.surrogates import DeepSetsSurrogate  # noqa: E402
from modules.surrogates.deepsets import build_surrogate_from_ckpt  # noqa: E402
from modules.surrogates.dual import combine_hurdle_outputs, combine_species_outputs  # noqa: E402

N_DET, PD = 100, 8


def _stats():
    return dict(in_mean=torch.zeros(PD + 2 * N_DET), in_std=torch.ones(PD + 2 * N_DET),
                out_mean=torch.zeros(2 * N_DET), out_std=torch.ones(2 * N_DET),
                hit_mean=torch.tensor([1.5, 2.5]), hit_std=torch.tensor([0.8, 0.4]))


def _model():
    torch.manual_seed(0)
    m = DeepSetsSurrogate(n_det=N_DET, primary_dim=PD, hidden=32, context=8, hurdle=True)
    m.set_normalization(_stats())
    return m


def test_forward_is_p_times_value():
    m = _model()
    q, xy = torch.randn(3, PD), torch.randn(3, N_DET, 2)
    p, mu = m.forward_parts(q, xy)
    assert p.shape == mu.shape == (3, N_DET, 2)
    assert torch.all((p > 0) & (p < 1))
    assert torch.allclose(m(q, xy), p * mu)


def test_checkpoint_round_trip():
    m = _model()
    cfg = dict(model_type="deepsets", hidden=32, context=8, n_enc=3, n_dec=3, dropout=0.0, hurdle=True)
    m2 = build_surrogate_from_ckpt(dict(state_dict=m.state_dict(), norm_stats=_stats(), config=cfg),
                                   N_DET, PD)
    q, xy = torch.randn(2, PD), torch.randn(2, N_DET, 2)
    assert m2.hurdle and torch.allclose(m(q, xy), m2(q, xy))
    # A mean-only config still builds the 2-output model.
    assert DeepSetsSurrogate(n_det=N_DET, primary_dim=PD, hidden=32, context=8).decoder[-1].out_features == 2


def test_combination_reduces_to_leading_edge_when_certain():
    B, n = 2, N_DET
    mus = [torch.rand(B, n, 2) * 3 + 0.5 for _ in range(3)]
    # Each species lit or dark with certainty (p in {0, 1}).
    ps = [torch.randint(0, 2, (B, n, 2)).float() for _ in range(3)]
    got = combine_hurdle_outputs(*zip(ps, mus))
    exact = combine_species_outputs(*(p * mu for p, mu in zip(ps, mus)))
    assert torch.allclose(got, exact, atol=1e-5), (got - exact).abs().max()


def test_combination_ignores_unlikely_species_in_the_min():
    # Species 0 surely lit at encoded T=3; species 1 almost surely dark.
    p0 = torch.ones(1, 1, 2); mu0 = torch.full((1, 1, 2), 3.0)
    p1 = torch.full((1, 1, 2), 1e-4); mu1 = torch.full((1, 1, 2), 1.0)
    t = combine_hurdle_outputs((p0, mu0), (p1, mu1))[..., 1]
    assert abs(float(t) - 3.0) < 1e-3, float(t)       # not dragged toward p1*mu1 ~ 0


def test_hurdle_loss_is_finite_and_differentiable():
    spec = importlib.util.spec_from_file_location("t02", os.path.join(_ROOT, "scripts", "02_train_fnn_deepsets.py"))
    t02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t02)
    m = _model()
    q, xy = torch.randn(4, PD), torch.randn(4, N_DET, 2)
    E = torch.where(torch.rand(4, N_DET) < 0.3, torch.rand(4, N_DET) * 3, torch.zeros(4, N_DET))
    T = torch.where(torch.rand(4, N_DET) < 0.1, torch.rand(4, N_DET) + 2, torch.zeros(4, N_DET))
    loss, lE, lT = t02.stage2_loss(m, q, xy, E, T)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nall {len(tests)} passed")

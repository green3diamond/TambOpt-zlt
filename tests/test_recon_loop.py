"""CPU checks of refresh acceptance, checkpoint compatibility and shower bookkeeping."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from modules.optimize import objective
from modules.surrogates.recon import DeepSetsRecon, build_recon_from_ckpt
from recon_loop import refresh as R

N_DET = 6
ARCH = {"hidden": 16, "context": 8, "n_enc": 2, "n_dec": 2, "pool": "maxmean"}
NORM = (
    torch.tensor([10.0, 2000.0, 0.3, 0.8]),
    torch.tensor([1500.0, 1000.0, 0.9, 1.2]),
    torch.tensor([0.0, 0.5, 0.1, 0.5]),
    torch.tensor([0.4, 0.2, 0.2, 0.3]),
)


def raises(fn, *a, exc=(ValueError, RuntimeError), **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False


def toy(seed=0):
    torch.manual_seed(seed)
    m = DeepSetsRecon(n_det=N_DET, input_features=4, output_dim=4, **ARCH)
    m.set_normalization(*NORM)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def ckpt_of(model):
    return {
        "state_dict": model.state_dict(),
        "val_total": 0.05,
        "epoch": 1,
        "num_detectors": N_DET,
        "input_features": 4,
        "input_mean": NORM[0],
        "input_std": NORM[1],
        "target_mean": NORM[2],
        "target_std": NORM[3],
        "config": dict(model_type="deepsets", output_dim=4, recon_input="kernel", **ARCH),
    }


def data(n, seed, shift=0.0):
    """Toy rows whose target is a fixed function of the features (learnable), shifted per 'layout'."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n, N_DET, 4, generator=g) * NORM[1] + NORM[0]
    z = ((f - NORM[0]) / NORM[1]).mean(1)
    t = torch.tanh(z + shift) * NORM[3] + NORM[2]
    return f, t


def test_fine_tune_learns_and_keeps_buffers():
    recon = toy()
    cfg = R.Config(lr=3e-3, epochs=5, batch=64)
    state, s = R.fine_tune(
        recon, data(2000, 1, 0.5), [data(4000, 2)], data(500, 3), cfg, seed=0, device="cpu"
    )
    assert s["accepted"] and state is not None and s["held_best"] < s["held_before"]
    assert s["n_held"] == 200 and s["n_fresh_train"] == 1800 and s["n_replay"] == 1800
    assert all(not p.requires_grad for p in recon.parameters()), "the live recon must stay frozen"
    before = {k: v.clone() for k, v in recon.state_dict().items()}
    R.install(recon, state)
    assert all(not p.requires_grad for p in recon.parameters())
    assert not all(torch.equal(before[k], v) for k, v in recon.state_dict().items())
    for k in R.BUFFERS:
        assert torch.equal(before[k], recon.state_dict()[k])
    hx, hy = data(2000, 1, 0.5)
    assert R.eval_loss(recon, hx, hy, "cpu") < R.eval_loss(toy(), hx, hy, "cpu")


def test_reject_when_no_gain():
    recon = toy()
    state, s = R.fine_tune(
        recon,
        data(500, 1, 0.5),
        [data(1000, 2)],
        data(100, 3),
        R.Config(lr=0.0, epochs=2),
        seed=0,
        device="cpu",
    )
    assert state is None and not s["accepted"] and s["best_epoch"] == 0
    assert s["anchor_after"] == s["anchor_before"]


def test_replay_share_and_bank_size():
    recon = toy()
    _, s = R.fine_tune(
        recon,
        data(1000, 1),
        [data(5000, 2)],
        data(50, 3),
        R.Config(lr=0.0, epochs=1, replay=0.0),
        seed=0,
        device="cpu",
    )
    assert s["n_replay"] == 0
    _, s = R.fine_tune(
        recon,
        data(1000, 1),
        [data(5000, 2)],
        data(50, 3),
        R.Config(lr=0.0, epochs=1, replay=0.75),
        seed=0,
        device="cpu",
    )
    assert s["n_replay"] == 3 * s["n_fresh_train"]
    assert raises(
        R.fine_tune,
        recon,
        data(1000, 1),
        [data(100, 2)],
        data(50, 3),
        R.Config(),
        seed=0,
        device="cpu",
    )


def test_replay_sources_in_order():
    """Recent examples are used first; the bank only tops up; one source matches a plain draw."""
    recent, bank = data(300, 4), data(5000, 2)
    x, _ = R.draw([recent, bank], 200, torch.Generator().manual_seed(0))
    assert all(any(torch.equal(r, row) for row in recent[0]) for r in x)
    x, _ = R.draw([recent, bank], 500, torch.Generator().manual_seed(0))
    from_recent = sum(any(torch.equal(r, row) for row in recent[0]) for r in x)
    assert len(x) == 500 and from_recent == 300
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    x1, _ = R.draw([bank], 100, g1)
    assert torch.equal(x1, bank[0][torch.randperm(len(bank[0]), generator=g2)[:100]])
    assert raises(R.draw, [recent], 301, torch.Generator())


def test_jitter_layout():
    """Offsets have the requested width, are reproducible and pass through the projection."""
    x, y = torch.zeros(4000), torch.full((4000,), 2000.0)
    a = R.jitter_layout(x, y, 50.0, torch.Generator().manual_seed(1), lambda e, n: (e, n))
    b = R.jitter_layout(x, y, 50.0, torch.Generator().manual_seed(1), lambda e, n: (e, n))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert abs(float(a[0].std()) - 50.0) < 3.0 and abs(float((a[1] - 2000).std()) - 50.0) < 3.0
    clamp = R.jitter_layout(x, y, 50.0, torch.Generator().manual_seed(1), lambda e, n: (e.clamp(max=0), n))
    assert float(clamp[0].max()) <= 0.0


def test_deterministic():
    a = R.fine_tune(
        toy(), data(800, 1, 0.5), [data(800, 2)], data(50, 3), R.Config(lr=3e-3, epochs=2), 5, "cpu"
    )
    b = R.fine_tune(
        toy(), data(800, 1, 0.5), [data(800, 2)], data(50, 3), R.Config(lr=3e-3, epochs=2), 5, "cpu"
    )
    assert a[1]["held_curve"] == b[1]["held_curve"]
    assert all(torch.equal(a[0][k], b[0][k]) for k in a[0])


def test_buffer_change_is_caught():
    recon = toy()
    state = {k: v.clone() for k, v in recon.state_dict().items()}
    state["in_mean"] += 1.0
    assert raises(R.install, toy(), state)
    assert raises(R.checkpoint, ckpt_of(recon), state)


def test_checkpoint_loads_and_predicts():
    recon = toy()
    state, _ = R.fine_tune(
        recon,
        data(1000, 1, 0.5),
        [data(1000, 2)],
        data(50, 3),
        R.Config(lr=3e-3, epochs=2),
        seed=0,
        device="cpu",
    )
    ck = R.checkpoint(ckpt_of(recon), state, version=1)
    loaded = build_recon_from_ckpt(ck, N_DET)
    live = toy()
    R.install(live, state)
    x, _ = data(64, 9)
    with torch.no_grad():
        assert torch.allclose(loaded(x), live(x))
    assert ck["refresh"]["version"] == 1


def test_features_match_objective():
    """R.features must be the exact stack objective._predict_primary feeds the recon."""
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(N_DET, generator=g) * 1000, torch.randn(N_DET, generator=g) * 1000
    ET = torch.rand(3, N_DET, 2, generator=g)
    seen = {}

    def fnn(prim, xy):
        return ET

    def recon(feats):
        seen["f"] = feats
        return torch.tensor([[0.0, 0.0, 1.0, 0.5]]).expand(feats.shape[0], -1)

    objective._predict_primary(x, y, torch.zeros(3, 6), fnn, recon)
    xy = torch.stack([x, y], -1).unsqueeze(0).expand(3, -1, -1)
    assert torch.equal(seen["f"], R.features(xy, ET))


def test_kept_index():
    p = torch.arange(20.0).view(10, 2)
    assert R.kept_index(p, p[[0, 3, 4, 9]]).tolist() == [0, 3, 4, 9]
    assert R.kept_index(p, p).tolist() == list(range(10))
    assert raises(R.kept_index, p, p[[3, 0]])
    assert raises(R.kept_index, p, torch.tensor([[99.0, 99.0]]))


def test_pool_blob():
    n_st, n_sp, k = 2, 3, 5
    blob = torch.zeros(n_st, n_sp, k, dtype=torch.bool)
    blob[:, 1, 2] = True
    blob[:, 2, 4] = True
    species = torch.arange(n_sp).repeat_interleave(k).repeat(n_st)
    strat = torch.arange(n_st).repeat_interleave(n_sp * k)
    assert R.pool_blob(blob.reshape(-1), species, strat).tolist() == [
        False,
        False,
        True,
        False,
        True,
    ]
    blob[1, 0, 0] = True
    assert raises(R.pool_blob, blob.reshape(-1), species, strat)


def test_invalid_refresh_inputs():
    for kw in (dict(replay=1), dict(holdout=0), dict(batch=0), dict(epochs=0)):
        assert raises(R.Config, **kw)
    assert raises(R.split, 1, 0.1, torch.Generator())
    tr, held = R.split(2, 0.99, torch.Generator())
    assert len(tr) == len(held) == 1 and tr[0] != held[0]
    assert raises(R.select, torch.tensor([True, False]), 2, 0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")

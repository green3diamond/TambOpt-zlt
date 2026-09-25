"""CPU self-test of analysis_core.scoring on a toy surrogate and recon."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from analysis_core.events import Events
from analysis_core.scoring import FixedLabels, Score, Scorer, paired
from modules.optimize import utility_of_xy

N_DET, B, CHUNK = 20, 96, 16


class ToySurrogate:
    supports_var = True

    def __call__(self, primary, xy):
        r = xy.norm(dim=-1) / 100.0
        E = torch.log1p(50.0 * torch.exp(-r) * (1.0 + primary[:, 3:4]))
        return torch.stack([E, r], -1)

    def forward_sample(self, primary, xy):
        mean = self(primary, xy)
        return mean + torch.randn_like(mean)


class SamplingSurrogate(ToySurrogate):
    """Broken on purpose: its plain call draws a sample."""

    def __call__(self, primary, xy):
        return self.forward_sample(primary, xy)

    def forward_sample(self, primary, xy):
        mean = ToySurrogate.__call__(self, primary, xy)
        return mean + torch.randn_like(mean)


class ToyRecon(torch.nn.Module):
    def forward(self, feats):
        m = feats.mean(1)
        d = torch.stack([m[:, 0], m[:, 1], m[:, 2] + 1.0], -1)
        return torch.cat([d / d.norm(dim=1, keepdim=True), torch.sigmoid(m[:, 3:4])], -1)


def setup(surrogate=ToySurrogate):
    g = torch.Generator().manual_seed(0)
    dirs = torch.randn(B, 3, generator=g)
    primary = torch.cat([dirs / dirs.norm(dim=1, keepdim=True), torch.rand(B, 1, generator=g),
                         torch.randn(B, 4, generator=g)], 1)
    run = SimpleNamespace(fnn=surrogate(), recon=ToyRecon(), device=torch.device("cpu"), surface=None)
    e, n = 30.0 * torch.randn(N_DET, generator=g), 30.0 * torch.randn(N_DET, generator=g)
    return run, Events([], primary), e, n


def test_chunks_reproduce_full_batch():
    run, ev, e, n = setup()
    got = Scorer(run, ev, "surrogate", CHUNK).score(e, n).U
    want = utility_of_xy(e, n, ev.primary, run.fnn, run.recon, sample=False)[0].item()
    assert want > 1e-3 and abs(got - want) <= 1e-5 * abs(want), (got, want)


def test_mean_path_is_deterministic():
    run, ev, e, n = setup()
    sc = Scorer(run, ev, "surrogate", CHUNK)
    assert np.array_equal(sc.score(e, n).chunks, sc.score(e, n).chunks)


def test_sampling_surrogate_is_caught():
    run, ev, e, n = setup(SamplingSurrogate)
    sc = Scorer(run, ev, "surrogate", CHUNK)
    assert not np.array_equal(sc.score(e, n).chunks, sc.score(e, n).chunks)


def test_paired_matches_numpy():
    a = Score(np.array([1.0, 2.0, 4.0, 3.0]), {})
    b = Score(np.array([0.5, 2.5, 3.0, 1.0]), {})
    d = a.chunks - b.chunks
    assert np.allclose(paired(a, b), (d.mean(), d.std(ddof=1) / 2.0))
    assert paired(a, a) == (0.0, 0.0)


def test_parts_split_the_events():
    _, ev, _, _ = setup()
    parts = [ev.part(i, 3) for i in range(3)]
    assert all(len(p) == B // 3 for p in parts)
    assert torch.equal(torch.cat([p.primary for p in parts]), ev.primary[: 3 * (B // 3)])


def test_bad_inputs_raise():
    run, ev, e, n = setup()
    for bad in (lambda: Scorer(run, ev, "surrogate", CHUNK + 1),
                lambda: Scorer(run, ev, "sampled", CHUNK),
                lambda: FixedLabels(torch.zeros(B, N_DET, 2))(ev.primary[:CHUNK], torch.zeros(CHUNK, N_DET, 2))):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("bad input was accepted")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)

"""CPU self-test of the removal-curve and single-detector-scan logic on a known additive utility."""
import logging
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "landscape_analysis"))

import numpy as np
import torch

import detector_importance as di
import loss_slices as ls
import removal_curves as rc
import single_detector_scan as sds
from analysis_core.scoring import Score

LOG = logging.getLogger("test")
WEIGHTS = torch.tensor([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.5])


class AdditiveScorer:
    """U is the sum of the weights of the detectors present, identified by East position."""

    def score(self, e, n):
        U = float(WEIGHTS[e.long()].sum())
        return Score(np.array([U, U]), {})


def evaluator():
    e = torch.arange(len(WEIGHTS), dtype=torch.float32)
    return rc.Evaluator(e, torch.zeros_like(e), rescale=False)


def test_most_important_first_removes_heaviest():
    order, picked = rc.greedy_order(evaluator(), AdditiveScorer(), 2, True, LOG)
    assert order == torch.argsort(WEIGHTS, descending=True)[:8].tolist(), order
    assert np.allclose(picked, WEIGHTS.sum() - torch.cumsum(WEIGHTS.sort(descending=True).values, 0)[:8])


def test_least_important_first_removes_lightest():
    order, _ = rc.greedy_order(evaluator(), AdditiveScorer(), 2, False, LOG)
    assert order == torch.argsort(WEIGHTS)[:8].tolist(), order


def test_reversed_order_is_caught():
    order, _ = rc.greedy_order(evaluator(), AdditiveScorer(), 2, True, LOG)
    assert order[::-1] != torch.argsort(WEIGHTS, descending=True)[:8].tolist()


def test_curve_counts_and_values():
    order = list(range(len(WEIGHTS)))
    c = rc.curve(evaluator(), AdditiveScorer(), order, floor=2)
    assert c["n"] == list(range(10, 1, -1))
    assert np.isclose(c["U"][0], WEIGHTS.sum()) and np.isclose(c["U"][-1], WEIGHTS[8:].sum())
    sparse = rc.curve(evaluator(), AdditiveScorer(), order, floor=2, every=4)
    assert sparse["n"] == [10, 6, 2]


def test_threshold_rescales_with_detector_count():
    ev = rc.Evaluator(torch.arange(10.0), torch.zeros(10), rescale=True)
    ev(AdditiveScorer(), [0, 1, 2, 3, 4])
    assert np.isclose(rc.objective.RECONSTRUCT_THRESHOLD, rc.BASE_THRESHOLD / 2)
    rc.objective.RECONSTRUCT_THRESHOLD = rc.BASE_THRESHOLD


def test_place_adds_or_moves_without_mutating():
    e, n = torch.zeros(3), torch.zeros(3)
    ea, na = sds.place(e, n, 5.0, 6.0, None)
    em, nm = sds.place(e, n, 5.0, 6.0, 1)
    assert len(ea) == 4 and (ea[-1], na[-1]) == (5.0, 6.0)
    assert len(em) == 3 and (em[1], nm[1]) == (5.0, 6.0)
    assert e.abs().sum() == 0 and n.abs().sum() == 0


def test_pick_detector():
    e, n = torch.tensor([0.0, 1.0, 10.0, -1.0]), torch.tensor([0.0, 1.0, 0.0, -1.0])
    assert sds.pick_detector(e, n, "edge") == 2 and sds.pick_detector(e, n, "3") == 3
    assert sds.pick_detector(e, n, "center") in (0, 1, 3)


def test_scan_grid_orientation_and_mask():
    mountain = SimpleNamespace(east_lo=0.0, east_hi=100.0, n_min=0.0, n_max=50.0)
    run = SimpleNamespace(mountain=mountain, on_mountain=lambda e, n: e < 60)
    E, N, EE, NN, on = sds.scan_grid(run, 25.0)
    assert EE.shape == (len(N), len(E)) == (3, 5)
    assert torch.equal(EE[0], E) and torch.equal(NN[:, 0], N)
    assert on[:, :3].all() and not on[:, 3:].any()


def test_knockout_recovers_weights():
    e = torch.arange(len(WEIGHTS), dtype=torch.float32)
    single = di.knockout(AdditiveScorer(), e, torch.zeros_like(e), [[i] for i in range(len(WEIGHTS))])
    assert np.allclose(single[:, 0], WEIGHTS.numpy()) and np.allclose(single[:, 1], 0.0)
    pair = di.knockout(AdditiveScorer(), e, torch.zeros_like(e), [[0, 2]])
    assert np.isclose(pair[0, 0], WEIGHTS[0] + WEIGHTS[2])


def test_cells_round_trip():
    run = SimpleNamespace(mountain=SimpleNamespace(n_min=0.0, east_lo=0.0))
    ids = di.cell_ids(run, torch.tensor([10.0, 260.0, 260.0]), torch.tensor([10.0, 10.0, 260.0]), 250.0)
    assert ids.tolist() == [0, 1, 10001]
    assert di.cell_centre(run, 10001, 250.0) == (375.0, 375.0)


def test_correlation_interval():
    x = np.arange(50.0)
    r, (lo, hi) = di.corr_ci(x, 2 * x + 1, n_boot=50)
    assert np.isclose(r, 1.0) and np.isclose(lo, 1.0) and np.isclose(hi, 1.0)


def test_edge_distance_on_toy_mountain():
    mountain = SimpleNamespace(east_lo=0.0, east_hi=100.0, n_min=0.0, n_max=50.0)
    run = SimpleNamespace(mountain=mountain, on_mountain=lambda e, n: e < 60)
    d = di.edge_distance(run, torch.tensor([10.0]), torch.tensor([0.0]), spacing=25.0, pad=200.0)
    assert np.isclose(d[0], 65.0), d


class SquareScorer:
    """U is the sum of squared coordinates, so the straight path from 0 to 1 bows below its chord."""

    def score(self, e, n):
        U = float((e ** 2 + n ** 2).sum())
        return Score(np.array([U, U]), {})


def test_flat_layout_split():
    e, n = ls.to_layout(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert e.tolist() == [1.0, 2.0, 3.0] and n.tolist() == [4.0, 5.0, 6.0]


def test_detector_direction_moves_each_detector_one_metre():
    d = ls.detector_direction(50, torch.Generator().manual_seed(3))
    assert torch.allclose(torch.stack(ls.to_layout(d), -1).norm(dim=1), torch.ones(50))


def test_pca_recovers_a_planar_path():
    g = torch.Generator().manual_seed(4)
    basis, _ = torch.linalg.qr(torch.randn(200, 2, generator=g))
    t = torch.linspace(0, 1, 300)
    coeffs = torch.stack([100 * t, 30 * torch.sin(6 * t)], -1)
    path = coeffs @ basis.T + 5.0
    dirs, var, coords = ls.principal_directions(path, path[-1])
    assert var[:2].sum() > 0.9999
    assert torch.allclose(coords @ dirs, path - path[-1], atol=1e-3)


def test_plane_values_on_known_function():
    run = SimpleNamespace(on_mountain=lambda e, n: torch.ones_like(e, dtype=torch.bool))
    centre, u, v = torch.zeros(4), torch.tensor([1.0, 0, 0, 0]), torch.tensor([0, 0, 1.0, 0])
    grid = ls.plane(SquareScorer(), run, centre, u, v, np.array([-1.0, 2.0]), np.array([0.0, 3.0]))
    assert np.allclose(grid[0], [[1.0, 4.0], [10.0, 13.0]]) and np.all(grid[2] == 0)


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

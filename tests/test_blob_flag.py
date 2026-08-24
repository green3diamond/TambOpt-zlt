"""Unit test for the degenerate-shower flag. No GPU, no checkpoints.

Since the Step-0 re-roll was removed (it recovered 0 of 56 — see
BLOB_GUARD_FINDINGS.md), `flag_blob_showers` is the ONLY guard standing between
a degenerate shower and the surrogate targets, so its edge cases are worth
pinning down: the hot-cloud cut itself, empty clouds, and the non-finite
energies the removed ratio check used to catch.

    python tests/test_blob_flag.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from modules.constants import BLOB_MEDIAN_E
from modules.data.dataset_builder import flag_blob_showers


def _cloud(energies):
    """(P, 5) point cloud carrying `energies` in column 3."""
    e = torch.tensor(energies, dtype=torch.float32)
    c = torch.zeros((len(energies), 5), dtype=torch.float32)
    c[:, 3] = e
    return c


def _check(name, clouds, expected):
    got = flag_blob_showers(torch.stack(clouds))
    assert got.tolist() == expected, f"{name}: got {got.tolist()}, want {expected}"
    print(f"  {name:24s} OK  {got.tolist()}")


hot = 10 * BLOB_MEDIAN_E
cold = 1.0

# The cut is on the MEDIAN, so a rod with one hot point is not a blob but a
# cloud that is hot throughout is.
_check("median not max",
       [_cloud([cold] * 9 + [hot]), _cloud([hot] * 10)],
       [False, True])

# Zero-energy rows are padding: excluded from the median, and a cloud that is
# all padding is empty rather than degenerate.
_check("padding excluded",
       [_cloud([hot, hot, 0.0, 0.0]), _cloud([0.0] * 4)],
       [True, False])

# Straddling the cut. nanmedian takes the lower of the two middle elements for
# an even count, so pair each value with itself to keep the test on the cut.
_check("cut boundary",
       [_cloud([BLOB_MEDIAN_E * 0.99] * 4), _cloud([BLOB_MEDIAN_E * 1.01] * 4)],
       [False, True])

# The cases the removed Step-0 ratio check used to own. NaN never reaches the
# median (NaN > 0 is False), so without the explicit isfinite test the first
# cloud would read as a clean single-point shower.
_check("non-finite flagged",
       [_cloud([cold, float("nan")]), _cloud([cold, float("inf")]),
        _cloud([cold, cold])],
       [True, True, False])

print("\nall blob-flag tests passed")

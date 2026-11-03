"""CPU self-test of the mean-energy forms (symbolic/energy_sr.py) on toy rows with known answers."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import energy_sr, rings_sr, times_sr


def test_forms_have_the_numbers_they_claim():
    X = np.random.default_rng(0).normal(size=(10, 7))
    for form, n in energy_sr.FORMS.items():
        assert energy_sr.design(form, X).shape == (10, n), form


def test_fit_recovers_known_constants_and_skips_empty_rings():
    rng = np.random.default_rng(1)
    v = rng.uniform(0, 3, (40, 6))
    theta = np.array([1.0, -0.5, 0.2, -0.03, 0.4, 0.1])
    y = energy_sr.predict(theta, "c", v)
    n = np.ones(y.shape)
    n[:, 3] = 0
    y[:, 3] = np.nan
    assert np.allclose(energy_sr.fit_form("c", v, y, n), theta)


def test_ring_means_match_the_expanded_rows():
    x, k = np.arange(1.0, 9.0), 11
    theta = np.array([1.0, -0.5, 0.2])
    got = energy_sr.ring_means(theta, "a", x, k)
    assert got.shape == (times_sr.N_RING,)
    assert np.allclose(got, 1.0 - 0.5 * times_sr.LN_R + 0.2 * np.log(12.0))


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

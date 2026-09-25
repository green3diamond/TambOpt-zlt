"""Bootstrap and binomial intervals, and binning, at one standard deviation."""
import numpy as np

LEVEL = 0.6827


def bootstrap_ci(x, stat=np.median, n_boot=500, seed=0):
    """Percentile bootstrap interval of stat(x)."""
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    s = np.array([stat(x[i]) for i in idx])
    return tuple(np.percentile(s, [50 * (1 - LEVEL), 50 * (1 + LEVEL)]))


def paired_bootstrap_ci(a, b, stat=np.median, n_boot=500, seed=0):
    """Percentile bootstrap interval of stat(a) - stat(b), resampling events jointly."""
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    s = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    return tuple(np.percentile(s, [50 * (1 - LEVEL), 50 * (1 + LEVEL)]))


def wilson(k, n, z=1.0):
    """Wilson score interval for k successes out of n; nan where n is 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bin_index(x, edges):
    """Bin of each value, with the outer edges closed."""
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

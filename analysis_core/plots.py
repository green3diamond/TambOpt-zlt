"""Shared figure style, mountain footprint underlay and figure saving."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})

KERNEL, SURROGATE = "C0", "C1"


def footprint(ax, mountain, max_points=20000):
    """Grey mesh centroids under a map, East on x and North on y."""
    c = mountain.centroids_ENU
    step = max(1, len(c) // max_points)
    ax.scatter(c[::step, 0], c[::step, 1], s=0.3, c="0.85", zorder=0, rasterized=True)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")


def save(fig, out, name, log):
    path = os.path.join(out, "figures", f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)

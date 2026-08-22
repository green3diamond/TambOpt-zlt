"""3D point cloud of specific corpus rows, side by side.

Complements the two neighbours: `cached_showers.py` draws the LEADING N showers
of each species as a 2D heatmap; this draws NAMED rows in 3D, which is what you
want when a diagnostic has already told you which rows to look at.

Same visual conventions as `angle_grid_dual.py::_plot_grid_3d` -- native
(x, y, layer) axes, log-normed energy colour, marker size scaled by energy --
but each panel is titled with that row's own statistics rather than a grid
position, and the colour scale is PER PANEL by default: a shower inflated 1e9x
would otherwise flatten every other panel to one colour.

Run:
    python plots/showers/inspect_rows.py --ckpt <corpus.pt> --rows 573000 573463
    python plots/showers/inspect_rows.py --ckpt <corpus.pt> --rows 0 1 2 --shared-color
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection

import modules  # noqa: F401 — package import; keeps modules on the path
import showerdata
from modules.constants import N_PLANES, EDA_OUTPUTS_DIR


def row_stats(pts, e_prim):
    """The statistics that identify a mis-scaled shower, per `filter_by_plane_ratio`."""
    real = pts[pts[:, 3] > 0]
    if not real.size:
        return dict(n=0, med=0.0, tot=0.0, ratio=0.0, top1=0.0)
    e = real[:, 3]
    lay = np.clip(np.rint(real[:, 2]).astype(int), 0, N_PLANES - 1)
    plane_e = np.bincount(lay, weights=e, minlength=N_PLANES)
    return dict(n=len(real), med=float(np.median(e)), tot=float(e.sum()),
                ratio=float(plane_e.max() / max(e_prim, 1e-30)),
                top1=float(e.max() / max(e.sum(), 1e-30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=str, required=True, help="corpus .pt")
    ap.add_argument("--rows", type=int, nargs="+", required=True,
                    help="row indices to draw, in order")
    ap.add_argument("--stride", type=int, default=1,
                    help="plot every Nth point (thins dense muon clouds)")
    ap.add_argument("--shared-color", action="store_true",
                    help="one colour scale across panels; off by default because a "
                         "mis-scaled shower flattens every other panel")
    ap.add_argument("--out", type=str,
                    default=os.path.join(EDA_OUTPUTS_DIR, "inspect_rows.png"))
    args = ap.parse_args()

    clouds, stats = [], []
    for r in args.rows:
        sub = showerdata.load(args.ckpt, start=r, stop=r + 1)
        p = np.asarray(sub.points)[0]
        ep = float(np.asarray(sub.energies).reshape(-1)[0])
        st = row_stats(p, ep)
        st["row"], st["e_prim"] = r, ep
        clouds.append(p[p[:, 3] > 0][::args.stride])
        stats.append(st)
        print(f"[row {r}] points={st['n']:,}  E_prim={ep:.3g}  "
              f"median point={st['med']:.4g}  total={st['tot']:.4g}  "
              f"max-plane/E_prim={st['ratio']:.4g}  top-1 share={100*st['top1']:.2f}%")

    C = len(clouds)
    fig = plt.figure(figsize=(5.2 * C, 5.4))
    shared = None
    if args.shared_color:
        allp = np.concatenate([c for c in clouds if len(c)], axis=0)
        pos = allp[:, 3][allp[:, 3] > 0]
        shared = LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))

    for k, (pts, st) in enumerate(zip(clouds, stats)):
        ax = fig.add_subplot(1, C, k + 1, projection="3d")
        if len(pts):
            e = pts[:, 3]
            norm = shared or LogNorm(vmin=max(float(e[e > 0].min()), 1e-12),
                                     vmax=float(e.max()))
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=e, s=2 + 14 * (e / (e.max() + 1e-12)),
                       cmap="inferno", norm=norm, alpha=0.55, edgecolors="none")
            cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.55, pad=0.10)
            cb.set_label("cluster energy", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("layer", fontsize=8)
        ax.set_zlim(-0.5, N_PLANES - 0.5)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"row {st['row']}   n={st['n']:,}\n"
                     f"median point {st['med']:.3g}   total {st['tot']:.3g}\n"
                     f"max-plane/$E_{{prim}}$ = {st['ratio']:.3g}", fontsize=9)

    fig.suptitle(f"{os.path.basename(args.ckpt)} — native (x, y, layer); "
                 f"colour = cluster energy (log)"
                 + ("" if args.shared_color else ", scaled PER PANEL"), fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()

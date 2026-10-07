"""Run 19: 3 draws of each of the 9 training layout types on the 12 km plane -> zoo_layouts.png (all on the plane)."""
import os, sys
sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import modules.constants as C
from modules.geometry import load_tr_mountain
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS
from modules.data.dataset_builder import _batch_layout_rng
OUT = os.path.dirname(os.path.abspath(__file__))
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
fig, axs = plt.subplots(3, len(_STRATEGIES), figsize=(3.2 * len(_STRATEGIES), 8))
for j, (name, fn, kw) in enumerate(_STRATEGIES):
    for i in range(3):
        e, n = _STRATEGY_FNS[fn](m, n_det=C.N_DETECTORS, rng=_batch_layout_rng(0, j, i), **kw)
        e, n = e.numpy(), n.numpy()
        d = np.sqrt(((mesh[None, :, :2] - np.stack([e, n], 1)[:, None]) ** 2).sum(-1)).min(1)
        assert e.size == 100 and d.max() < 300, (name, d.max())
        ax = axs[i, j]; ax.scatter(mesh[:, 0], mesh[:, 1], s=0.5, c="#e8e6df", lw=0); ax.scatter(e, n, s=5, c="#2a78d6")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{j}: {name}", fontsize=8)
fig.suptitle(f"Run 19 training layouts on the 12 km plane (E {m.east_lo:.0f}..{m.east_hi:.0f}, N {m.n_min:.0f}..{m.n_max:.0f} m)")
fig.tight_layout(); fig.savefig(f"{OUT}/zoo_layouts.png", dpi=90); print("[ok] all 9 types on the plane")

"""Layout-zoo checks: (1) 3 draws of each new strategy on the run 18 plane -> zoo_layouts.png;
(2) build_training_pairs(strategies=[0, 5]) on 360 events reproduces the existing strategy-0 rows."""
import os, sys
sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import modules.constants as C
from modules.geometry import load_tr_mountain, SurfaceUpMap
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS
from modules.data.dataset_builder import build_training_pairs, _batch_layout_rng
OUT = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)
fig, axs = plt.subplots(3, 4, figsize=(16, 12))
for j, s in enumerate(range(5, 9)):
    name, fn, kw = _STRATEGIES[s]
    for i in range(3):
        e, n = _STRATEGY_FNS[fn](m, n_det=C.N_DETECTORS, rng=_batch_layout_rng(0, s, i), **kw)
        e, n = e.numpy(), n.numpy()
        d = np.sqrt(((mesh[None, :, :2] - np.stack([e, n], 1)[:, None]) ** 2).sum(-1)).min(1)
        assert e.size == 100 and np.isfinite(e).all() and d.max() < 200, (name, e.size, d.max())
        ax = axs[i, j]; ax.scatter(mesh[:, 0], mesh[:, 1], s=1, c="#e8e6df", lw=0); ax.scatter(e, n, s=8, c="#2a78d6")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{s}: {name} (batch {i})")
fig.tight_layout(); fig.savefig(f"{OUT}/zoo_layouts.png", dpi=90); print("[ok] generators, max dist to mesh < 200 m")

surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
prim, xy, E, T, st, sp, blob = build_training_pairs(m, surface, C.DUAL_SHOWER_CACHE_PATH, batch_size=C.BATCH_SIZE,
                                                    max_showers=360, seed=0, device=dev, verbose=False,
                                                    strategies=[0, 5])
n = prim.shape[0] // 2
D = C.TRAINING_DATASET_FOLDER
xy0, E0, T0 = (torch.load(f"{D}/{f}.pt") for f in ("xy", "E", "T"))
k = n // 3            # species 0 rows of strategy 0
print("strategy ids", st.unique().tolist(), "rows/block", n)
print("xy  max |diff|", float((xy[:k] - xy0[:k]).abs().max()))
print("E   max |diff|", float((torch.log1p(E[:k]) - E0[:k]).abs().max()), " T max |diff|", float((T[:k] - T0[:k]).abs().max()))
assert st[:n].eq(0).all() and st[n:].eq(5).all()

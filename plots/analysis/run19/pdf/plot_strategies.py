"""One draw of every run 19 training strategy (batch 0, Stage 1's seed 0) on the run 19 plane."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import modules.constants as C
from modules.geometry import load_tr_mountain
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS
from modules.data.dataset_builder import _batch_layout_rng

m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
mesh = np.asarray(m.centroids_ENU)[:, :2] / 1e3
fig, axs = plt.subplots(3, 4, figsize=(21, 14))
for ax in axs.flat:
    ax.axis("off")
for ax, (s_idx, (name, fn_name, kw)) in zip(axs.flat, enumerate(_STRATEGIES)):
    ax.axis("on")
    e, n = _STRATEGY_FNS[fn_name](m, n_det=C.N_DETECTORS, rng=_batch_layout_rng(0, s_idx, 0), **kw)
    e, n = np.asarray(e) / 1e3, np.asarray(n) / 1e3
    ax.scatter(*mesh.T, s=0.4, color="#e4e3dd", lw=0)
    ax.scatter(e, n, s=10, color="#2a78d6", lw=0)
    xy = np.stack([e, n], 1) * 1e3
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    ax.set_title(f"{s_idx}: {name}  ({fn_name}{', ' + ', '.join(f'{k}={v:g}' for k, v in kw.items()) if kw else ''})\n"
                 f"median NN spacing {np.median(d.min(1)):.0f} m", fontsize=11)
    ax.set_xlim(mesh[:, 0].min() - 0.3, mesh[:, 0].max() + 0.3); ax.set_ylim(mesh[:, 1].min() - 0.3, mesh[:, 1].max() + 0.3)
    ax.set_aspect("equal"); ax.tick_params(labelsize=8); ax.set_xlabel("East [km]", fontsize=9); ax.set_ylabel("North [km]", fontsize=9)
fig.suptitle("Run 19 training layouts: one draw per strategy (100 detectors; each batch of 60 showers draws a new layout)", fontsize=15)
fig.subplots_adjust(left=0.03, right=0.995, top=0.92, bottom=0.04, wspace=0.18, hspace=0.38)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_strategies_batch0.png")
fig.savefig(out, dpi=80, facecolor="white"); print("[write]", out)

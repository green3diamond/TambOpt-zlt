"""Smoke test for 04_optimize_lbfgs_kernel.py --pool main on CPU: loads 4 pool
events, checks that sub-batch backward reproduces the full-batch U and gradient."""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "kopt", os.path.join(_REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

from modules.constants import GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, N_PLANES  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.layouts import LearnableXY  # noqa: E402

mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                            east_entry=K.EAST_ENTRY, layer_east_dx=K.LAYER_EAST_DX,
                            n_planes=N_PLANES)
surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(K.DEVICE)
pool = K.MainPool.__new__(K.MainPool)
pool.meta = K._load_corpus_metadata(mountain, K.DUAL_SHOWER_CACHE_PATH, K.MAX_SHOWERS)
pool.n, pool.bw = pool.meta.k_sp, None
print("pool events", pool.n)
batch = pool.load(100, 104)
# Widths as the precompute would store them (float16 round trip).
batch = (batch[0], [K.adaptive_bandwidths(c, K.KERNEL_INTERP_K, K.SIGMA_SPATIAL)
                    .half().float() for c in batch[0]], batch[2])
recon = K.load_recon(K.RECON_FOLDER + "_deepsets", K.DEVICE)
e, n = K.init_layout(mountain, "grid", K.DEVICE)

res = {}
for chunk in (4, 2, 1):
    xy = LearnableXY(e.clone(), n.clone(), device=str(K.DEVICE))
    tot = K.pool_utility(xy, batch, surface, recon, chunk, backward=True)
    g = torch.cat([xy.x.grad, xy.y.grad])
    res[chunk] = (tot["U"], g)
    print(f"chunk={chunk}  U={tot['U']:.6f}  r={tot['r_mean']:.3f}  |grad|={g.norm():.4e}")
for c in (2, 1):
    dU = abs(res[c][0] - res[4][0])
    dg = (res[c][1] - res[4][1]).norm() / res[4][1].norm().clamp(min=1e-30)
    print(f"chunk {c} vs 4: |dU|={dU:.2e}  rel grad diff={dg:.2e}")
    assert dU < 1e-4 and dg < 1e-4
print("OK")

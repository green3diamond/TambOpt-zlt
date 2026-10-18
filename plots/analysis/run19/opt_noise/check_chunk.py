"""Does the Stage 4 sub-batch size (--chunk) change U or its gradient? Same 4,096 pool showers and layout, closure =
pool_utility(backward=True) with sub-batches of 32 ... 512: U, gradient (relative difference and cosine vs 32), time per
closure and peak GPU memory. Readouts: no background + kNN gate (exactly comparable), 1 us background + gate (the noise
draws depend on the sub-batch split, so only statistically comparable).

    python check_chunk.py <layout.pt> [chunks ...]
"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
R = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k"
os.environ.update(TAMBOPT_UTILITY="two_tier", TAMBOPT_TWO_TIER_LAMBDA="0.3",
                  TAMBOPT_TWO_TIER_CALIB=R + "/two_tier_calibration_kernel_5deg.json")
_s = importlib.util.spec_from_file_location("s4", os.path.join(REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
S4 = importlib.util.module_from_spec(_s); _s.loader.exec_module(S4)
from modules.layouts import LearnableXY  # noqa: E402

path = sys.argv[1]
chunks = [int(c) for c in sys.argv[2:]] or [32, 64, 128, 256, 512]
dev = S4.DEVICE
L = torch.load(path, map_location="cpu")
mountain = S4.load_tr_mountain(S4.GEOMETRY_PATH_RESOLVED, S4.GEOMETRY_GROUP, S4.DET_KEY, east_entry=S4.EAST_ENTRY,
                               layer_east_dx=S4.LAYER_EAST_DX, n_planes=S4.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
recon = S4.load_recon(R + "/test_v6_run_03_recentered_deepsets", dev)
pool = S4.MainPool(mountain)
batch = pool.load(1000, 1000 + 4096)
print(f"layout {path}, {batch[2].shape[0]} showers", flush=True)
for W in (0.0, 1000.0):
    resp = dict(surrogate=None, p_min=0.5, window_ns=W, gate="knn23", t_fallback_us=27.7)
    ref = None
    for ch in chunks:
        xy = LearnableXY(torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev), device=str(dev)).to(dev)
        g = torch.Generator(device=dev); g.manual_seed(7)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        try:
            tot = S4.pool_utility(xy, batch, surface, recon, ch, backward=True, resp=resp, gen=g)
        except torch.cuda.OutOfMemoryError:
            print(f"  W={W:g} chunk {ch:4d}: CUDA out of memory", flush=True); torch.cuda.empty_cache(); continue
        torch.cuda.synchronize(); dt = time.time() - t0
        grad = torch.cat([p.grad.flatten() for p in xy.parameters()])
        if ref is None:
            ref = (tot["U"], grad.clone())
        rel = float((grad - ref[1]).norm() / ref[1].norm().clamp(min=1e-30))
        cos = float(torch.nn.functional.cosine_similarity(grad, ref[1], dim=0))
        print(f"  W={W:g} chunk {ch:4d}: U={tot['U']:+.6f} (dU={tot['U'] - ref[0]:+.2e})  |grad|={float(grad.norm()):.4e}  "
              f"rel diff vs {chunks[0]}={rel:.2e}  cos={cos:.6f}  {dt:6.1f} s  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
              flush=True)

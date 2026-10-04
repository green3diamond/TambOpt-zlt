"""Smoke test of the run-18 shower-frame code paths on run 17's data (GPU, ~5 min).

The constants point at run 18, whose corpus does not exist before Stage 0, so this
overrides the corpus / mesh / recon paths with run 17's and exercises:
  1. Stage 1: build_training_pairs on a few hundred events (shower-frame placement,
     widths per chunk, frames per sub-batch) -> finite E/T, lit fractions;
  2. widths on shower-frame clouds respect KERNEL_INTERP_MAX (50 m);
  3. load_events -> PlacedClouds.frames -> KernelDualLabels(differentiable) -> recon
     utility and its gradient w.r.t. the layout (finite, non-zero);
  4. Stage 4's MainPool.load + pool_utility (frames sliced with the events).

    python tests/smoke_run18_kernel.py
"""
import importlib.util
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

import modules.constants as C  # noqa: E402
from modules.data.dataset_builder import build_training_pairs  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne  # noqa: E402
from modules.layouts import LearnableXY  # noqa: E402
from modules.optimize import utility_of_xy  # noqa: E402
from modules.showers.kernel import adaptive_bandwidths  # noqa: E402
import plots.layouts.true_utility as TU  # noqa: E402

R17 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k"
CORPUS = f"{R17}/v6_run_00/cashed_showers_tau_dual.pt"
HELDOUT = f"{R17}/v6_run_00/cashed_showers_tau_dual_heldout.pt"
RECON = f"{R17}/test_v6_run_03_recentered_deepsets_kernel100"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"KERNEL_FRAME={C.KERNEL_FRAME} T_INTERP={C.KERNEL_T_INTERP} CIRCULAR={C.KERNEL_CIRCULAR} "
      f"BW_MAX={C.KERNEL_INTERP_MAX} device={dev}")

mountain = load_tr_mountain(f"{R17}/inputs/plane45.h5", "plane45", "detector1", east_entry=C.EAST_ENTRY,
                            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)

# 1. Stage 1 on 120 events per species
t0 = time.time()
prim, xy, E, T, strat, species, blob = build_training_pairs(
    mountain, surface, CORPUS, batch_size=60, max_showers=360, seed=0, device=dev, verbose=True,
    load_chunk=120)
assert torch.isfinite(E).all() and torch.isfinite(T).all()
lit = (E > torch.log1p(torch.tensor(1.0))).float().mean()
print(f"[1] Stage 1: {E.shape[0]} rows in {time.time() - t0:.0f}s; detectors >= 1 count {lit:.1%}; "
      f"T > 0 {(T > 0).float().mean():.1%}; E (log1p) mean {E.mean():.3f}")

# 2. widths on shower-frame clouds
species_clouds, B, n_pairs, primary_batch = None, None, None, None
TU.HELDOUT_SHOWER_CACHE_PATH = HELDOUT
TU.HELDOUT_POSITIONS_PATH = HELDOUT.replace(".pt", "_positions.pt")
species_clouds, B, n_pairs, primary_batch = TU.load_events(64, dev, mountain)
assert species_clouds.frames is not None and species_clouds.frames.shape == (B, 12)
for s, c in enumerate(species_clouds):
    h = adaptive_bandwidths(c.to(dev), C.KERNEL_INTERP_K, C.SIGMA_SPATIAL)
    real = c[..., 3] > 0
    print(f"[2] {C.SPECIES_NAMES[s]}: widths median {h[real.to(dev)].median():.1f} m, max {h.max():.1f} m; "
          f"|t2| median {c[..., 1][real].abs().median():.0f} m vs |t1| {c[..., 0][real].abs().median():.0f} m")
    assert float(h.max()) <= C.KERNEL_INTERP_MAX + 1e-3

# 3. held-out kernel -> recon utility and gradient
spec = importlib.util.spec_from_file_location("k04", os.path.join(_ROOT, "scripts/04_optimize_lbfgs_kernel.py"))
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
recon = K.load_recon(RECON, dev)
kern = TU.KernelDualLabels(species_clouds, surface, dev, chunk=32, differentiable=True,
                           expected_primary=primary_batch)
torch.manual_seed(0)
e0, n0 = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme="grid")
e0, n0 = project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())
xy_mod = LearnableXY(e0.to(dev), n0.to(dev), device=str(dev)).to(dev)
x, y = xy_mod()
U, r, parts = utility_of_xy(x, y, primary_batch, kern, recon)
(-U).backward()
g = torch.cat([xy_mod.x.grad, xy_mod.y.grad])
print(f"[3] held-out kernel utility U={float(U):.3f}, |grad|={g.norm():.3e}, finite={bool(torch.isfinite(g).all())}")
assert torch.isfinite(g).all() and g.norm() > 0

# 4. Stage 4 pool
K.DUAL_SHOWER_CACHE_PATH = CORPUS
pool = K.MainPool.__new__(K.MainPool)
pool.meta = K._load_corpus_metadata(mountain, CORPUS, 0)
pool.n, pool.bw = pool.meta.k_sp, None
batch = pool.load(100, 164)
batch = (batch[0], [adaptive_bandwidths(c, C.KERNEL_INTERP_K, C.SIGMA_SPATIAL) for c in batch[0]],
         batch[2], batch[3])
xy_mod = LearnableXY(e0.to(dev), n0.to(dev), device=str(dev)).to(dev)
tot = K.pool_utility(xy_mod, batch, surface, recon, 32, backward=True)
g = torch.cat([xy_mod.x.grad, xy_mod.y.grad])
print(f"[4] pool utility U={tot['U']:.3f}, |grad|={g.norm():.3e}")
assert torch.isfinite(g).all() and g.norm() > 0
print("smoke ok")

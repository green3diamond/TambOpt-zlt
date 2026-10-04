"""Average area-normalisation factor (sigma/h)^2 of the interpolated kernel, per species,
over real cloud points of the first N Stage-4 pool events (widths from the k=8 cache)."""
import sys
import numpy as np
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")
import showerdata  # noqa: E402
from modules.constants import DUAL_SHOWER_CACHE_PATH, SIGMA_SPATIAL, SPECIES_NAMES, TRAINING_DATASET_FOLDER  # noqa: E402
from modules.data.dataset_builder import _load_corpus_metadata  # noqa: E402,F401

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
per_sp = showerdata.load_inc_particles(DUAL_SHOWER_CACHE_PATH).pdg.shape[0] // len(SPECIES_NAMES)
for s, name in enumerate(SPECIES_NAMES):
    h = np.load(f"{TRAINING_DATASET_FOLDER}/kernel_bw_k8_{name}.npy", mmap_mode="r")[:N].astype(np.float32)
    e = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=s * per_sp, stop=s * per_sp + N).points[..., 3]
    real = e > 0
    f = (SIGMA_SPATIAL / h[:, :e.shape[1]][real]) ** 2
    w = e[real].astype(np.float64)
    print(f"{name}: points {real.sum()}  mean factor {f.mean():.3f}  median {np.median(f):.3f}  "
          f"energy-weighted mean {np.sum(f * w) / w.sum():.3f}  "
          f"mean over widened points only {f[f < 0.999].mean():.3f}  frac widened {(f < 0.999).mean():.3f}")

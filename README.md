# TambOpt — TAMBO Detector Optimization & Simulation Suite

TambOpt is a differentiable end-to-end pipeline for optimizing cosmic ray detector layouts for the TAMBO Observatory. This repo is the v6 pipeline (the only maintained generation) — it combines flow-matching shower generation, a differentiable plane-aware detector-response kernel, DeepSets forward-surrogate and reconstruction networks, and gradient/global optimization of detector `(x, y)` positions on the Malata mountain mesh.

Earlier generations (`detector_optimization/` through `_v5`) are on the `legacy-full-repo` git branch. How the current pipeline works: [`docs/THEORY.md`](docs/THEORY.md).

## Layout

```
modules/          # importable library, six domain subpackages:
                  #   geometry/ layouts/ showers/ surrogates/ data/ optimize/
                  #   each re-exports its public names, so:
                  #     from modules.geometry import load_tr_mountain, SurfaceUpMap
scripts/          # 00-04 pipeline stages: generate data → build dataset → train surrogate → train recon → optimize layout
slurm/            # SLURM sbatch wrappers; env.sh is the shared preamble they all source
plots/            # figures + evaluation, grouped: lib/ showers/ training/ layouts/
decay_locations/  # CORSIKA observation-plane source + Julia tau-injection pipeline
notebooks/        # EDA / diagnostic notebooks
docs/             # THEORY.md (architecture reference) and dev diary
data/             # mountain mesh geometry (malata.h5)
.pycache/         # all bytecode, gitignored (see below)
```

Bytecode is kept out of the source tree, so no `__pycache__/` dirs appear beside
the sources. It is set in exactly two places:

- **`slurm/env.sh`** — the shared preamble every SLURM wrapper sources
  (`source slurm/env.sh`). It loads the conda env and exports
  `PYTHONPYCACHEPREFIX="$REPO_ROOT/.pycache"`. Change the location here.
- **`_pathfix.py`** — sets the same value as a fallback, for notebooks and
  interactive sessions that never source `env.sh`.

For an interactive shell that goes through neither:

```bash
source slurm/env.sh          # from the repo root
```

Note the prefix is global to the interpreter, so third-party packages cache
there too — `.pycache/` runs to a few tens of MB. It is gitignored and safe to
delete at any time.

See `docs/THEORY.md` for the full pipeline architecture and the reasoning behind the DeepSets surrogate/reconstruction design.

## Setup

Python 3.13 with PyTorch 2.9 (CUDA GPU), numpy 2.3, scipy 1.16, h5py 3.15, scikit-learn 1.7, pandas 2.3, matplotlib 3.10, tqdm. There is no lock file; these are the versions the results were produced with.

```bash
conda create -n multiproc_env python=3.13
conda activate multiproc_env
pip install torch numpy scipy h5py scikit-learn pandas matplotlib tqdm
source slurm/env.sh     # bytecode prefix; on the original cluster it also loads the module and conda env
```

`slurm/env.sh` and the SLURM wrappers assume the original cluster (modules, partitions, GPU constraints, absolute paths under `/n/...`); on another machine, edit them or run the scripts directly.

### Inputs

All paths are set in `modules/constants.py` and overridden by environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `TAMBOPT_RUN_LOCATION` | run folder; every stage reads and writes below it | run 19 folder |
| `TAMBOPT_GEOMETRY_PATH`, `TAMBOPT_GEOMETRY_GROUP` | detector-region mesh (HDF5, group name) | run 19 plane mesh |
| `TAMBOPT_TAU_PATH` | tau decay primaries (HDF5) | run 19 plane taus |
| `TAMBOPT_SITE` | `plane12` or `malata` | `plane12` |
| `TAMBOPT_UTILITY` | Stage 4 utility (see `docs/THEORY.md`) | see `constants.py` |

Only the Malata mesh (`data/malata.h5`, group `malata`) is in the repo. The tau decay files come from `decay_locations/` (Julia injection pipeline) or `scripts/17_make_plane_inputs.py` (plane), and the shower generator needs the per-species AllShowers checkpoints, which are not distributed with the repo. `slurm/run20_guard.sh` shows a complete set of variables for the Malata run.

## Running the pipeline

Stages, in order. Each reads the previous stage's output from `TAMBOPT_RUN_LOCATION`, and Stages 1-4 resume after an interruption.

| Stage | Script | Output |
|---|---|---|
| 0 | `scripts/00_generate_data_dual_species.py` | paired per-species shower corpus |
| 1 | `scripts/01_build_dataset_northeast.py` | kernel labels (E, T) for every shower under a zoo of layouts |
| 2 | `scripts/02_train_fnn_deepsets.py --species {electron,muon,photon}` | detector-response surrogate `fnn_<species>.pt` (optional; only needed for `--response surrogate`) |
| 3 | `scripts/03_train_recon_deepsets.py` | frozen reconstruction `recon.pt` |
| 4 | `scripts/04_optimize_lbfgs_kernel.py` | `layout_best.pt` and the optimisation log |

Run `python scripts/04_optimize_lbfgs_kernel.py --precompute-bw <species index>` once per run, before Stage 4, to cache the kernel widths. Each script's docstring lists its options. On SLURM, `slurm/run19_pipeline.sh {gen,build,recon,tests}` wraps Stages 0, 1 and 3, and `slurm/kernel_opt_run19.sh` wraps Stage 4. The Malata chain is submitted with:

```bash
bash slurm/run20_submit.sh <filter_job>     # <filter_job>: SLURM job id of scripts/20_filter_corpus.py
```

This submits the chain with dependencies (build, recon, count fine-tune, five Stage 4 runs, validation, tables). See `slurm/` for the individual evaluation and plotting wrappers.

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
AllShowers-flow-matching-surrogate/
                  # flow-matching shower generator (AllShowers), with
                  # PointCountFM/ (points-per-layer model) inside it
.pycache/         # all bytecode, gitignored (see below)
```

## Flow-matching shower surrogate

[`AllShowers-flow-matching-surrogate/`](AllShowers-flow-matching-surrogate/README.md)
holds the generative shower models, copied in from the standalone
[AllShowers](https://github.com/hamzahanif2210/AllShowers) and
[PointCountFM](https://github.com/hamzahanif2210/PointCountFM) forks. Their
git history and training outputs were not copied. The two models run as a chain:

1. **PointCountFM**
   ([`AllShowers-flow-matching-surrogate/PointCountFM/`](AllShowers-flow-matching-surrogate/PointCountFM/README.md))
   generates the number of points per layer from the incident particle
   type, energy and direction. It writes them into a condition file.
2. **AllShowers** ([`AllShowers-flow-matching-surrogate/allshowers/`](AllShowers-flow-matching-surrogate/README.md))
   is a transformer conditional flow-matching model. It reads that condition
   file and generates the shower point clouds.

Both are self-contained and run from their own folder, with no `pip install`.
Each has its own `mkresultdir.py` for Slurm training runs. Their `results/`,
`plots/` and `logs/` outputs are gitignored. See each folder's README for
setup, training and generation.

This folder is for training and developing the models. The pipeline does not
use it yet. `modules/showers/generate.py` (Step 0) still calls the generator
and checkpoints in the external `TAMBO-opt` repo
(`/n/home05/zdimitrov/tambo/TAMBO-opt`).

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

## Running the pipeline

```
sbatch slurm/run_all_script_batch.sh
```

drives the full `00` → `04` pipeline end-to-end, with `pipeline_status.json` checkpointing so already-completed steps are skipped on re-submission. See `slurm/` for the individual evaluation/plotting job wrappers.

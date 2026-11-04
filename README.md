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

## Running the pipeline

```
bash slurm/run20_submit.sh <filter_job>     # submits the run 20 chain with dependencies
```

chains the current run's stages (Stage 4 is `scripts/04_optimize_lbfgs_kernel.py`). See `slurm/` for the individual evaluation/plotting job wrappers.

### Paper 1 optimizer tests

```
bash slurm/paper1_opt_submit.sh [steps]     # 16 Stage 4 runs + 2 scoring jobs, default 20,000 steps
```

Original utility; {Adam + L-BFGS, SGD} × {centre, uniform grid} × {kernel, surrogate} × {plane (run 19), Malata (run 20)}.
Needs `<run>/diagnostics/paper1/ugrid.pt` (the uniform-grid start) in both run folders. Results: `_p1` Stage 4 folders,
scores in `tmp_rings/optres/p1_scores_<site>.json`.

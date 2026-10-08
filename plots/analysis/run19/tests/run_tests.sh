#!/bin/bash
# Run 19 layout tests (called by `sbatch slurm/run19_pipeline.sh tests`, which sets up the env, the GPU and the
# run 19 guard). Layouts -> one scoring pass with the run's recon -> detector-count grid -> plots/tables.
# Outputs in <RUN_LOCATION>/diagnostics/tests/ (and tests/detcount/). Re-running skips finished steps.
set -e
export MPLCONFIGDIR=/tmp/mplcache_$USER
H=plots/analysis/run19/tests
T=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION + "/diagnostics/tests")')
mkdir -p "$T"
[ -s "$T/layouts.txt" ] || python -u $H/make_layouts.py
[ -s "$T/pairs.npz" ] || python -u $H/score_all.py
[ -s "$T/detcount/detector_count_grid.json" ] || python -u scripts/diagnostics/detector_count_grid.py --out "$T/detcount"
python -u $H/analyze.py

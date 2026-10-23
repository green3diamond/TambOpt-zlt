#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=96g
#SBATCH --time=0-12:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J run19_lit
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19 layout tests rescored with lit = E > 0.001 (the kNN gate's hit) instead of >= 1 count: same layouts, recon
# and 5,000 showers -> tests/scores_lit0p001.json, pairs_lit0p001.npz, tests/detcount_lit0p001/, analyze plots _lit0p001.
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
H=plots/analysis/run19/tests
T=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION + "/diagnostics/tests")')
( [ -s "$T/pairs_lit0p001.npz" ] || python -u $H/score_all.py --tag lit0p001 --lit-min 0.001 ) &&
( [ -s "$T/detcount_lit0p001/detector_count_grid.json" ] || python -u scripts/diagnostics/detector_count_grid.py --n-events 5000 --lit-min 0.001 --out "$T/detcount_lit0p001" ) &&
python -u $H/analyze.py --tag lit0p001
rc=$?; notify_end $rc; exit $rc

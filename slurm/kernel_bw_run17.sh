#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-2
#SBATCH -J kernel_bw_run17
#SBATCH -o slurm_logs/slurm-%A_%a-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Interpolated-kernel widths for Stage 4's event pool, one species per array
# task. Needed once before `04_optimize_lbfgs_kernel.py --pool main`. Resumable.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION as R; sys.exit(0 if "17_plane45_interp_hurdle_100k" in R else 1)' \
    || { echo "constants.py does not point to run 17"; exit 1; }

python -u scripts/04_optimize_lbfgs_kernel.py --precompute-bw "$SLURM_ARRAY_TASK_ID"

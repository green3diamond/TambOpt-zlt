#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=128g
#SBATCH --time=1-00:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2
#SBATCH -J kernel_bw_run19
#SBATCH -o slurm_logs/slurm-%A_%a-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Shower-frame kernel widths (k = 8, capped at 50 m) for Stage 4's event pool, one species per array
# task. Needs only the Stage 0 corpus; written into the Stage 1 folder. Resumable.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh
mkdir -p "$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.TRAINING_DATASET_FOLDER)')"
python -u scripts/04_optimize_lbfgs_kernel.py --precompute-bw "$SLURM_ARRAY_TASK_ID"

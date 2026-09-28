#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J stage1_interp_smoke
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Step-1 smoke test with the run's kernel; writes nothing into the run folder.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/diagnostics/stage1_interp_smoke.py "$@"

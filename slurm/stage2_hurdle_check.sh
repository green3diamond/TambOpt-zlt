#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J stage2_hurdle_check
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Baseline mean-only vs lit/dark hurdle Stage 2 on run 14 (sigma=10). Args pass through.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/diagnostics/stage2_hurdle_check.py "$@"

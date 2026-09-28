#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J interp_kernel_check
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Point spacing and label zero-fractions, fixed vs interpolated kernel. Args pass through.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/diagnostics/interp_kernel_check.py "$@"

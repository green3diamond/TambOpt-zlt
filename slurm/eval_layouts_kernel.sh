#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J eval_layouts_kernel
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Score saved layouts against the kernel on held-out events. Args pass through.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u scripts/diagnostics/eval_layouts_kernel.py "$@"

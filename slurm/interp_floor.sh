#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=70g
#SBATCH --time=04:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J interp_floor
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Aleatoric floor with the fixed or interpolated kernel. Args pass through, e.g.
#   sbatch slurm/interp_floor.sh --n-prim 64 --m-real 32 --interp-k 8 --out slurm_logs/floor_k8.json
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
python -u plots/training/aleatoric_floor.py "$@"

#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J ensemble_with_upr
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# One-chain ensemble run with an acceptance term lam*mean(r) added to U.
# Usage: sbatch slurm/ensemble_with_upr.sh <lam> <opt_suffix>
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
LAM="${1:?lam}"; SUFFIX="${2:?opt_suffix}"
python -u scripts/diagnostics/ensemble_with_upr.py --lam "$LAM" -- \
    --chains 1 --schemes grid --adam-epochs 5000 --lbfgs-iters 500 \
    --opt_suffix "$SUFFIX"

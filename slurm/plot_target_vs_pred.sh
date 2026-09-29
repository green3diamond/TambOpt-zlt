#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=150g
#SBATCH --time=03:00:00
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J plot_target_vs_pred
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Stage-2/3 diagnostic figures for whichever run RUN_LOCATION points at:
# --dual = fnn_<species>_target_vs_pred/_conditional.png, recon_target_vs_pred/
# _conditional.png, plus the recon-on-kernel pair (*_kernel.png). Args pass through.
source slurm/env.sh
python -u plots/training/02_nn_target_vs_pred.py "$@"

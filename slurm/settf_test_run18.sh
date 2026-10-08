#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=20:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH -J settf_test
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Attention recon (SetTransformerRecon, --model settf) on run 18's zoo data, the same setup as the wd 0.03 run of
# wd_sweep_run18.sh (AdamW wd 0.03, 80 epochs, patience 30, no L-BFGS) so the val curves compare directly.
#   sbatch slurm/settf_test_run18.sh [epochs] [suffix]     (smoke: 1 zoo_settf_smoke); input mode reltime1 (reference: earliest timed detector with >= 1 count)
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
export TAMBOPT_DATASET_TAG=zoo
export MPLCONFIGDIR=/tmp/mplcache_$USER
python -u scripts/03_train_recon_deepsets.py --model settf --recon-input kernel --input-mode reltime1 --weight-decay 0.03 \
    --epochs ${1:-80} --adam-patience 30 --lbfgs-iters 0 --out-suffix ${2:-zoo_settf_rt1}

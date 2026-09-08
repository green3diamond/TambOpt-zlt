#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -J recon_diag
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
source slurm/env.sh
python -u plots/training/03_recon_diagnostics.py \
    --lbfgs-log "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/11_750k_max_100k_LBFGS/test_v6_run_03_recentered_deepsets/recon_lbfgs_iter_log_scraped.json"

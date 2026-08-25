#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=40g
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J eval_u_vs_e_08
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

source slurm/env.sh

# Same as run_eval_utility_vs_energy.sh, retargeted at the 08_after_refactoring
# lbfgs_ensemble full-corpus runs. Writes utility_vs_energy.png into each run dir.
B="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/08_after_refactoring"
python -u plots/layouts/utility_vs_energy.py --run-dir \
    "$B/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_center" \
    "$B/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_grid"

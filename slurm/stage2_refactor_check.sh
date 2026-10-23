#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=48g
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J stage2_refactor_check
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
#
# Equivalence check for the NLL removal: the mean-only trainer should reproduce
# run 14's first epochs, because run 14 already trained mean-only in all but
# name (--nll-warmup-frac 1.0 => the warm-up WAS the whole schedule).
#
# Baseline to compare against, run 14 electron
# (slurm_logs/slurm-47742193-run_all_script_batch.out):
#   ep 1  train=0.9775 (E=0.9906 T=0.9645)  val=0.9235 (E=0.9447 T=0.9023)
#   ep 2  train=0.8475 (E=0.9199 T=0.7751)  val=0.7591 (E=0.8511 T=0.6671)
#   ep 3  train=0.7549 (E=0.8630 T=0.6468)  val=0.7281 (E=0.8292 T=0.6270)
#   ep 4  train=0.7305 (E=0.8454 T=0.6156)  val=0.7059 (E=0.8129 T=0.5990)
#   ep 5  train=0.7181 (E=0.8352 T=0.6010)  val=0.7098 (E=0.8143 T=0.6053)
#
# --epochs 100, NOT 5, is deliberate and load-bearing: OneCycleLR sizes
# total_steps from the epoch count, so a 5-epoch run would anneal ~20x faster
# and its epoch-1..5 learning rates would not be run 14's. Run the same 100-epoch
# schedule and read the first few epochs off it; the 30 min wall clock ends the
# job long before it finishes, which is fine -- this is a comparison, not a run.
#
# --out-suffix keeps it off the production fnn_electron.pt. --lbfgs-iters 0
# skips Phase 2, which this check never reaches.
#
# Exact agreement is not expected. The decoder's final Linear is now (2, hidden)
# instead of (4, hidden): rows 0-1 draw the same values (row-major fill, same
# fan_in) so the mean head starts identical, but the smaller draw consumes fewer
# RNG values, which shifts the per-batch permutation-augmentation stream. That
# augmentation is a no-op on a permutation-equivariant model, so the losses
# should still land on top of each other.

source slurm/env.sh

python -u scripts/02_train_fnn_deepsets.py \
    --species electron \
    --epochs 100 \
    --lbfgs-iters 0 \
    --out-suffix _meanonly_check

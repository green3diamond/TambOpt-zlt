#!/bin/bash
#SBATCH -p test
#SBATCH --mem=90g
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH -J stage2_zero_diag
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
#
# Part A of the sigma=10 Stage 2 investigation: measure the target collapse
# before changing any constant or adding any head.
#
# CPU, not GPU. Measurements (1)-(3) are pure tensor reductions and kernel
# evaluations with no training; (4) is one forward pass over the val split and
# the script's DEVICE falls back to CPU on its own. Nothing here benefits from
# a GPU and gpu_test's 64 GB/GPU cap is tighter than what loading run 13's
# labels needs.
#
# Memory: run 13's E.pt and T.pt are 450 MB each but run 13 is the FULL corpus
# (28.6M val rows in its training log vs run 14's 3.75M), so measure_labels
# holds several multi-GB float tensors at once. 90 GB is deliberate headroom
# over the 35.9 GB that the comparable recon-kernel plot job actually used --
# an OOM here costs a resubmit, and `test` is not contended.
#
# Time: the threshold sweep is the only slow part -- 9 thresholds x 600 showers
# x however many strategies _STRATEGIES defines, each a (B, P, n_det) kernel
# evaluation on CPU. 3 h is generous; shrink --sweep-showers if it runs long.

source slurm/env.sh

python -u scripts/diagnostics/stage2_zero_inflation.py \
    --json "slurm_logs/stage2_zero_diag_${SLURM_JOB_ID}.json"

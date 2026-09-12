#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=20:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J continue_ensemble_adam
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Continue ONE scheme's ensemble optimization from an existing layout_best.pt,
# in place -- for resuming with more Adam/L-BFGS budget after a completed run,
# rather than re-exploring a fresh K-chain ensemble from scratch. --chains 1
# (not the usual 6): this is a single continued trajectory, not a fresh
# perturbed-restart ensemble.
#
# Usage:
#   sbatch slurm/continue_ensemble_adam.sh <scheme> <init_from.pt> [adam_epochs] [lbfgs_iters]
#   sbatch slurm/continue_ensemble_adam.sh grid   /path/to/grid_layout_best.pt   15000 5000
#   sbatch slurm/continue_ensemble_adam.sh center /path/to/center_layout_best.pt 15000 5000
#
# Overwrites that scheme's existing
# test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_<scheme>/ IN PLACE --
# copy layout_best.pt elsewhere first (as <init_from.pt>, since that's what
# this script warm-starts from) if you want the pre-continuation result kept
# too. Removes that dir's adam_resume.pt before running so
# 04_optimize_lbfgs_ensemble.py's main() "already complete" skip check
# doesn't just reload the old cached result instead of continuing.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

SCHEME="${1:?usage: $0 <scheme: grid|center> <init_from.pt> [adam_epochs=15000] [lbfgs_iters=5000]}"
INIT_FROM="${2:?usage: $0 <scheme> <init_from.pt> [adam_epochs] [lbfgs_iters]}"
ADAM_EPOCHS="${3:-15000}"
LBFGS_ITERS="${4:-5000}"

DIR="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_ensemble_full_corpus_${SCHEME}"
rm -f "$DIR/adam_resume.pt"

python -u scripts/04_optimize_lbfgs_ensemble.py \
    --schemes "$SCHEME" --chains 1 --adam-epochs "$ADAM_EPOCHS" --lbfgs-iters "$LBFGS_ITERS" \
    --init_from "$INIT_FROM"

# --grid-layout only matters for scoring which BASELINE (grid vs center) the
# eval compares the optimized layout against -- matches the convention used
# throughout run_10pct_mean_analytic.sh / run_all_script_batch.sh.
FLAG=""; [ "$SCHEME" = grid ] && FLAG="--grid-layout"

python -u plots/layouts/true_utility.py $FLAG \
    --layout "$DIR/layout_best.pt" 2>&1 | tee "$DIR/true_utility_eval.txt"

python -u plots/layouts/utility_vs_energy.py $FLAG --run-dir "$DIR"

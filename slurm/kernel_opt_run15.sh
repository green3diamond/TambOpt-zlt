#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J kernel_opt_run15
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel-in-the-loop Stage 4 on run 15, recon utility, larger Adam steps.
# Usage: sbatch slurm/kernel_opt_run15.sh <grid|center> [lr] [epochs] [main|heldout]
#   main    : Stage 4's own event pool, 512 fresh events per step, best layout
#             on a separate 1,024-event scoring set (needs kernel_bw_run15.sh first)
#   heldout : the original fixed first 64 held-out events
# Resumes from opt_state.pt after a preemption.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

INIT=${1:-grid}; LR=${2:-10}; EPOCHS=${3:-20000}; POOL=${4:-main}
SUFFIX="recon_utility_lr${LR}_${INIT}"

python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION as R; sys.exit(0 if "15_sigma10_interp_hurdle" in R else 1)' \
    || { echo "constants.py does not point to run 15"; exit 1; }

if [ "$POOL" = main ]; then
    EXTRA="--pool main --batch-events 512 --chunk 32--n-score 1024 --score-every 250"
else
    EXTRA="--pool heldout --n-events 64 --chunk 8"
fi
python -u scripts/04_optimize_lbfgs_kernel.py --utility recon --init "$INIT" \
    --lr "$LR" --epochs "$EPOCHS" --log-every 100 $EXTRA \
    --out-suffix "$SUFFIX" --skip-analytic-eval || exit 1

DIR="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_kernel_${SUFFIX}"
flag=""; [ "$INIT" = grid ] && flag="--grid-layout"
python -u plots/layouts/true_utility.py --layout "$DIR/layout_best.pt" $flag 2>&1 | tee "$DIR/true_utility_eval.txt"
python -u plots/layouts/utility_vs_energy.py --run-dir "$DIR"

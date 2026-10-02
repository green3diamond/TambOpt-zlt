#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J kernel_opt_run17
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel-in-the-loop Stage 4 on run 17 (as kernel_opt_run15.sh): Stage 4's event pool, 512 fresh
# events per step, best layout on a separate 1,024-event scoring set. Needs kernel_bw_run17.sh first.
# Usage: sbatch slurm/kernel_opt_run17.sh <grid|center> <current|twotier> [lambda] [lr] [epochs]
# Two-tier uses the KERNEL thresholds (5 deg, 0.2): the recon sees kernel responses here, so the
# surrogate calibration (two_tier_calibration_5deg.json) does not apply. Resumes from opt_state.pt.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

INIT=${1:-grid}; UTIL=${2:-current}; LAM=${3:-0.3}; LR=${4:-10}; EPOCHS=${5:-20000}
python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION as R; sys.exit(0 if "17_plane45_interp_hurdle_100k" in R else 1)' \
    || { echo "constants.py does not point to run 17"; exit 1; }
R=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION)')
if [ "$UTIL" = current ]; then
    TAG=current
else
    export TAMBOPT_UTILITY=two_tier TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TWO_TIER_CALIB="$R/two_tier_calibration_kernel_5deg.json"
    TAG="twotier_l${LAM/./p}"
fi
SUFFIX="${TAG}_lr${LR}_${INIT}"

python -u scripts/04_optimize_lbfgs_kernel.py --utility recon --init "$INIT" \
    --lr "$LR" --epochs "$EPOCHS" --log-every 100 \
    --pool main --batch-events 512 --chunk 32 --n-score 1024 --score-every 250 \
    --out-suffix "$SUFFIX" --skip-analytic-eval || exit 1

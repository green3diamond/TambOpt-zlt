#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=96g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH -J kernel_opt_run18
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel-in-the-loop Stage 4 on run 18 (as kernel_opt_run17.sh): the Stage-1 event pool, 512 fresh events
# per step, best layout on a separate 1,024-event scoring set; the run's recon (trained on the
# shower-frame kernel, default *_deepsets folder). Needs kernel_bw_run18.sh and the recon first.
# Usage: sbatch slurm/kernel_opt_run18.sh <grid|center> <current|twotier> [lambda] [lr] [epochs] [calib]
#   calib (two-tier only): two_tier_calibration_<calib>.json in the run folder, default kernel_5deg;
#   e.g. kernel_5deg_angleonly ("good" = angle within 5 deg, no energy condition) -> suffix _angleonly.
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run18_guard.sh

INIT=${1:-grid}; UTIL=${2:-current}; LAM=${3:-0.3}; LR=${4:-10}; EPOCHS=${5:-20000}; CALIB=${6:-kernel_5deg}
RK=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RECON_FOLDER + "_deepsets")')
[ -f "$RK/recon.pt" ] || { echo "recon missing: $RK/recon.pt (run run18_pipeline.sh recon)"; exit 1; }
if [ "$UTIL" = current ]; then
    TAG=current
else
    export TAMBOPT_UTILITY=two_tier TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TWO_TIER_CALIB="$R/two_tier_calibration_${CALIB}.json"
    [ -f "$TAMBOPT_TWO_TIER_CALIB" ] || { echo "missing $TAMBOPT_TWO_TIER_CALIB"; exit 1; }
    TAG="twotier_l${LAM/./p}${CALIB#kernel_5deg}"
fi
SUFFIX="${TAG}_lr${LR}_${INIT}"

python -u scripts/04_optimize_lbfgs_kernel.py --utility recon --init "$INIT" \
    --lr "$LR" --epochs "$EPOCHS" --log-every 100 \
    --pool main --batch-events 512 --chunk 32 --n-score 1024 --score-every 250 \
    --recon-folder "$RK" --out-suffix "$SUFFIX" --skip-analytic-eval || exit 1

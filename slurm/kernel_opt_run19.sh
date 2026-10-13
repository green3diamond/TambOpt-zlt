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
#SBATCH -J kernel_opt_run19
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel-in-the-loop Stage 4 on run 19 (as kernel_opt_run18.sh): the Stage-1 event pool, 512 fresh events
# per step, best layout on a separate 1,024-event scoring set; the run's recon (trained on the
# shower-frame kernel, default *_deepsets folder). Needs kernel_bw_run19.sh and the recon first.
# Usage: sbatch slurm/kernel_opt_run19.sh <grid|center> <current|twotier> [lambda] [lr] [epochs] [calib]
#   calib (two-tier only): two_tier_calibration_<calib>.json in the run folder, default kernel_5deg;
#   e.g. kernel_5deg_angleonly ("good" = angle within 5 deg, no energy condition) -> suffix _angleonly;
#   kernel_1deg_angleonly -> suffix _1deg_angleonly.
# Env: RECON_DIR=<folder with recon.pt> (default <RECON_FOLDER>_deepsets), RECON_TAG appended to the suffix,
#      N_DET=<detectors> (default 100; other counts get _n<N> in the suffix).
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh

INIT=${1:-grid}; UTIL=${2:-current}; LAM=${3:-0.3}; LR=${4:-10}; EPOCHS=${5:-20000}; CALIB=${6:-kernel_5deg}
RK=${RECON_DIR:-$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RECON_FOLDER + "_deepsets")')}
[ -f "$RK/recon.pt" ] || { echo "recon missing: $RK/recon.pt (run run18_pipeline.sh recon)"; exit 1; }
if [ "$UTIL" = current ]; then
    TAG=current
else
    export TAMBOPT_UTILITY=two_tier TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TWO_TIER_CALIB="$R/two_tier_calibration_${CALIB}.json"
    [ -f "$TAMBOPT_TWO_TIER_CALIB" ] || { echo "missing $TAMBOPT_TWO_TIER_CALIB"; exit 1; }
    if [[ "$CALIB" == kernel_5deg* ]]; then TAG="twotier_l${LAM/./p}${CALIB#kernel_5deg}"
    else TAG="twotier_l${LAM/./p}_${CALIB#kernel_}"; fi
fi
N_DET=${N_DET:-100}
SUFFIX="${TAG}_lr${LR}_${INIT}${RECON_TAG:+_$RECON_TAG}"
[ "$N_DET" != 100 ] && SUFFIX="${SUFFIX}_n${N_DET}"

python -u scripts/04_optimize_lbfgs_kernel.py --utility recon --init "$INIT" \
    --lr "$LR" --epochs "$EPOCHS" --log-every 100 \
    --pool main --batch-events 512 --chunk 32 --n-score 1024 --score-every 250 \
    --recon-folder "$RK" --out-suffix "$SUFFIX" --n-det "$N_DET" --skip-analytic-eval || exit 1

#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=96g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J kernel_opt_run19
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel-in-the-loop Stage 4 on run 19 (as kernel_opt_run18.sh): the Stage-1 event pool, 512 fresh events
# per step, best layout on a separate 1,024-event scoring set; the run's recon (trained on the
# shower-frame kernel, default *_deepsets folder). Needs kernel_bw_run19.sh and the recon first.
# Usage: sbatch slurm/kernel_opt_run19.sh <grid|center> <current|twotier|threetier|swgotrig> [lambda] [lr] [epochs] [calib]
#   calib (two-tier only): two_tier_calibration_<calib>.json in the run folder, default kernel_5deg;
#   e.g. kernel_5deg_angleonly ("good" = angle within 5 deg, no energy condition) -> suffix _angleonly;
#   kernel_1deg_angleonly -> suffix _1deg_angleonly.
# Env: RECON_DIR=<folder with recon.pt> (default <RECON_FOLDER>_deepsets), RECON_TAG appended to the suffix,
#      N_DET=<detectors> (default 100; other counts get _n<N> in the suffix).
#      BG_WINDOW=<ns> cosmic-ray background in one readout window (suffix _bg1us / _bg10us / _bg<ns>ns),
#      GATE=knn23 kNN 2-of-3 gate (suffix _knn23), RESPONSE=surrogate Stage 2 hurdle surrogate instead of the
#      kernel (suffix _sur; FNN_DIR overrides the folder). No mail on preemption: one mail at the real end.
#      CHUNK=<sub-batch> showers per backward (default 32; 64 = same gradient, ~20% faster, 19 GB).
#      FREEZE_UNSEEN=1 detectors with no signal in a batch keep their position; LBFGS_CHUNKS=<n> chunked L-BFGS after
#      Adam (LBFGS_EVENTS showers per chunk, default 4096; LBFGS_MAXITER, default 40). Neither changes the suffix:
#      they continue the same run (resume from opt_state.pt / lbfgs_state.pt).
#      PREV_JOB=<id> for a continuation job (sbatch -d afternotok:<id>): runs only if <id> ended at its time limit
#      (or on a node failure); after a real error it exits at once rather than hit the same error again.
if [ -n "$PREV_JOB" ]; then
    PREV_STATE=$(sacct -j "$PREV_JOB" -X -n -o State | awk 'NR==1 {print $1}')
    case "$PREV_STATE" in
        TIMEOUT|NODE_FAIL) echo "[continue] job $PREV_JOB ended $PREV_STATE -> resuming" ;;
        *) echo "[continue] job $PREV_JOB ended ${PREV_STATE:-?}, not a time limit -> nothing to do"; exit 0 ;;
    esac
fi
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/${RUN_GUARD:-run19}_guard.sh                  # RUN_GUARD=run20: the Malata run

INIT=${1:-grid}; UTIL=${2:-current}; LAM=${3:-0.3}; LR=${4:-10}; EPOCHS=${5:-20000}; CALIB=${6:-kernel_5deg}
RK=${RECON_DIR:-$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RECON_FOLDER + "_deepsets")')}
[ -f "$RK/recon.pt" ] || { echo "recon missing: $RK/recon.pt (run run18_pipeline.sh recon)"; exit 1; }
if [ "$UTIL" = current ]; then
    TAG=current
elif [ "$UTIL" = swgotrig ]; then
    # branch swgo-trigger: two-tier reward x P(>= TRIG_K detectors with >= 1 particle), Poisson-binomial on the
    # expected counts of a TRIG_AREA m^2 detector (default 2); TRIG_ENERGY=1 adds the energy condition
    export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TRIG_K="${TRIG_K:-3}" \
           TAMBOPT_TRIG_AREA="${TRIG_AREA:-2}" TAMBOPT_TRIG_ENERGY="${TRIG_ENERGY:-0}"
    TAG="swgotrig_l${LAM/./p}_k${TAMBOPT_TRIG_K}_a${TAMBOPT_TRIG_AREA}"
    [ "$TAMBOPT_TRIG_ENERGY" = 1 ] && TAG="${TAG}_E"
    # TRIG_MODE=step: the notebook's near-step trigger on expected counts (TRIG_MU0, TRIG_SLOPE; suffix _step)
    export TAMBOPT_TRIG_MODE="${TRIG_MODE:-poisson}" TAMBOPT_TRIG_MU0="${TRIG_MU0:-1.0}" TAMBOPT_TRIG_SLOPE="${TRIG_SLOPE:-10.0}"
    [ "$TAMBOPT_TRIG_MODE" = step ] && TAG="${TAG}_step"
elif [ "$UTIL" = threetier ]; then
    # exclusive levels: 0.9 within 1 deg, 0.3 within 5 deg, 0.1 detected (>= 3 detectors > 1 particle); angle only
    export TAMBOPT_UTILITY=three_tier TAMBOPT_THREE_TIER="${THREE_TIER_W:-0.9,0.3,0.1}"
    TAG="threetier_${TAMBOPT_THREE_TIER//[.,]/}"
else
    export TAMBOPT_UTILITY=two_tier TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TWO_TIER_CALIB="$R/two_tier_calibration_${CALIB}.json"
    [ -f "$TAMBOPT_TWO_TIER_CALIB" ] || { echo "missing $TAMBOPT_TWO_TIER_CALIB"; exit 1; }
    if [[ "$CALIB" == kernel_5deg* ]]; then TAG="twotier_l${LAM/./p}${CALIB#kernel_5deg}"
    else TAG="twotier_l${LAM/./p}_${CALIB#kernel_}"; fi
fi
N_DET=${N_DET:-100}
SUFFIX="${TAG}_lr${LR}_${INIT}${RECON_TAG:+_$RECON_TAG}"
[ "$N_DET" != 100 ] && SUFFIX="${SUFFIX}_n${N_DET}"
# Env BATCH (showers per step, default 512) and LR_FINAL (log-linear lr decay to this) add _b<B> / _lrd<F> to the suffix.
[ -n "$BATCH" ] && [ "$BATCH" != 512 ] && SUFFIX="${SUFFIX}_b${BATCH}"
[ -n "$LR_FINAL" ] && SUFFIX="${SUFFIX}_lrd${LR_FINAL/./p}"
if [ -n "$BG_WINDOW" ] && [ "$BG_WINDOW" != 0 ]; then
    case "$BG_WINDOW" in 1000) SUFFIX="${SUFFIX}_bg1us" ;; 10000) SUFFIX="${SUFFIX}_bg10us" ;; *) SUFFIX="${SUFFIX}_bg${BG_WINDOW}ns" ;; esac
fi
[ "${GATE:-none}" != none ] && SUFFIX="${SUFFIX}_${GATE}"
[ "${RESPONSE:-kernel}" = surrogate ] && SUFFIX="${SUFFIX}_sur"

python -u scripts/04_optimize_lbfgs_kernel.py --init "$INIT" \
    --lr "$LR" --epochs "$EPOCHS" --log-every 100 \
    --batch-events ${BATCH:-512} ${LR_FINAL:+--lr-final $LR_FINAL} --chunk ${CHUNK:-32} --n-score 1024 --score-every 250 \
    --recon-folder "$RK" --out-suffix "$SUFFIX" --n-det "$N_DET" \
    --bg-window-ns "${BG_WINDOW:-0}" --gate "${GATE:-none}" --response "${RESPONSE:-kernel}" ${FNN_DIR:+--fnn-folder $FNN_DIR} \
    ${FREEZE_UNSEEN:+--freeze-unseen} --lbfgs-chunks "${LBFGS_CHUNKS:-0}" --lbfgs-chunk-events "${LBFGS_EVENTS:-4096}" \
    --lbfgs-max-iter "${LBFGS_MAXITER:-40}"
rc=$?
notify_end $rc
exit $rc

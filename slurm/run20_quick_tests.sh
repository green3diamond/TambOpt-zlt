#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 05:00:00
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20 quick tests: SGD from the 100-detector grid (INIT=<file.pt> for another start), lr 1 m, 100 steps, scored every
# 25; 1 deg (THETA=5: 5 deg) Poisson utility, default recon, expected counts, no background. 4 MIG slices:
#   energy     log10 E [GeV] bands 5.00-5.67, 5.67-6.33, 6.33-7.00 and the full range; 8 x 2,048 showers loaded per step
#              (a band keeps ~1/3), scoring set 3,072 (a band keeps ~1,000)
#   construct  one construction penalty each (modules/optimize/construction.py), full range, 4 x 2,048 per step; lambda
#              chosen so the penalty's gradient at the grid is ~half the utility's per-detector RMS (1e-3 of 2e-3):
#              slope 0.03, access 3, pads 0.2, contour 0.1
# THETA=5: the 5 deg reward instead (g = sigmoid(5 - a), q with 20 deg: New U's), names quick5_<name> (QTAG); its
# gradient at the grid is ~5x smaller (grad_check.json: per-detector RMS 0.0007 vs 0.0034 per 2,048 showers), so the
# construction lambdas are 5x smaller: slope 0.006, access 0.6, pads 0.04, contour 0.02.
# RESP=surrogate: the full runs on the run 20 Stage 2 surrogate (response surrogate, expected counts from its lit
# values): STEPS (default 1000) steps, names sur5_<name> with THETA=5 (sur_<name> otherwise).
# NSTEPS=<n> with the kernel: full runs of n steps, names ker5_<name> (as the surrogate runs, on the kernel).
# TRIGONLY=1: the trigger alone as utility (TAMBOPT_TRIG_ONLY, no angle / energy reward): tags trg / trg_1us.
# BG=1000: 1 us cosmic background, all detectors read out, trigger on all (no gate); names get _1us after the tag (e.g.
# sur5_1us_<name>). GATE=knn23 adds the kNN 2-of-3 gate with the trigger on read-out detectors (TRIG_GATED; _1us_knn).
# CHUNK=<n> showers per backward (default 16; memory and speed only: the surrogate runs use 512).
# SMOKE=1: 3 steps of every config of the set, one after another on one slice. ONLY=<0-3>: only that config, on one GPU
# (one job per config on gpu_requeue: sbatch -p gpu_requeue --gres=gpu:1 ...; resumes from opt_state.pt after a requeue).
# With fewer GPUs than configs the 4 share them round-robin (on gpu_requeue: one GPU, 4 processes, as the MIG tests).
# Out-suffix quick_<name>; logs <run>/slurm_logs/quick_<name>.log.
#   sbatch -J r20_quick_energy slurm/run20_quick_tests.sh energy
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')
L=$R/slurm_logs
GRID=$R/diagnostics/triplets/layouts/grid_n99.pt
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=poisson TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0 \
       TAMBOPT_TRIG_GATED=0
if [ "${THETA:-1}" = 5 ]; then
    export TAMBOPT_TRIG_THETA=5 TAMBOPT_TRIG_WIDTH=1 TAMBOPT_TRIG_QREF=20; QTAG=quick5
else
    export TAMBOPT_TRIG_THETA=1 TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5; QTAG=quick
fi
SET=${1:?energy|construct}
STEPS=100; EXTRA=""
if [ "${RESP:-kernel}" = surrogate ]; then
    STEPS=${NSTEPS:-1000}; EXTRA="--response surrogate"; QTAG=${QTAG/quick/sur}
elif [ -n "$NSTEPS" ]; then          # full kernel runs (the surrogate runs' counterpart): ker5_<name>
    STEPS=$NSTEPS; QTAG=${QTAG/quick/ker}
fi
if [ -n "$TRIGONLY" ]; then
    export TAMBOPT_TRIG_ONLY=1; QTAG=trg
fi
if [ -n "$PENONLY" ]; then          # PENONLY=1: no utility at all, the construction penalty alone (tag pen; construct set)
    export TAMBOPT_UTIL_ZERO=1; QTAG=pen
fi
if [ -n "$LHS" ]; then            # LHS=<seed>: start from a Latin-hypercube layout (Stage 4 --init lhs:<seed>), tag _lhs<seed>
    INIT=lhs:$LHS; QTAG=${QTAG}_lhs$LHS
fi
if [ -n "$BG" ] && [ "$BG" != 0 ]; then
    EXTRA="$EXTRA --bg-window-ns $BG"
    QTAG=${QTAG}_$([ "$BG" = 1000 ] && echo 1us || echo bg${BG}ns)
    if [ "${GATE:-none}" = knn23 ]; then
        EXTRA="$EXTRA --gate knn23"; export TAMBOPT_TRIG_GATED=1; QTAG=${QTAG}_knn
    fi
fi
[ -n "$SMOKE" ] && STEPS=3
case $SET in
    energy) NAMES=(e500_567 e567_633 e633_700 eall)
            ARGS=("--e-range 5.0 5.667 --accum 8 --n-score 3072" "--e-range 5.667 6.333 --accum 8 --n-score 3072"
                  "--e-range 6.333 7.01 --accum 8 --n-score 3072" "--accum 8 --n-score 3072")
            ENVS=("" "" "" "") ;;
    construct) NAMES=(slope access pads contour)
            ARGS=("--accum 4" "--accum 4" "--accum 4" "--accum 4")
            ENVS=("TAMBOPT_CONSTRUCT=slope=0.03" "TAMBOPT_CONSTRUCT=access=3" "TAMBOPT_CONSTRUCT=pads=0.2"
                  "TAMBOPT_CONSTRUCT=contour=0.1")
            [ "${THETA:-1}" = 5 ] && ENVS=("TAMBOPT_CONSTRUCT=slope=0.006" "TAMBOPT_CONSTRUCT=access=0.6"
                                           "TAMBOPT_CONSTRUCT=pads=0.04" "TAMBOPT_CONSTRUCT=contour=0.02") ;;
esac
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <i>
    n=${QTAG}_${NAMES[$2]}$([ -n "$SMOKE" ] && echo _smoke)
    env CUDA_VISIBLE_DEVICES=${DEVS[$(($1 % ${#DEVS[@]}))]} ${ENVS[$2]} python -u scripts/04_optimize_lbfgs_kernel.py --init ${INIT:-grid} --n-det 100 \
        --optim sgd --lr 1 --epochs $STEPS --batch-events 2048 --chunk ${CHUNK:-16} --score-every 25 --log-every 10 \
        --ckpt-every 10 ${ARGS[$2]} $EXTRA --out-suffix $n > $L/$n.log 2>&1
}
rc=0
if [ -n "$ONLY" ]; then
    run 0 $ONLY || rc=1
elif [ -n "$SMOKE" ]; then
    for i in 0 1 2 3; do run 0 $i || rc=1; done
else
    P=""
    for i in 0 1 2 3; do run $i $i & P="$P $!"; done
    for p in $P; do wait $p || rc=1; done
fi
# CHAIN=1: a continuation job of the same name (sbatch -d afternotok:<this>) resumes from the checkpoints if this one
# hits its time limit; once the runs have finished it is not needed
[ "$rc" = 0 ] && [ -n "$CHAIN" ] && [ -z "$SMOKE" ] && scancel --state=PENDING -u "$USER" -n "$SLURM_JOB_NAME"
exit $rc

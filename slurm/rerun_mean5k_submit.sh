#!/bin/bash
# Rerun of every optimization in the MAIN TEXT of papers 1 and 2 (docs: ~/.claude plan "rerun-mean5k"): the original
# launch settings (paper1_opt_submit.sh, run20_submit.sh, run20_cable_submit.sh, the run-19 trigger / two-tier /
# three-tier lines, run20_quick_tests.sh), changed to
#   - the mean arrival time without noise (TAMBOPT_TIME_MODEL=mean, TAMBOPT_TIME_NOISE=0; no effect on surrogate runs)
#   - 5,120 showers per step (accum 1), at most 2,000 steps
#   - plateau schedule: score every 25 steps, step halved after 4 scores without a gain > 0.05, stop at 1/30 of the
#     starting step (replaces --score-every and the two-tier runs' lr_final)
#   - sub-batch sizes from the GPU's memory (--chunk 0, --score-chunk 0), checkpoints every 25 steps
#   - scoring set (best layout, plateau) 10,240 held-out pool showers instead of 1,024
#   - folder names: the original out-suffix + _m5k (old runs untouched)
#   - gpu_requeue, each run a chain of 3 jobs of 10 h (2 afternotok continuations resume from the checkpoints)
# Jobs run in this worktree (branch rerun-mean5k). 44 runs:
#   p1     16  both sites x {center, ugrid} x {Adam + 20 L-BFGS chunks, SGD} x {kernel, surrogate}, Orig. utility, lr 10
#   r19     6  12 km plane: two-tier lambda 0.3, SWGO trigger 5 deg Poisson, three-tier; grid and centre; lr 10
#   r20     4  Malata: SWGO trigger 5 deg Poisson and step; grid and centre; lr 10
#   cable   3  Malata: SWGO trigger 5 deg Poisson + cable 0.1 (grid, centre), 0.2 (grid)
#   quick  15  Malata, SGD lr 1 from the grid, 5 deg trigger: ker5 / sur5 / sur5_1us x {eall, access, pads, contour},
#              pen x {access, pads, contour}
#   bash slurm/rerun_mean5k_submit.sh [all|p1|r19|r20|cable|quick]       (SMOKE=1: one gpu_test job, see below)
set -e
W=/n/home05/zdimitrov/tambo/TambOpt_rerun
cd $W
WHAT=${1:-all}; STEPS=${STEPS:-2000}; NCONT=${NCONT:-2}; TLIM=${TLIM:-10:00:00}; T=${TAG:-_m5k}
V=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6
declare -A RUN=([plane]=$V/19_plane45S_12km_showerframe_400k [malata]=$V/20_malata_showerframe)
declare -A GUARD=([plane]=run19 [malata]=run20)
PLAT="--plateau-every 25 --plateau-patience 4 --plateau-factor 0.5 --plateau-tol 0.05 --plateau-min-frac 0.0333333 --n-score 10240"
COMMON="TAMBOPT_TIME_MODEL=mean TAMBOPT_TIME_NOISE=0 BATCH=5120 ACCUM=1 CHUNK=0 SCORE_CHUNK=0 CKPT_EVERY=25 CHAIN=1"
SB="sbatch --parsable --chdir=$W -p gpu_requeue -t $TLIM --exclude=holygpu8a19102 --mail-type=END"   # one mail per sub-job (Slurm cannot filter by end state)
QSB="$SB --gres=gpu:1 -c 8 --mem=96g --constraint=a100|h100|h200 --requeue --open-mode=append"

chain () {   # <job name> <env> <script and args...>: first job + NCONT continuations
    local name=$1 e=$2; shift 2
    local j c
    j=$(env $COMMON $e XARGS="$PLAT" $SB -J $name "$@")
    for c in $(seq $NCONT); do
        j=$(env $COMMON $e XARGS="$PLAT" PREV_JOB=$j $SB -J $name -d afternotok:$j "$@")
    done
    echo "  $name: last job $j"
}
qchain () {  # <job name> <env> <set>: run20_quick_tests.sh, one config (ONLY) on one GPU
    local name=$1 e=$2 set=$3
    local j c
    j=$(env $COMMON $e THETA=5 NSTEPS=$STEPS TAG=$T XARGS="$PLAT --ckpt-every 25 --score-chunk 0" $QSB -J $name \
        slurm/run20_quick_tests.sh $set)
    for c in $(seq $NCONT); do
        j=$(env $COMMON $e THETA=5 NSTEPS=$STEPS TAG=$T XARGS="$PLAT --ckpt-every 25 --score-chunk 0" $QSB -J $name \
            -d afternotok:$j slurm/run20_quick_tests.sh $set)
    done
    echo "  $name: last job $j"
}

if [ -n "$SMOKE" ]; then
    # one gpu_test job, 10 steps each, folders *_m5ksmoke: kernel Adam + 1 L-BFGS chunk (Malata), kernel trigger
    # (12 km), surrogate, quick-test kernel and surrogate with 1 us background; GPU utilization every 5 s
    sbatch --chdir=$W -p gpu_test --gres=gpu:1 -c 8 --mem=64g -t 01:00:00 -J m5k_smoke -o slurm_logs/slurm-%j-%x.out --wrap "
        set -x; export $COMMON GPU_LOG_EVERY=5 XARGS='$PLAT'; unset CHAIN
        RUN_GUARD=run20 INIT_TAG=center RECON_TAG=p1 LBFGS_CHUNKS=1 LBFGS_EVENTS=1024 SUFFIX_OVERRIDE=current_lr10_center_p1_m5ksmoke \
            bash slurm/kernel_opt_run19.sh center current 0.3 10 10
        RUN_GUARD=run19 TRIG_AREA=2 TRIG_MODE=poisson SUFFIX_OVERRIDE=swgotrig_l0p3_k3_a2_lr10_grid_b2048_m5ksmoke \
            bash slurm/kernel_opt_run19.sh grid swgotrig 0.3 10 10
        RUN_GUARD=run20 INIT_TAG=center RECON_TAG=p1 RESPONSE=surrogate SUFFIX_OVERRIDE=current_lr10_center_p1_sur_m5ksmoke \
            bash slurm/kernel_opt_run19.sh center current 0.3 10 10
        THETA=5 NSTEPS=10 TAG=_m5ksmoke ONLY=3 XARGS='$PLAT --ckpt-every 25 --score-chunk 0' bash slurm/run20_quick_tests.sh energy
        THETA=5 NSTEPS=10 TAG=_m5ksmoke ONLY=1 RESP=surrogate BG=1000 XARGS='$PLAT --ckpt-every 25 --score-chunk 0' \
            bash slurm/run20_quick_tests.sh construct"
    exit 0
fi

if [ $WHAT = all ] || [ $WHAT = p1 ]; then
    echo "p1 (16)"
    for site in plane malata; do
        R=${RUN[$site]}
        for start in center ugrid; do
            if [ $start = ugrid ]; then INIT=$R/diagnostics/paper1/ugrid.pt; else INIT=center; fi
            for resp in kernel surrogate; do
                for opt in adam sgd; do
                    suf=current_lr10_${start}_p1; [ $resp = surrogate ] && suf=${suf}_sur; [ $opt = sgd ] && suf=${suf}_sgd
                    e="RUN_GUARD=${GUARD[$site]} INIT_TAG=$start RECON_TAG=p1 RESPONSE=$resp OPTIM=$opt SUFFIX_OVERRIDE=$suf$T"
                    [ $opt = adam ] && e="$e LBFGS_CHUNKS=20"
                    chain m5k_p1_${site}_${start}_${resp}_${opt} "$e" slurm/kernel_opt_run19.sh $INIT current 0.3 10 $STEPS
                done
            done
        done
    done
fi
if [ $WHAT = all ] || [ $WHAT = r19 ]; then
    echo "r19 (6)"
    for s in grid center; do
        chain m5k_r19_twotier_$s "RUN_GUARD=run19 SUFFIX_OVERRIDE=twotier_l0p3_lr10_${s}_b2048_lrd1$T" \
            slurm/kernel_opt_run19.sh $s twotier 0.3 10 $STEPS kernel_5deg
        chain m5k_r19_swgotrig_$s "RUN_GUARD=run19 TRIG_AREA=2 TRIG_MODE=poisson SUFFIX_OVERRIDE=swgotrig_l0p3_k3_a2_lr10_${s}_b2048$T" \
            slurm/kernel_opt_run19.sh $s swgotrig 0.3 10 $STEPS
        chain m5k_r19_threetier_$s "RUN_GUARD=run19 SUFFIX_OVERRIDE=threetier_090301_lr10_$s$T" \
            slurm/kernel_opt_run19.sh $s threetier 0.3 10 $STEPS
    done
fi
if [ $WHAT = all ] || [ $WHAT = r20 ]; then
    echo "r20 (4)"
    for s in grid center; do
        for mode in poisson step; do
            tag=""; [ $mode = step ] && tag=_step
            chain m5k_r20_swgotrig${tag}_$s "RUN_GUARD=run20 TRIG_AREA=2 TRIG_MODE=$mode SUFFIX_OVERRIDE=swgotrig_l0p3_k3_a2${tag}_lr10_${s}_b2048$T" \
                slurm/kernel_opt_run19.sh $s swgotrig 0.3 10 $STEPS
        done
    done
fi
if [ $WHAT = all ] || [ $WHAT = cable ]; then
    echo "cable (3)"
    for sc in grid:0.1 grid:0.2 center:0.1; do
        s=${sc%:*}; c=${sc#*:}
        chain m5k_r20_cab${c/./p}_$s "RUN_GUARD=run20 TRIG_AREA=2 TRIG_MODE=poisson CABLE=$c SUFFIX_OVERRIDE=swgotrig_l0p3_k3_a2_lr10_${s}_b2048_cab${c/./p}$T" \
            slurm/kernel_opt_run19.sh $s swgotrig 0.3 10 $STEPS
    done
fi
if [ $WHAT = all ] || [ $WHAT = quick ]; then
    echo "quick (15)"
    for r in "ker5:" "sur5:RESP=surrogate" "sur5_1us:RESP=surrogate BG=1000"; do
        tag=${r%%:*}; e=${r#*:}
        qchain m5k_r20_${tag}_eall "$e ONLY=3" energy
        for i in 1:access 2:pads 3:contour; do
            qchain m5k_r20_${tag}_${i#*:} "$e ONLY=${i%%:*}" construct
        done
    done
    for i in 1:access 2:pads 3:contour; do
        qchain m5k_r20_pen_${i#*:} "RESP=surrogate PENONLY=1 ONLY=${i%%:*}" construct
    done
fi

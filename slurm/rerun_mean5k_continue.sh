#!/bin/bash
# Continue the seven 12 km centre-start reruns (rerun_mean5k_submit.sh, _m5k) that had not spread over the plane in
# 2,000 steps: from each run's last opt_state.pt, copied into a new folder <name>_m5kc (the _m5k results stay), with
# the step size reset to 50 m (--plateau-reset) and the plateau schedule halving it down to 50/150 = 0.33 m (the
# floor of the first runs), up to 6,000 steps in all; Adam runs then redo their 20 L-BFGS chunks. Same mean time
# (no noise), 5,120 showers per step, 10,240 scoring showers. 4 jobs of 10 h per run (3 continuations).
# Still-running _m5k jobs of these runs are cancelled first.
#   bash slurm/rerun_mean5k_continue.sh
set -e
W=/n/home05/zdimitrov/tambo/TambOpt_rerun
cd $W
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
K=$R/test_v6_run_04_optimize_lbfgs_kernel
LR=50; EPOCHS=6000; NCONT=3; TLIM=10:00:00
PLAT="--plateau-every 25 --plateau-patience 4 --plateau-factor 0.5 --plateau-tol 0.05 --plateau-min-frac 0.0066667 --n-score 10240 --plateau-reset"
COMMON="TAMBOPT_TIME_MODEL=mean TAMBOPT_TIME_NOISE=0 BATCH=5120 ACCUM=1 CHUNK=0 SCORE_CHUNK=0 CKPT_EVERY=25 CHAIN=1 RUN_GUARD=run19"
SB="sbatch --parsable --chdir=$W -p gpu_requeue -t $TLIM --exclude=holygpu8a19102 --mail-type=END"
# name of the _m5k job | out-suffix | env | wrapper arguments (init, utility, lambda; lr and epochs appended)
RUNS=(
 "m5k_p1_plane_center_kernel_adam|current_lr10_center_p1|RESPONSE=kernel OPTIM=adam LBFGS_CHUNKS=20|center current 0.3"
 "m5k_p1_plane_center_kernel_sgd|current_lr10_center_p1_sgd|RESPONSE=kernel OPTIM=sgd|center current 0.3"
 "m5k_p1_plane_center_surrogate_adam|current_lr10_center_p1_sur|RESPONSE=surrogate OPTIM=adam LBFGS_CHUNKS=20|center current 0.3"
 "m5k_p1_plane_center_surrogate_sgd|current_lr10_center_p1_sur_sgd|RESPONSE=surrogate OPTIM=sgd|center current 0.3"
 "m5k_r19_swgotrig_center|swgotrig_l0p3_k3_a2_lr10_center_b2048|TRIG_AREA=2 TRIG_MODE=poisson|center swgotrig 0.3"
 "m5k_r19_threetier_center|threetier_090301_lr10_center||center threetier 0.3"
 "m5k_r19_twotier_center|twotier_l0p3_lr10_center_b2048_lrd1||center twotier 0.3"
)
for r in "${RUNS[@]}"; do
    IFS='|' read -r job suf env args <<< "$r"
    ids=$(squeue -u "$USER" -h -n "$job" -o "%i" | tr '\n' ' ')
    if [ -n "$ids" ]; then scancel $ids; echo "  cancelled $job: $ids"; fi
done
sleep 20
for r in "${RUNS[@]}"; do
    IFS='|' read -r job suf env args <<< "$r"
    src=${K}_${suf}_m5k; dst=${K}_${suf}_m5kc
    mkdir -p "$dst"
    [ -f "$dst/opt_state.pt" ] || cp "$src/opt_state.pt" "$dst/opt_state.pt"
    [ "$args" = "center twotier 0.3" ] && extra=kernel_5deg || extra=""
    name=${job/m5k_/m5kc_}
    j=$(env $COMMON $env SUFFIX_OVERRIDE=${suf}_m5kc XARGS="$PLAT" $SB -J $name slurm/kernel_opt_run19.sh $args $LR $EPOCHS $extra)
    for c in $(seq $NCONT); do
        j=$(env $COMMON $env SUFFIX_OVERRIDE=${suf}_m5kc XARGS="$PLAT" PREV_JOB=$j $SB -J $name -d afternotok:$j \
            slurm/kernel_opt_run19.sh $args $LR $EPOCHS $extra)
    done
    echo "  $name: from $(basename $src) -> $(basename $dst), last job $j"
done

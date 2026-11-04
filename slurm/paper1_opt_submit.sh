#!/bin/bash
# Paper 1 optimizer test set (original utility, U_theta + U_phi + U_E): {Adam then chunked L-BFGS, SGD} x {centre,
# uniform grid (diagnostics/paper1/ugrid.pt)} x {kernel, surrogate response} x {12 km plane (run 19), Malata (run 20)}
# = 16 Stage 4 runs, 512 showers/step, lr 10, <steps> steps (default 20,000), suffix tag _p1. Each run is a 3-day job
# plus <NCONT> continuations (default 2) that resume from the checkpoint after a time limit. Then one gpu_test job per
# site scores the starts and the 8 best layouts on 5,000 held-out showers with the kernel (tmp_rings/optres/score.py).
#   bash slurm/paper1_opt_submit.sh [steps]
set -e
cd /n/home05/zdimitrov/tambo/TambOpt
STEPS=${1:-20000}; NCONT=${NCONT:-2}; TLIM=${TLIM:-3-00:00:00}
V=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6
O=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/optres
K=test_v6_run_04_optimize_lbfgs_kernel_current_lr10
declare -A RUN=([plane]=$V/19_plane45S_12km_showerframe_400k [malata]=$V/20_malata_showerframe)
declare -A GUARD=([plane]=run19 [malata]=run20)
for site in plane malata; do
    R=${RUN[$site]}; JOBS=""; PAIRS=""
    for start in center ugrid; do
        if [ $start = ugrid ]; then INIT=$R/diagnostics/paper1/ugrid.pt; else INIT=center; fi
        for resp in kernel surrogate; do
            for opt in adam sgd; do
                E="RUN_GUARD=${GUARD[$site]} INIT_TAG=$start RECON_TAG=p1 RESPONSE=$resp OPTIM=$opt CHAIN=1"
                [ $opt = adam ] && E="$E LBFGS_CHUNKS=20"
                name=p1_${site}_${start}_${resp}_${opt}
                J=$(env $E sbatch --parsable --exclude=holygpu8a19102 -t $TLIM -J $name slurm/kernel_opt_run19.sh $INIT current 0.3 10 $STEPS)
                JOBS="$JOBS:$J"
                for c in $(seq $NCONT); do
                    J=$(env $E PREV_JOB=$J sbatch --parsable --exclude=holygpu8a19102 -t $TLIM -J $name -d afternotok:$J \
                        slurm/kernel_opt_run19.sh $INIT current 0.3 10 $STEPS)
                    JOBS="$JOBS:$J"
                done
                suf=${K}_${start}_p1; [ $resp = surrogate ] && suf=${suf}_sur; [ $opt = sgd ] && suf=${suf}_sgd
                PAIRS="$PAIRS ${start}_${resp}_${opt}=$R/$suf/layout_best.pt"
            done
        done
    done
    if [ $site = plane ]; then C0=$R/diagnostics/opt/layouts/start_center.pt; else C0=$O/malata_start_center.pt; fi
    START="center=$C0 ugrid=$R/diagnostics/paper1/ugrid.pt"
    S=$(sbatch --parsable -p gpu_test --gres=gpu:1 -c 4 --mem=64g -t 01:00:00 -J p1_score_$site -d afterany$JOBS \
        -o $O/p1_score_${site}_%j.out --wrap "source slurm/env.sh; source slurm/${GUARD[$site]}_guard.sh; export MPLCONFIGDIR=/tmp/mplcache_\$USER; \
        python -u $O/score.py $O/p1_scores_$site.json $START $PAIRS")
    echo "$site: optimisation jobs${JOBS//:/ }; scoring $S"
done

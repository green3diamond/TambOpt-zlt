#!/bin/bash
# Run 20 (Malata): submit the whole chain with dependencies, after the corpus filter job <filter_job>
# (scripts/20_filter_corpus.py). Same settings as run 19; every job sources slurm/run20_guard.sh (RUN_GUARD=run20).
#   build -> {kernel widths, recon} -> {drawn-count fine-tune, 5 Stage 4 swgo_trigger runs} -> layouts ->
#   {validation: 2 recons x 5 count seeds, ghosts} -> tables + one notification mail
#
#   bash slurm/run20_submit.sh <filter_job>
set -e
cd /n/home05/zdimitrov/tambo/TambOpt
FILT=$1
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe
O=$R/diagnostics/swgotrig; L=$O/validate_count_layouts.txt
mkdir -p $O $R/slurm_logs
export RUN_GUARD=run20
GPU="-p gpu_requeue --gres=gpu:1 --constraint=a100|h100|h200 --exclude=holygpu8a19102 --requeue --open-mode=append"
PRE="source slurm/env.sh; source slurm/notify.sh; source slurm/gpu_gate.sh; gpu_health_gate; source slurm/run20_guard.sh; export MPLCONFIGDIR=/tmp/mplcache_\$USER"
sub() { sbatch --parsable "$@"; }

BUILD=$(sub -J r20_build --mem=320g -d afterok:$FILT slurm/run19_pipeline.sh build)
BW=$(sub -J r20_kernel_bw -d afterok:$BUILD slurm/kernel_bw_run19.sh)
RECON=$(sub -J r20_recon -d afterok:$BUILD slurm/run19_pipeline.sh recon)
FT=$(sub $GPU -J r20_ft_count -c 8 --mem=200g -t 08:00:00 -d afterok:$RECON -o $R/slurm_logs/slurm-%j-%x.out \
     --chdir=$PWD --wrap "$PRE; python -u scripts/03c_finetune_recon_noise.py --counting poisson --min-hit 3 \
     --windows 0 1000 10000 --epochs 28 --patience 5; rc=\$?; notify_end \$rc; exit \$rc")
OPT=""
for spec in "grid 2 poisson" "center 2 poisson" "grid 20 poisson" "grid 2 step" "center 2 step"; do
    set -- $spec
    OPT="$OPT:$(BATCH=2048 CHUNK=64 TRIG_AREA=$2 TRIG_MODE=$3 sub -J r20_opt_${1}_a$2_$3 -d afterok:$RECON:$BW \
                slurm/kernel_opt_run19.sh $1 swgotrig 0.3 10 5000)"
done
S4=$R/test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3
LAY=$(sub -p serial_requeue -c 2 --mem=16g -t 01:00:00 -J r20_layouts -d afterok$OPT -o $R/slurm_logs/slurm-%j-%x.out \
      --chdir=$PWD --wrap "source slurm/env.sh; source slurm/run20_guard.sh; \
      python plots/analysis/run19/triplets/make_triplets.py --disks 100 150 200 300 400 --tri-r 10 25 && \
      python plots/analysis/run19/counting/make_region_triplets.py --radii 150 --tri-r 25 && \
      T=$R/diagnostics/triplets/layouts && printf '%s\n' \
        grid_n99=\$T/grid_n99.pt disk_r400_n99=\$T/disk_r400_n99.pt disk_r150_n99=\$T/disk_r150_n99.pt \
        tri33_r10_n99=\$T/tri33_r10_n99.pt tri33_r25_n99=\$T/tri33_r25_n99.pt \
        tri33disk_R150_r25_n99=$R/diagnostics/counting/layouts/tri33disk_R150_r25_n99.pt \
        opt_a2_grid=${S4}_a2_lr10_grid_b2048/layout_best.pt opt_a2_center=${S4}_a2_lr10_center_b2048/layout_best.pt \
        opt_a20_grid=${S4}_a20_lr10_grid_b2048/layout_best.pt \
        opt_a2_step_grid=${S4}_a2_step_lr10_grid_b2048/layout_best.pt \
        opt_a2_step_center=${S4}_a2_step_lr10_center_b2048/layout_best.pt > $L")

# validation: drawn counts, 5 seeds, held-out chunks of 5,000; prefixes as run 19 (validate_count = default recon,
# valcnt_count = drawn-count fine-tune)
VAL=""
for rk in "deepsets validate_count" "deepsets_count valcnt_count"; do
    set -- $rk
    for seed in 1 2 3 4 5; do
        pre=$2_s$seed
        VAL="$VAL:$(sub $GPU -J r20_val -c 4 --mem=64g -t 02:00:00 -d afterok:$LAY:$FT -o $R/slurm_logs/slurm-%j-%x.out \
            --chdir=$PWD --wrap "$PRE; for s in 0 5000 10000 15000 20000; do f=$O/${pre}_\$s.json; [ -s \$f ] && continue; \
            python -u plots/analysis/run19/triplets/score_knn.py --layouts $L --recon-dir \$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)')_$1 \
            --counting poisson --count-seed $seed --windows 0 1000 10000 --n-events 5000 --event-start \$s \
            --out \$f.tmp_\$SLURM_JOB_ID && mv \$f.tmp_\$SLURM_JOB_ID \$f || exit 1; done")"
    done
done
GH=$(sub $GPU -J r20_ghost -c 4 --mem=64g -t 02:00:00 -d afterok:$LAY:$FT -o $R/slurm_logs/slurm-%j-%x.out --chdir=$PWD \
     --wrap "$PRE; RF=\$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)'); \
     python -u plots/analysis/run19/swgotrig/score_ghost.py --layouts $L --recon-dirs \${RF}_deepsets \${RF}_deepsets_count \
     --n-windows 20000 --out $O/ghost_scores.json.tmp_\$SLURM_JOB_ID && mv $O/ghost_scores.json.tmp_\$SLURM_JOB_ID $O/ghost_scores.json")
TAB=$(sub -p serial_requeue -c 1 --mem=8g -t 00:30:00 -J r20_tables -d afterany$VAL:$GH -o $R/slurm_logs/slurm-%j-%x.out \
      --chdir=$PWD --wrap "source slurm/env.sh; source slurm/notify.sh; source slurm/run20_guard.sh; \
      D=plots/analysis/run19/swgotrig; python \$D/make_count_tables.py && python \$D/make_triggered_tables.py && \
      python \$D/make_ghost_tables.py && python \$D/make_compact_tables.py && python \$D/make_compact_tables.py --notion && \
      python \$D/make_compact_tables.py --only count --notion; rc=\$?; notify_end \$rc; exit \$rc")
echo "filter $FILT -> build $BUILD, widths $BW, recon $RECON, ft $FT, opt${OPT//:/ }, layouts $LAY, val${VAL//:/ }, ghost $GH, tables $TAB"

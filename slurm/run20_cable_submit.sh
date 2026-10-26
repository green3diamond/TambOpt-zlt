#!/bin/bash
# Run 20 (Malata): Stage 4 with the cable cost (swgo_trigger Poisson 2 m^2 minus lambda x 30 x 3-D MST length / grid_n99's,
# scripts/04_optimize_lbfgs_kernel.py, CABLE=<lambda>), then the run 20 validation again with the cable layouts added.
#   4 opt runs (after the recon and kernel widths): lambda 0.05 / 0.1 / 0.2 from the grid, 0.1 from the centre
#   -> after them AND the main chain's tables: append opt_cab* to validate_count_layouts.txt, rescore every layout
#   (2 recons x 5 count seeds, same files as run20_submit.sh, overwritten), ghosts, tables + one notification mail
#
#   bash slurm/run20_cable_submit.sh <recon_job> <widths_job> <main_tables_job>
set -e
cd /n/home05/zdimitrov/tambo/TambOpt
RECON=$1; BW=$2; MAINTAB=$3
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe
O=$R/diagnostics/swgotrig; L=$O/validate_count_layouts.txt
export RUN_GUARD=run20
GPU="-p gpu_requeue --gres=gpu:1 --constraint=a100|h100|h200 --exclude=holygpu8a19102 --requeue --open-mode=append"
PRE="source slurm/env.sh; source slurm/notify.sh; source slurm/gpu_gate.sh; gpu_health_gate; source slurm/run20_guard.sh; export MPLCONFIGDIR=/tmp/mplcache_\$USER"
sub() { sbatch --parsable "$@"; }

OPT=""; LINES=""
S4=$R/test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_a2_lr10
for spec in "grid 0.05" "grid 0.1" "grid 0.2" "center 0.1"; do
    set -- $spec
    OPT="$OPT:$(CABLE=$2 BATCH=2048 CHUNK=64 TRIG_AREA=2 TRIG_MODE=poisson sub -J r20_cab_${1}_${2} \
                -d afterok:$RECON:$BW slurm/kernel_opt_run19.sh $1 swgotrig 0.3 10 5000)"
    LINES="$LINES opt_cab${2/./p}_$1=${S4}_$1_b2048_cab${2/./p}/layout_best.pt"
done
LAY=$(sub -p serial_requeue -c 1 --mem=2g -t 00:10:00 -J r20_cab_layouts -d afterok$OPT,afterany:$MAINTAB \
      -o $R/slurm_logs/slurm-%j-%x.out --wrap "grep -v '^opt_cab' $L > $L.tmp; printf '%s\n' $LINES >> $L.tmp; mv $L.tmp $L; cat $L")

VAL=""
for rk in "deepsets validate_count" "deepsets_count valcnt_count"; do
    set -- $rk
    for seed in 1 2 3 4 5; do
        pre=$2_s$seed
        VAL="$VAL:$(sub $GPU -J r20_cab_val -c 4 --mem=64g -t 03:00:00 -d afterok:$LAY -o $R/slurm_logs/slurm-%j-%x.out \
            --chdir=$PWD --wrap "$PRE; for s in 0 5000 10000 15000 20000; do f=$O/${pre}_\$s.json; \
            grep -q opt_cab \$f 2>/dev/null && continue; \
            python -u plots/analysis/run19/triplets/score_knn.py --layouts $L --recon-dir \$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)')_$1 \
            --counting poisson --count-seed $seed --windows 0 1000 10000 --n-events 5000 --event-start \$s \
            --out \$f.tmp_\$SLURM_JOB_ID && mv \$f.tmp_\$SLURM_JOB_ID \$f || exit 1; done")"
    done
done
GH=$(sub $GPU -J r20_cab_ghost -c 4 --mem=64g -t 02:00:00 -d afterok:$LAY -o $R/slurm_logs/slurm-%j-%x.out --chdir=$PWD \
     --wrap "$PRE; RF=\$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)'); \
     python -u plots/analysis/run19/swgotrig/score_ghost.py --layouts $L --recon-dirs \${RF}_deepsets \${RF}_deepsets_count \
     --n-windows 20000 --out $O/ghost_scores.json.tmp_\$SLURM_JOB_ID && mv $O/ghost_scores.json.tmp_\$SLURM_JOB_ID $O/ghost_scores.json")
TAB=$(sub -p serial_requeue -c 1 --mem=8g -t 00:30:00 -J r20_cab_tables -d afterany$VAL:$GH -o $R/slurm_logs/slurm-%j-%x.out \
      --chdir=$PWD --wrap "source slurm/env.sh; source slurm/notify.sh; source slurm/run20_guard.sh; \
      D=plots/analysis/run19/swgotrig; python \$D/make_count_tables.py && python \$D/make_triggered_tables.py && \
      python \$D/make_ghost_tables.py && python \$D/make_compact_tables.py && python \$D/make_compact_tables.py --notion && \
      python \$D/make_compact_tables.py --only count --notion; rc=\$?; notify_end \$rc; exit \$rc")
echo "cable opt${OPT//:/ }, layouts $LAY, val${VAL//:/ }, ghost $GH, tables $TAB"

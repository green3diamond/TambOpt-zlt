#!/bin/bash
# Run 20b (Malata): Stage 4 for 1 deg reconstruction under background, three runs from the grid, all on gpu_requeue:
#   1deg_poisson  reward soft(a < 1 deg) [width 0.2] + 0.3 (1 - g) / (1 + (a / 5)^2), Poisson trigger on read-out detectors
#   1deg_step     the same with the step trigger (SWGOLO notebook)
#   oldu          Old U: soft(a < 5) soft(|dlog10 E| < 0.2) + 0.3 (1 - g) q, no trigger term
# Common: the run 20 drawn-count fine-tuned recon, straight-through Poisson-drawn counts, 10 us background, kNN 2-of-3
# gate, 8 x 2,048 showers per step (--accum 8), step size halved when the scoring-set U stalls, down to lr / 100 (PLATEAU), up to <steps>
# steps, checkpoints every 10, as <TLIM> jobs (default 8 h, so they start sooner) with <NCONT> continuations each (default 2)
# that resume from the checkpoint if the previous job hit its time limit and are cancelled once the run has finished. Then: the three layouts added to validate_count_layouts.txt, validation (2 recons x 5
# count seeds, slurm/run20_val_chunks.sh, only what is missing), ghosts, tables and figures, one notification mail.
#
#   bash slurm/run20b_submit.sh <lr> [steps, default 1000]
set -e
cd /n/home05/zdimitrov/tambo/TambOpt
LR=$1; STEPS=${2:-1000}
[ -n "$LR" ] || { echo "usage: bash slurm/run20b_submit.sh <lr> [steps]"; exit 1; }
R=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe
O=$R/diagnostics/swgotrig; L=$O/validate_count_layouts.txt
RC=$R/test_v6_run_03_recentered_deepsets_count
[ -f $RC/recon.pt ] || { echo "missing $RC/recon.pt"; exit 1; }
export RUN_GUARD=run20
GPU="-p gpu_requeue --gres=gpu:1 --constraint=a100|h100|h200 --exclude=holygpu8a19102 --requeue --open-mode=append"
PRE="source slurm/env.sh; source slurm/notify.sh; source slurm/gpu_gate.sh; gpu_health_gate; source slurm/run20_guard.sh; export MPLCONFIGDIR=/tmp/mplcache_\$USER"
sub() { sbatch --parsable "$@"; }
COMMON="RECON_DIR=$RC RECON_TAG=count BG_WINDOW=10000 GATE=knn23 COUNTING=poisson ACCUM=8 PLATEAU=1 \
PLATEAU_MIN_FRAC=0.01 CKPT_EVERY=10 BATCH=2048 CHUNK=${CHUNK:-64} TRIG_AREA=2"
S4=$R/test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_a2
TAIL=lr${LR}_grid_count_b2048_bg10us_knn23_cnt_acc8_plat
declare -A ENV=([1deg_poisson]="TRIG_THETA=1 TRIG_WIDTH=0.2 TRIG_QREF=5 TRIG_GATED=1 TRIG_MODE=poisson"
                [1deg_step]="TRIG_THETA=1 TRIG_WIDTH=0.2 TRIG_QREF=5 TRIG_GATED=1 TRIG_MODE=step"
                [oldu]="TRIG_OFF=1 TRIG_ENERGY=1 TRIG_MODE=poisson")
declare -A DIR=([1deg_poisson]=${S4}_th1_gtrig_$TAIL [1deg_step]=${S4}_step_th1_gtrig_$TAIL [oldu]=${S4}_E_notrig_$TAIL)
TLIM=${TLIM:-08:00:00}; NCONT=${NCONT:-2}
OPT=""; LINES=""; BEST=""
for k in 1deg_poisson 1deg_step oldu; do
    J=$(env $COMMON ${ENV[$k]} CHAIN=1 sbatch --parsable --exclude=holygpu8a19102 -t $TLIM -J r20b_$k \
        slurm/kernel_opt_run19.sh grid swgotrig 0.3 $LR $STEPS)
    OPT="$OPT:$J"
    for c in $(seq $NCONT); do
        J=$(env $COMMON ${ENV[$k]} CHAIN=1 PREV_JOB=$J sbatch --parsable --exclude=holygpu8a19102 -t $TLIM -J r20b_$k \
            -d afternotok:$J slurm/kernel_opt_run19.sh grid swgotrig 0.3 $LR $STEPS)
        OPT="$OPT:$J"
    done
    LINES="$LINES opt20b_$k=${DIR[$k]}/layout_best.pt"; BEST="$BEST ${DIR[$k]}/layout_best.pt"
done
# afterany: the continuations not needed end cancelled; the layout files must exist
LAY=$(sub -p serial_requeue -c 1 --mem=2g -t 00:10:00 -J r20b_layouts -d afterany$OPT -o $R/slurm_logs/slurm-%j-%x.out \
      --wrap "ls $BEST || exit 1; grep -v '^opt20b_' $L > $L.tmp; printf '%s\n' $LINES >> $L.tmp; mv $L.tmp $L; cat $L")
VAL=""
for rk in "deepsets validate_count" "deepsets_count valcnt_count"; do
    set -- $rk
    for seed in 1 2 3 4 5; do
        VAL="$VAL:$(sub -J r20b_val -d afterok:$LAY slurm/run20_val_chunks.sh $1 $2 $seed)"
    done
done
GH=$(sub $GPU -J r20b_ghost -c 4 --mem=64g -t 02:00:00 -d afterok:$LAY -o $R/slurm_logs/slurm-%j-%x.out --chdir=$PWD \
     --wrap "$PRE; RF=\$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)'); \
     python -u plots/analysis/run19/swgotrig/score_ghost.py --layouts $L --recon-dirs \${RF}_deepsets \${RF}_deepsets_count \
     --n-windows 20000 --out $O/ghost_scores.json.tmp_\$SLURM_JOB_ID && mv $O/ghost_scores.json.tmp_\$SLURM_JOB_ID $O/ghost_scores.json")
TAB=$(sub -p serial_requeue -c 2 --mem=16g -t 00:45:00 -J r20b_tables -d afterok$VAL:$GH -o $R/slurm_logs/slurm-%j-%x.out \
      --chdir=$PWD --wrap "source slurm/env.sh; source slurm/notify.sh; source slurm/run20_guard.sh; \
      export MPLCONFIGDIR=/tmp/mplcache_\$USER; D=plots/analysis/run19/swgotrig; python \$D/make_count_tables.py && \
      python \$D/make_triggered_tables.py && python \$D/make_ghost_tables.py && python \$D/make_compact_tables.py && \
      python \$D/make_compact_tables.py --notion && python \$D/make_compact_tables.py --only count --notion && \
      python plots/analysis/run20/plot_opt_layouts.py && python plots/analysis/run20/plot_opt_training.py --set 20b; \
      rc=\$?; notify_end \$rc; exit \$rc")
echo "opt${OPT//:/ }, layouts $LAY, val${VAL//:/ }, ghost $GH, tables $TAB"

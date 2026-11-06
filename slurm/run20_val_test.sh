#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20 validation for one recon on gpu_test (2 jobs per user there, so one job per recon instead of one per seed):
# drawn counts, count seeds 1-5 over 4 GPUs in parallel (seeds 1 and 5 share the first), held-out chunks of 5,000,
# same score files as run20_submit.sh (a chunk whose file already lists opt_cab* layouts is skipped). GHOST=1 also
# scores the ghost windows for both recons afterwards.
#   sbatch -J r20_val_default slurm/run20_val_test.sh deepsets validate_count
#   GHOST=1 sbatch -J r20_val_count slurm/run20_val_test.sh deepsets_count valcnt_count
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SFX=$1; PRE=$2
O=$(python -c 'import modules.constants as C, os; print(os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig"))')
L=$O/validate_count_layouts.txt
RF=$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)')
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
echo "devices: ${DEVS[*]}  recon ${RF}_$SFX  prefix $PRE"

seed_run () {   # seed, device
    for s in 0 5000 10000 15000 20000; do
        f=$O/${PRE}_s$1_$s.json
        grep -q opt_cab "$f" 2>/dev/null && continue
        CUDA_VISIBLE_DEVICES=$2 python -u plots/analysis/run19/triplets/score_knn.py --layouts $L --recon-dir ${RF}_$SFX \
            --counting poisson --count-seed $1 --windows 0 1000 10000 --n-events 5000 --event-start $s --kernel-chunk 16 \
            --out $f.tmp_$SLURM_JOB_ID > $O/../../slurm_logs/val_${PRE}_s$1.log 2>&1 && mv $f.tmp_$SLURM_JOB_ID $f || return 1
    done
}
( seed_run 1 ${DEVS[0]} && seed_run 5 ${DEVS[0]} ) & p1=$!
seed_run 2 ${DEVS[1]} & p2=$!
seed_run 3 ${DEVS[2]} & p3=$!
seed_run 4 ${DEVS[3]} & p4=$!
rc=0
for p in $p1 $p2 $p3 $p4; do wait $p || rc=1; done
echo "[val] $PRE done rc=$rc"
if [ "$GHOST" = 1 ]; then
    CUDA_VISIBLE_DEVICES=${DEVS[1]} python -u plots/analysis/run19/swgotrig/score_ghost.py --layouts $L \
        --recon-dirs ${RF}_deepsets ${RF}_deepsets_count --n-windows 20000 \
        --out $O/ghost_scores.json.tmp_$SLURM_JOB_ID && mv $O/ghost_scores.json.tmp_$SLURM_JOB_ID $O/ghost_scores.json || rc=1
    echo "[ghost] rc=$rc"
fi
exit $rc

#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 01:30:00
#SBATCH -J r20_snap_sameu
#SBATCH -o slurm_logs/slurm-%j-%x.out
# Run 20: the sur5snap snapshots scored with the optimised objective, surrogate and kernel response
# (scripts/diagnostics/r20_score_snaps_same_u.py), one run per MIG slice; then the figure.
#   sbatch --chdir=<checkout> slurm/run20_snap_same_u.sh
source slurm/env.sh
source slurm/notify.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')/slurm_logs
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=poisson TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0 \
       TAMBOPT_TRIG_GATED=0 TAMBOPT_TRIG_THETA=5 TAMBOPT_TRIG_WIDTH=1 TAMBOPT_TRIG_QREF=20
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <name> <env> <args>
    env CUDA_VISIBLE_DEVICES=${DEVS[$1]} $3 python -u scripts/diagnostics/r20_score_snaps_same_u.py --name $2 --chunk 16 \
        $4 > $L/snap_same_u_$2.log 2>&1
}
run 0 eall "" "--n-score 3072" & P="$!"
run 1 e500_567 "" "--n-score 3072 --e-range 5.0 5.667" & P="$P $!"
run 2 pads "TAMBOPT_CONSTRUCT=pads=0.04" "--n-score 1024" & P="$P $!"
run 3 contour "TAMBOPT_CONSTRUCT=contour=0.02" "--n-score 1024" & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
[ $rc = 0 ] && python -u plots/analysis/run20/score_snapshots.py --same-u || rc=1
notify_end $rc
exit $rc

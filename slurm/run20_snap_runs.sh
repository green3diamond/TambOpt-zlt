#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=holygpu8a19102,holygpu8a31105
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r20_snap
#SBATCH -o slurm_logs/slurm-%j-%x.out
# Run 20: 4 of the surrogate runs (sur5: 5 deg Poisson utility, SGD lr 1 m, 1,000 steps from the 100-detector grid, no
# background) rerun with the same settings and seeds, keeping the layout every 100 steps (--snap-every 100), so their
# path can be scored on the kernel (plots/analysis/run20/score_snapshots.py). 4 processes on one GPU:
#   all E, E 5.00-5.67 (8 x 2,048 loaded per step, scoring set 3,072), + pads (0.04), + contour (0.02) (4 x 2,048)
# Out-suffix sur5snap_<name>; logs <run>/slurm_logs/sur5snap_<name>.log.
#   sbatch --chdir=<checkout> slurm/run20_snap_runs.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')/slurm_logs
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=poisson TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0 \
       TAMBOPT_TRIG_GATED=0 TAMBOPT_TRIG_THETA=5 TAMBOPT_TRIG_WIDTH=1 TAMBOPT_TRIG_QREF=20
run () {  # <name> <env> <args>
    env $2 python -u scripts/04_optimize_lbfgs_kernel.py --init grid --n-det 100 --optim sgd --lr 1 --epochs 1000 \
        --batch-events 2048 --chunk 512 --score-every 25 --log-every 10 --ckpt-every 10 --snap-every 100 \
        --response surrogate $3 --out-suffix sur5snap_$1 > $L/sur5snap_$1.log 2>&1
}
run eall "" "--accum 8 --n-score 3072" & P="$!"
run e500_567 "" "--e-range 5.0 5.667 --accum 8 --n-score 3072" & P="$P $!"
run pads "TAMBOPT_CONSTRUCT=pads=0.04" "--accum 4" & P="$P $!"
run contour "TAMBOPT_CONSTRUCT=contour=0.02" "--accum 4" & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 05:00:00
#SBATCH -J r20_sgdsparse
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20: does the step trigger make SGD cluster detectors when the trigger is not saturated? As run20_sgd_test.sh (1 deg
# reward, step trigger, default recon, expected counts, no background, 4 x 2,048 showers per step, 100 steps, scored
# every 25), from two starts, at lr 1 / 3 m:
#   20-detector grid (sparse: the trigger is far from saturated)
#   tri33_r25_n99 (33 triplets of 25 m on the grid): do the clusters hold together or spread?
# Out-suffix sgdtest_step_<name>; logs <run>/slurm_logs/sgdtest_step_<name>.log.
#   sbatch slurm/run20_sgd_sparse_test.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=step TAMBOPT_TRIG_THETA=1 \
       TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5 TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0 TAMBOPT_TRIG_GATED=0
TRI=$R/diagnostics/triplets/layouts/tri33_r25_n99.pt
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <name> <lr> <init args>
    env CUDA_VISIBLE_DEVICES=${DEVS[$1]} python -u scripts/04_optimize_lbfgs_kernel.py $4 \
        --optim sgd --lr $3 --epochs 100 --batch-events 2048 --accum 4 --chunk 16 --n-score 1024 --score-every 25 \
        --log-every 10 --ckpt-every 10 --out-suffix sgdtest_step_$2 > $R/slurm_logs/sgdtest_step_$2.log 2>&1
}
run 0 n20_lr1 1 "--init grid --n-det 20" & P="$!"
run 1 n20_lr3 3 "--init grid --n-det 20" & P="$P $!"
run 2 tri25_lr1 1 "--init $TRI" & P="$P $!"
run 3 tri25_lr3 3 "--init $TRI" & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

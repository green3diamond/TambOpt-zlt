#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 05:00:00
#SBATCH -J r20_sgdcenter
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20: does SGD from the centre start move towards a spread (grid-like) layout? As run20_sgd_test.sh (1 deg reward,
# default recon, expected counts, no background, 4 x 2,048 showers per step, 100 steps, scored every 25), 100 detectors
# from --init center, Poisson and step trigger at lr 3 / 10 m.
# Out-suffix sgdtest_center_<trigger>_lr<lr>; logs <run>/slurm_logs/sgdtest_center_<trigger>_lr<lr>.log.
#   sbatch slurm/run20_sgd_center_test.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')/slurm_logs
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_THETA=1 TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5 \
       TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0 TAMBOPT_TRIG_GATED=0
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <trigger> <lr>
    n=center_$2_lr$3
    env CUDA_VISIBLE_DEVICES=${DEVS[$1]} TAMBOPT_TRIG_MODE=$2 python -u scripts/04_optimize_lbfgs_kernel.py --init center \
        --optim sgd --lr $3 --epochs 100 --batch-events 2048 --accum 4 --chunk 16 --n-score 1024 --score-every 25 \
        --log-every 10 --ckpt-every 10 --out-suffix sgdtest_$n > $L/sgdtest_$n.log 2>&1
}
run 0 poisson 3 & P="$!"
run 1 poisson 10 & P="$P $!"
run 2 step 3 & P="$P $!"
run 3 step 10 & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 05:00:00
#SBATCH -J r20_sgdtest
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20: does plain gradient ascent (--optim sgd) raise U from the grid, where Adam drifted down (runs 20b / 20c)?
# 1 deg Poisson utility, default recon, expected counts, 4 x 2,048 showers per step, 100 steps, fixed step, scored every
# 25 on the fixed 1,024-shower scoring set. 4 MIG slices: no background at lr 0.3 / 1 / 3 m, 1 us + kNN gate at lr 1 m.
# Out-suffix sgdtest_<name>; logs <run>/slurm_logs/sgdtest_<name>.log. TRIG_MODE=step: the step trigger (SWGOLO notebook)
# instead of the Poisson one; names get step_.
#   sbatch slurm/run20_sgd_test.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')/slurm_logs
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=${TRIG_MODE:-poisson} TAMBOPT_TRIG_THETA=1 \
       TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5 TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0
P2=""; [ "${TRIG_MODE:-poisson}" = step ] && P2=step_
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <name> <lr> <gated 0|1> <extra args>
    env CUDA_VISIBLE_DEVICES=${DEVS[$1]} TAMBOPT_TRIG_GATED=$4 python -u scripts/04_optimize_lbfgs_kernel.py --init grid \
        --optim sgd --lr $3 --epochs 100 --batch-events 2048 --accum 4 --chunk 16 --n-score 1024 --score-every 25 \
        --log-every 10 --ckpt-every 10 --out-suffix sgdtest_$P2$2 $5 > $L/sgdtest_$P2$2.log 2>&1
}
run 0 nobg_lr0p3 0.3 0 "" & P="$!"
run 1 nobg_lr1 1 0 "" & P="$P $!"
run 2 nobg_lr3 3 0 "" & P="$P $!"
run 3 1us_lr1 1 1 "--bg-window-ns 1000 --gate knn23" & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

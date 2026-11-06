#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=240g
#SBATCH -t 06:00:00
#SBATCH -J r20_diag
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20: why the Stage 4 utility does not improve. 4 MIG slices in parallel:
#   0: scripts/diagnostics/r20_grad_check.py (gradient signal vs noise at the grid and the optimised layout, headroom)
#   1-3: Stage 4 from the grid at lr 1 / 3 / 10, 500 steps, scored every 50 (same settings as the run 20 Poisson runs,
#        --chunk 32 for the 20 GB slice), out-suffix diag_lr<lr>
#   sbatch slurm/run20_diag.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_MODE=poisson
L=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')/slurm_logs
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=${DEVS[0]} python -u scripts/diagnostics/r20_grad_check.py > $L/diag_grad.log 2>&1 & P0=$!
i=1; PIDS=""
for lr in 1 3 10; do
    CUDA_VISIBLE_DEVICES=${DEVS[$i]} python -u scripts/04_optimize_lbfgs_kernel.py --init grid --lr $lr --epochs 500 \
        --batch-events 2048 --chunk 32 --n-score 1024 --score-every 50 --log-every 50 --ckpt-every 100 \
        --out-suffix swgotrig_diag_lr$lr > $L/diag_lr$lr.log 2>&1 & PIDS="$PIDS $!"
    i=$((i + 1))
done
rc=0
for p in $P0 $PIDS; do wait $p || rc=1; done
exit $rc

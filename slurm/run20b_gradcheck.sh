#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=200g
#SBATCH -t 02:00:00
#SBATCH -J r20b_gradcheck
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20b: is the Stage 4 gradient under 10 us background + kNN gate + drawn counts signal or noise?
# scripts/diagnostics/r20_grad_check.py on 4 MIG slices, the drawn-count recon, 8 batches of 2,048 at the grid and at
# each run's current layout (opt_state.pt), plus 8 background / count redraws of one batch:
#   0-2: the three run 20b utilities (1deg_poisson, 1deg_step, oldu) as optimised
#   3:   1deg_poisson without background, gate or drawn counts (expected counts, ungated trigger)
# Writes <run>/diagnostics/grad_check_20b_<name>.json, logs in <run>/slurm_logs/gradcheck_20b_<name>.log.
#   sbatch slurm/run20b_gradcheck.sh
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')
RC=$R/test_v6_run_03_recentered_deepsets_count
S4=$R/test_v6_run_04_optimize_lbfgs_kernel_swgotrig_l0p3_k3_a2
T=lr3_grid_count_b2048_bg10us_knn23_cnt_acc8_plat
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0
ONE="TAMBOPT_TRIG_THETA=1 TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5 TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0"
NOISE="--bg-window-ns 10000 --gate knn23 --counting poisson --draws 8"
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
run () {  # <slice> <name> <env> <args>
    env CUDA_VISIBLE_DEVICES=${DEVS[$1]} $3 python -u scripts/diagnostics/r20_grad_check.py --recon-dir $RC --opt "" \
        --chunk 16 --out $R/diagnostics/grad_check_20b_$2.json $4 > $R/slurm_logs/gradcheck_20b_$2.log 2>&1
}
run 0 1deg_poisson "$ONE TAMBOPT_TRIG_GATED=1 TAMBOPT_TRIG_MODE=poisson" \
    "$NOISE --opt-state current=${S4}_th1_gtrig_$T/opt_state.pt" & P="$!"
run 1 1deg_step "$ONE TAMBOPT_TRIG_GATED=1 TAMBOPT_TRIG_MODE=step" \
    "$NOISE --opt-state current=${S4}_step_th1_gtrig_$T/opt_state.pt" & P="$P $!"
run 2 oldu "TAMBOPT_TRIG_OFF=1 TAMBOPT_TRIG_ENERGY=1 TAMBOPT_TRIG_GATED=0 TAMBOPT_TRIG_MODE=poisson" \
    "$NOISE --opt-state current=${S4}_E_notrig_$T/opt_state.pt" & P="$P $!"
run 3 1deg_poisson_nonoise "$ONE TAMBOPT_TRIG_GATED=0 TAMBOPT_TRIG_MODE=poisson" \
    "--opt-state current=${S4}_th1_gtrig_$T/opt_state.pt" & P="$P $!"
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

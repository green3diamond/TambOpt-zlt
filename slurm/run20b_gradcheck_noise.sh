#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:3
#SBATCH -c 12
#SBATCH --mem=150g
#SBATCH -t 01:30:00
#SBATCH -J r20b_gcnoise
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20b: which noise source makes the Stage 4 gradient noise? scripts/diagnostics/r20_grad_check.py at the grid only,
# 1 deg Poisson utility, drawn-count recon, one config per MIG slice. Config = <bg ns>:<counting none|poisson>:<gate none|knn23>
# (trigger on read-out detectors only when gated; 8 redraws of one batch when there is background or drawn counts).
# Writes <run>/diagnostics/grad_check_20b_noise_<bg>_<counting>_<gate>.json, logs in <run>/slurm_logs/.
#   sbatch slurm/run20b_gradcheck_noise.sh 0:poisson:none 0:poisson:knn23 125:poisson:knn23
# RECON=default: the default recon (<RECON_FOLDER>_deepsets) instead of the drawn-count fine-tune; names get _default.
source slurm/env.sh
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$(python -c 'import modules.constants as C; print(C.RUN_LOCATION)')
RF=$(python -c "import modules.constants as C; print(C.RECON_FOLDER)")
RC=$R/test_v6_run_03_recentered_deepsets_count; TAG=""
[ "$RECON" = default ] && RC=${RF}_deepsets TAG=_default
export TAMBOPT_UTILITY=swgo_trigger TAMBOPT_TWO_TIER_LAMBDA=0.3 TAMBOPT_TRIG_K=3 TAMBOPT_TRIG_AREA=2 \
       TAMBOPT_TRIG_MU0=1.0 TAMBOPT_TRIG_SLOPE=10.0 TAMBOPT_TRIG_MODE=poisson TAMBOPT_TRIG_THETA=1 \
       TAMBOPT_TRIG_WIDTH=0.2 TAMBOPT_TRIG_QREF=5 TAMBOPT_TRIG_ENERGY=0 TAMBOPT_TRIG_OFF=0
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
i=0; P=""
for c in "$@"; do
    IFS=: read -r bg cnt gate <<< "$c"
    name=noise_${bg}_${cnt}_${gate}$TAG
    env CUDA_VISIBLE_DEVICES=${DEVS[$i]} TAMBOPT_TRIG_GATED=$([ "$gate" = none ] && echo 0 || echo 1) \
        python -u scripts/diagnostics/r20_grad_check.py --recon-dir $RC --opt "" \
        --chunk 16 --bg-window-ns $bg --gate $gate --counting $cnt --draws 8 \
        --out $R/diagnostics/grad_check_20b_$name.json > $R/slurm_logs/gradcheck_20b_$name.log 2>&1 & P="$P $!"
    i=$((i + 1))
done
rc=0
for p in $P; do wait $p || rc=1; done
exit $rc

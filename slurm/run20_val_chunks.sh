#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=holygpu8a19102
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20 validation of one recon and one count seed on gpu_requeue: drawn counts, windows 0 / 1 / 10 us, held-out
# chunks of 5,000 (score_knn.py, same files as run20_submit.sh). A chunk whose file already scores every layout of
# validate_count_layouts.txt is skipped, so a requeue or a rerun after adding layouts only does what is missing.
# KCHUNK = showers per kernel pass (default 64; 16 fits a 20 GB MIG slice).
#   RUN_GUARD=run20 sbatch -J r20_val slurm/run20_val_chunks.sh <recon suffix: deepsets | deepsets_count> <prefix> <seed>
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SFX=$1; PRE=$2; SEED=$3
O=$(python -c 'import modules.constants as C, os; print(os.path.join(C.RUN_LOCATION, "diagnostics", "swgotrig"))')
L=$O/validate_count_layouts.txt
RF=$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)')
complete () {   # file scores every layout of $L
    python -c 'import json, sys
d = json.load(open(sys.argv[1]))["scores"]
sys.exit(0 if all(l.split("=", 1)[0] in d for l in open(sys.argv[2]) if "=" in l) else 1)' "$1" "$L" 2>/dev/null
}
for s in 0 5000 10000 15000 20000; do
    f=$O/${PRE}_s${SEED}_$s.json
    complete "$f" && continue
    python -u plots/analysis/run19/triplets/score_knn.py --layouts $L --recon-dir ${RF}_$SFX --counting poisson \
        --count-seed $SEED --windows 0 1000 10000 --n-events 5000 --event-start $s --kernel-chunk ${KCHUNK:-64} \
        --out $f.tmp_$SLURM_JOB_ID && mv $f.tmp_$SLURM_JOB_ID $f || exit 1
done

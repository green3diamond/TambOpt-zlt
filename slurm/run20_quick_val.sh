#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=holygpu8a19102
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 20 quick tests: validate <run 20>/diagnostics/quick/layouts.txt (plots/analysis/run20/export_final_layouts.py) on one
# held-out chunk of 5,000 showers with one recon: drawn counts (count seed 1), windows 0 / 1 / 10 us, all read out and
# kNN gate, per energy band (--e-bins 5 5.667 6.333 7). Writes diagnostics/quick/val_<recon suffix>_<start>.json;
# skips a chunk already written.
# QTAG=quick5: the 5 deg tests' list and outputs (diagnostics/quick5).
#   RUN_GUARD=run20 sbatch -J r20_quick_val slurm/run20_quick_val.sh <deepsets | deepsets_count> <event start>
source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run20_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SFX=$1; S=$2
O=$(python -c 'import modules.constants as C, os; print(os.path.join(C.RUN_LOCATION, "diagnostics", os.environ.get("QTAG", "quick")))')
RF=$(python -c 'import modules.constants as C; print(C.RECON_FOLDER)')
f=$O/val_${SFX}_$S.json
[ -f "$f" ] && { echo "exists: $f"; exit 0; }
python -u plots/analysis/run19/triplets/score_knn.py --layouts $O/layouts.txt --recon-dir ${RF}_$SFX --counting poisson \
    --count-seed 1 --windows 0 1000 10000 --n-events 5000 --event-start $S --e-bins 5 5.667 6.333 7 \
    --out $f.tmp_$SLURM_JOB_ID && mv $f.tmp_$SLURM_JOB_ID $f

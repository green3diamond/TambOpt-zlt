#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH -c 8
#SBATCH --mem=96g
#SBATCH --time=0-12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J r19_count
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k/diagnostics/counting/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt/.claude/worktrees/bg-kernel-units
# Counting noise on run 19 (branch bg-kernel-units, modules/optimize/counting.py). MODE (1st argument):
#   smoke     grid + 6 km disk, 5,000 showers, windows 0 / 1 us, --counting none and poisson
#   score     background test layouts (score_noise) and the kNN gate (score_knn), --counting poisson, clean recon
#   finetune  03c --counting poisson -> <RECON_FOLDER>_deepsets_count (resumes from finetune_resume.pt after a requeue)
#   rescore   score again with the fine-tuned recon
# Outputs in <RUN_LOCATION>/diagnostics/counting/; each step is skipped if its output exists, so a requeue resumes.
# Requeue partition: no --mail-type; notify_end submits a 1-minute job at the end.
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
OUT=$R/diagnostics/counting
FT=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RECON_FOLDER + "_deepsets_count")')
MODE=${1:?mode: smoke | score | finetune | rescore}
step () { local out="$1"; shift; [ -s "$out" ] && { echo ">>> skip ($out exists)"; return 0; }; echo ">>> $*"; "$@"; }
case "$MODE" in
smoke)
    for c in none poisson; do
        step $OUT/smoke_noise_$c.json python -u plots/analysis/run19/noise/score_noise.py --layouts $OUT/smoke_layouts.txt \
            --no-opt --windows 0 1000 --counting $c --out $OUT/smoke_noise_$c.json || break
    done ;;
score)
    step $OUT/noise_scores_poisson.json python -u plots/analysis/run19/noise/score_noise.py --counting poisson \
        --out $OUT/noise_scores_poisson.json &&
    step $OUT/knn_scores_poisson.json python -u plots/analysis/run19/triplets/score_knn.py --counting poisson \
        --out $OUT/knn_scores_poisson.json ;;
finetune)
    python -u scripts/03c_finetune_recon_noise.py --counting poisson ;;
rescore)
    step $OUT/noise_scores_poisson_ft.json python -u plots/analysis/run19/noise/score_noise.py --counting poisson \
        --recon-dir $FT --out $OUT/noise_scores_poisson_ft.json &&
    step $OUT/knn_scores_poisson_ft.json python -u plots/analysis/run19/triplets/score_knn.py --counting poisson \
        --recon-dir $FT --out $OUT/knn_scores_poisson_ft.json ;;
*) echo "unknown mode $MODE"; exit 1 ;;
esac
rc=$?; notify_end $rc; exit $rc

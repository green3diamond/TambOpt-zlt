#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=48g
#SBATCH --time=1:00:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -J true_utility_only
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Standalone surrogate-U vs true-U comparison for whichever stage-4 layouts
# already exist — read-only apart from its own output file, so it is safe to run
# alongside the main pipeline job (that job is writing the _center dir; this
# reads _grid). Output goes to true_utility_eval_<jobid>.txt rather than the
# pipeline's own true_utility_eval.txt so neither clobbers the other.
source slurm/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION; print(RUN_LOCATION)')"
A="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_ensemble_full_corpus"

for SCHEME in grid center; do
    DIR="${A}_${SCHEME}"
    if [ ! -f "$DIR/layout_best.pt" ]; then
        echo ">>> skip $SCHEME — no layout_best.pt yet"
        continue
    fi
    FLAG=""
    [ "$SCHEME" = "grid" ] && FLAG="--grid-layout"
    echo "========================================================================"
    echo ">>> true_utility  scheme=$SCHEME  (recon trained with --recon-input mix)"
    echo "========================================================================"
    python -u plots/layouts/true_utility.py $FLAG \
        --layout "$DIR/layout_best.pt" 2>&1 \
        | tee "$DIR/true_utility_eval_${SLURM_JOB_ID}.txt"
done

#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g        			
#SBATCH --time=2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J run_all_script_batch
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# sigma-10 experiment: SIGMA_SPATIAL 50 -> 10 m and RECONSTRUCT_THRESHOLD
# 10 -> 5 (both modules/constants.py + optimize/objective.py), on a tenth of the
# corpus (DATASET_FRACTION = 0.10). Runs against RUN_LOCATION =
# 14_sigma10_10pct, which holds the shower cache MOVED out of
# 13_full_750k_no_sampling -- Stage 0 is not rerun, but Stage 1 IS, because
# SIGMA_SPATIAL enters the kernel labels at build time and every E.pt/T.pt
# written before this change is stale.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

# Own status file: distinct from the production pipeline's, whose keys are
# already marked "done" under different flags.
#
# It ships pre-seeded with 00_generate_data_dual_species.py = "done". Stage 0 is
# what writes RUN_LOCATION/v6_run_00, and this run's cache was MOVED in from
# 13_full_750k_no_sampling rather than generated -- regenerating it would be
# ~284 GB and days of GPU for a corpus that already exists. This runner does not
# call Stage 0 at all, so the key is belt-and-braces; if you ever reset the file
# to '{}', put that key back before adding a Stage 0 step.
STATUS_FILE="pipeline_status.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

_status_done() {
    python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$1')=='done' else 1)"
}
_status_mark_done() {
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$1']='done'; json.dump(d, open('$STATUS_FILE.tmp','w'), indent=2); os.replace('$STATUS_FILE.tmp','$STATUS_FILE')"
}

run_step () {
    local step="$1"; shift
    if _status_done "$step"; then
        echo ">>> Skipping $step (already done)"
        return 0
    fi
    echo ">>> Running $step $*"
    python -u "scripts/$step" "$@" || exit $?
    _status_mark_done "$step"
}

# Every step below is keyed and checkpointed against $STATUS_FILE -- an
# already-"done" key is skipped outright, so a re-submit of this script picks
# up wherever the last run left off instead of redoing finished work.
run_once () {
    local key="$1"; shift
    if _status_done "$key"; then
        echo ">>> Skipping $key (already done)"; return 0
    fi
    echo ">>> Running $key"
    "$@" || exit $?
    _status_mark_done "$key"
}

# Stage 1: rebuild the corpus. NOT part of this runner before the sigma change
# -- it is here because SIGMA_SPATIAL is baked into the labels by
# compute_labels_batch, so the surrogate cannot be retrained on the old E.pt/T.pt.
run_step 01_build_dataset_northeast.py

# Stage 2: mean-only regressor, no variance head trained.
run_step 02_train_fnn_deepsets.py

# Stage 3: Stage-2 mean input, mixed 0.7 mean / 0.3 real kernel labels.
run_step 03_train_recon_deepsets.py \
    --recon-input mix --mix-frac 0.3


# Stage 4, production ensemble optimizer (against the frozen recon/surrogate,
# not the kernel) -- writes OPT_FOLDER + "_lbfgs_ensemble_full_corpus_{grid,center}",
# same convention run_all_script_batch.sh's eval section already expects.
run_step 04_optimize_lbfgs_ensemble.py --chains 6

A="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_ensemble_full_corpus"

for scheme in grid center; do
    DIR="${A}_${scheme}"
    flag=""; [ "$scheme" = grid ] && flag="--grid-layout"

    run_once "eval_true_utility_${scheme}" bash -c \
        "python -u plots/layouts/true_utility.py $flag \
             --layout \"\$0/layout_best.pt\" 2>&1 | tee \"\$0/true_utility_eval.txt\"" \
        "$DIR"

    run_once "eval_utility_vs_energy_${scheme}" \
        python -u plots/layouts/utility_vs_energy.py $flag --run-dir "$DIR"
done

run_once eval_traj_adam_first500 \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "${A}_center" "${A}_grid" \
        --only zoom --zoom-epochs 500 \
        -o "$(dirname "$A")/ensemble_distinct_trajectory.mp4"

run_once eval_traj_lbfgs_chunks \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "${A}_center" "${A}_grid" \
        --phase lbfgs --lbfgs-chunk 0-9 --fit-view --min-per-chunk 0 \
        --zoom-epochs 0 --seconds 20 --fps 60 \
        -o "$(dirname "$A")/ensemble_lbfgs_chunk0-9.mp4"
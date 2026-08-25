#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g        			
#SBATCH --time=20:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J run_all_script_batch
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

source slurm/env.sh

# --- checkpointing: steps marked done in pipeline_status.json are skipped ---
# Delete the file (or a step's entry) to force a rerun.
# -s not -f: the file can exist but be EMPTY (a preempted gpu_requeue job killed
# between open(...,"w") truncating it and json.dump refilling it). -f accepted
# the 0-byte file, so every json.load below threw and no step was ever marked
# done -- the whole pipeline re-ran from scratch every time.
STATUS_FILE="pipeline_status.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

run_step () {
    local step="$1"; shift
    # Status-file key stays the bare script name (matches existing
    # pipeline_status*.json entries); the file itself now lives in scripts/.
    local script_path="scripts/$step"
    if python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$step')=='done' else 1)"; then
        echo ">>> Skipping $step (already done)"
        return 0
    fi
    echo ">>> Running $step $*"
    python -u "$script_path" "$@" || exit $?
    # write to a temp file then os.replace (atomic): a preemption can no longer
    # leave the status file truncated to 0 bytes.
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$step']='done'; json.dump(d, open('$STATUS_FILE.tmp','w'), indent=2); os.replace('$STATUS_FILE.tmp','$STATUS_FILE')"
}

# Step 0 now resumes automatically (progress.json next to each output corpus,
# per species) — a preempted run just needs the same command re-run, no manual
# row/offset bookkeeping. --n-pairs 0 = all in-band tau events (~751,931);
# Step 0 itself splits off HOLDOUT_FRAC (5%) into a separate corpus before
# generating, so this is 750k-scale total, not 750k into training.
run_step 00_generate_data_dual_species.py --n-pairs 70_000
run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py
run_step 03_train_recon_deepsets.py
run_step 04_optimize_lbfgs_ensemble.py --chains 1

# --- evaluation ------------------------------------------------------------
# Same checkpointing as run_step, but for entry points outside scripts/ (the
# plots/ evaluators take a full command rather than a bare script name, since
# each needs its own flags and per-scheme layout paths).
run_eval () {
    local key="$1"; shift
    if python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$key')=='done' else 1)"; then
        echo ">>> Skipping $key (already done)"
        return 0
    fi
    echo ">>> Running $key"
    "$@" || exit $?
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$key']='done'; json.dump(d, open('$STATUS_FILE.tmp','w'), indent=2); os.replace('$STATUS_FILE.tmp','$STATUS_FILE')"
}

# Step 4 writes OPT_FOLDER + "_lbfgs_ensemble_full_corpus_{scheme}" per scheme
# (04_optimize_lbfgs_ensemble.py's OPT_DIR_TEMPLATE), for both schemes in its
# default --schemes grid,center. Resolve the run root from constants so this
# follows RUN_LOCATION rather than repeating the path.
R="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import RUN_LOCATION; print(RUN_LOCATION)')"
A="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_ensemble_full_corpus"
GDIR="${A}_grid"
CDIR="${A}_center"

# 1) True utility: surrogate-U vs kernel-U for each scheme's optimized layout,
#    against that scheme's own baseline. --grid-layout picks the grid baseline;
#    omitting it uses center. Tee into the run dir so the table lives beside the
#    layout it scored, not only in this job's slurm log.
run_eval eval_true_utility_grid bash -c \
    'python -u plots/layouts/true_utility.py --grid-layout \
         --layout "$0/layout_best.pt" 2>&1 | tee "$0/true_utility_eval.txt"' \
    "$GDIR"

run_eval eval_true_utility_center bash -c \
    'python -u plots/layouts/true_utility.py \
         --layout "$0/layout_best.pt" 2>&1 | tee "$0/true_utility_eval.txt"' \
    "$CDIR"

# 2) Per-event utility vs primary energy; writes utility_vs_energy.png into
#    each run dir. Both schemes in one call so they share the loaded models.
run_eval eval_utility_vs_energy \
    python -u plots/layouts/utility_vs_energy.py --run-dir \
        "$CDIR" "$GDIR"

# 3) Trajectory videos, center and grid side by side in each. CPU-only work on
#    a GPU node, but it is ~12 min and keeps the pipeline to one job.
#    Adam warm-start, first 500 epochs. --only zoom skips the full-trajectory
#    render; that pass appends _first500ep to the -o stem.
run_eval eval_traj_adam_first500 \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "$CDIR" "$GDIR" \
        --only zoom --zoom-epochs 500 \
        -o "$R/ensemble_distinct_trajectory.mp4"

#    L-BFGS sweep chunks 0-9. --min-per-chunk 0 because the per-chunk floor is
#    pointless over a 10-chunk range, and --fit-view drops the strong-Wolfe
#    probes that fling the layout off-map so the axes can fit the detectors.
run_eval eval_traj_lbfgs_chunks \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "$CDIR" "$GDIR" \
        --phase lbfgs --lbfgs-chunk 0-9 --fit-view --min-per-chunk 0 \
        --zoom-epochs 0 --seconds 20 --fps 60 \
        -o "$R/ensemble_lbfgs_chunk0-9.mp4"

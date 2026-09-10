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
source slurm/gpu_gate.sh
gpu_health_gate

# The muon holdout generation batch (00_generate_data_dual_species.py) hit a
# late-run CUDA OOM twice (jobs 42240054, 42469213) despite its own batch-size
# back-off bottoming out at batch_size=1 -- the error showed 11+ GiB "reserved
# but unallocated", i.e. allocator fragmentation from ~8000s of varying-shape
# point-cloud tensors, not true memory exhaustion. expandable_segments lets
# the allocator grow existing segments instead of fragmenting into new ones.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- checkpointing: steps marked done in pipeline_status.json are skipped ---
# Delete the file (or a step's entry) to force a rerun.
# -s not -f: the file can exist but be EMPTY (a preempted gpu_requeue job killed
# between open(...,"w") truncating it and json.dump refilling it). -f accepted
# the 0-byte file, so every json.load below threw and no step was ever marked
# done -- the whole pipeline re-ran from scratch every time.
STATUS_FILE="pipeline_status.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

# Shared by run_step/run_step_as/run_eval below -- write to a temp file then
# os.replace (atomic): a preemption can no longer leave the status file
# truncated to 0 bytes.
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

# Step 0 now resumes automatically (progress.json next to each output corpus,
# per species) — a preempted run just needs the same command re-run, no manual
# row/offset bookkeeping. --n-pairs 0 = all in-band tau events (~751,931);
# Step 0 itself splits off HOLDOUT_FRAC (5%) into a separate corpus before
# generating, so this is 750k-scale total, not 750k into training.
run_step 00_generate_data_dual_species.py #--n-pairs 70_000
run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py
# --recon-input mix: recon trains on a 70/30 kernel-weighted per-row mix of the
# surrogate draw and the REAL kernel labels Step 1 wrote. Pure "sample" (the old
# default) only ever showed recon surrogate-generated responses while
# true_utility.py scores it on kernel ones -- a train/eval mismatch worth ~25 U
# of standing artifact gap, present even on the UN-optimized grid baseline
# (job 42695512: surrogate-U 47.94 vs true-U 22.78). Stage 4 still feeds recon
# surrogate output, so mix rather than pure kernel keeps it good at both.
run_step 03_train_recon_deepsets.py --recon-input mix --mix-frac 0.7
run_step 04_optimize_lbfgs_ensemble.py --chains 6

# --- evaluation ------------------------------------------------------------
# run_step's twin for the plots/ evaluators: a full command, not a bare script.
run_eval () {
    local key="$1"; shift
    if _status_done "$key"; then
        echo ">>> Skipping $key (already done)"
        return 0
    fi
    echo ">>> Running $key"
    "$@" || exit $?
    _status_mark_done "$key"
}

# Step 4's own output dirs (OPT_DIR_TEMPLATE): OPT_FOLDER + "_lbfgs_ensemble_full_corpus_{scheme}".
read -r R A < <(python -c '
import sys; sys.path.insert(0, ".")
from modules.constants import RUN_LOCATION, OPT_FOLDER
print(RUN_LOCATION, OPT_FOLDER)')
A="${A}_lbfgs_ensemble_full_corpus"
GDIR="${A}_grid"
CDIR="${A}_center"

# --grid-layout scores each scheme against its OWN baseline (omitted = center).
for scheme in grid center; do
    DIR="${A}_${scheme}"
    flag=""; [ "$scheme" = grid ] && flag="--grid-layout"

    # surrogate-U vs kernel-U; tee so the table lands beside the layout it scored.
    run_eval "eval_true_utility_${scheme}" bash -c \
        "python -u plots/layouts/true_utility.py $flag \
             --layout \"\$0/layout_best.pt\" 2>&1 | tee \"\$0/true_utility_eval.txt\"" \
        "$DIR"

    # per-event utility vs primary energy -> utility_vs_energy.png
    run_eval "eval_utility_vs_energy_${scheme}" \
        python -u plots/layouts/utility_vs_energy.py $flag --run-dir "$DIR"
done

# Trajectory videos. --only zoom = Adam phase only, first 500 epochs.
run_eval eval_traj_adam_first500 \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "$CDIR" "$GDIR" \
        --only zoom --zoom-epochs 500 \
        -o "$R/ensemble_distinct_trajectory.mp4"

# L-BFGS phase, chunks 0-9. --fit-view drops the strong-Wolfe probes that
# fling the layout off-map so the axes fit the detectors.
run_eval eval_traj_lbfgs_chunks \
    python -u plots/layouts/04_trajectory_gif.py \
        --run-dir "$CDIR" "$GDIR" \
        --phase lbfgs --lbfgs-chunk 0-9 --fit-view --min-per-chunk 0 \
        --zoom-epochs 0 --seconds 20 --fps 60 \
        -o "$R/ensemble_lbfgs_chunk0-9.mp4"

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

# --- GPU health gate -------------------------------------------------------
# gpu_requeue preemption can hand the requeued job the SAME node it was just
# preempted from, while our own killed process still pins the GPU: every CUDA
# alloc then fails with "CUDA-capable device(s) is/are busy or unavailable"
# (jobs 42556695, 43192939, 43844284, 43857690, 44186645). Step 0 streams into
# a ~100 GB corpus file on lustre, so a SIGKILLed writer can sit in
# uninterruptible I/O for minutes.
#
# Waiting on the node is the WRONG trade. Measured: the stuck writer survived
# 210s of waiting (job 44186645), while a requeue turns the job around in ~3
# min (44186645 preempted 23:57:23, restarted 00:00:24) AND re-rolls the node,
# which is the only thing that actually clears it. So the probe budget is kept
# just long enough to ride out a genuine quick race, and anything longer is
# handed to a requeue.
THIS_NODE="$(hostname -s)"
GATE_FAILS=".gpu_gate_fails"     # consecutive gate-triggered requeues

# Poll inside ONE python process: `import torch` alone costs ~30s, so a shell
# retry loop would burn its budget on imports. Exit 0 = usable, 1 = still busy.
python - <<'PY'
import sys, time, torch
DEADLINE, STEP = 90, 30
t0 = time.time()
while True:
    try:
        torch.zeros(1, device="cuda"); torch.cuda.synchronize(); sys.exit(0)
    except RuntimeError as e:
        if time.time() - t0 > DEADLINE:
            print(f"[gpu-gate] still busy after {DEADLINE}s: {e}", flush=True)
            sys.exit(1)
        print(f"[gpu-gate] busy at t+{time.time()-t0:.0f}s, retrying", flush=True)
        time.sleep(STEP)
PY

if [ $? -ne 0 ]; then
    # Count CONSECUTIVE gate failures, not total restarts: SLURM_RESTART_COUNT
    # also counts ordinary preemptions (6 in 10.5h on this partition), so
    # capping on it conflates "unlucky node once" with "GPU broken everywhere"
    # and lets the worst case balloon. The streak resets on any good GPU below.
    n=$(( $(cat "$GATE_FAILS" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$GATE_FAILS"
    if [ "$n" -ge 5 ]; then
        echo ">>> GPU busy on $n consecutive nodes -- real fault, giving up"
        rm -f "$GATE_FAILS"
        exit 1
    fi
    # We cannot blacklist the node: `scontrol update ExcNodeList` is REFUSED on
    # a RUNNING job ("Job is no longer pending execution", verified in job
    # 44336590), and the script is dead once the requeue lands. A bare requeue
    # re-rolls the node anyway; the probe re-runs on whatever we get next.
    echo ">>> GPU busy on ${THIS_NODE} (strike $n/5) -- requeueing to re-roll the node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 120        # let the controller act before the script would exit
    exit 0
fi
rm -f "$GATE_FAILS"
echo ">>> GPU OK on ${THIS_NODE} (restart #${SLURM_RESTART_COUNT:-0})"

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

# 2) Per-event utility vs primary energy; writes utility_vs_energy.png into each
#    run dir. Each scheme now overlays its OWN baseline (--grid-layout picks the
#    grid one, omitting it uses center — same convention as the true_utility
#    calls above), so this is one call per scheme rather than both in one: a
#    single call can only carry one baseline, and pairing the grid layout against
#    a center baseline would make the ΔU in its title meaningless. Both curves
#    are scored on kernel labels (the script's default), so the numbers match the
#    true-U column of the true_utility tables above rather than the surrogate's.
run_eval eval_utility_vs_energy_grid \
    python -u plots/layouts/utility_vs_energy.py --grid-layout --run-dir "$GDIR"

run_eval eval_utility_vs_energy_center \
    python -u plots/layouts/utility_vs_energy.py --run-dir "$CDIR"

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

#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=20:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J run_10pct_mean_analytic
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# stage3-analytic-validation experiment: 10%-data (TRAIN_FRACTION in
# modules/constants.py), mean-only Stage 2, analytic Stage 4. Runs against
# RUN_LOCATION = 12_10pct_mean_analytic, whose Stage 0/1 outputs were copied
# verbatim from the production run (not regenerated here).

source slurm/env.sh

# --- GPU health gate: a requeued job can land on the same node it was just
# preempted from, still holding the GPU (see run_all_script_batch.sh for the
# full incident history) ---
THIS_NODE="$(hostname -s)"
GATE_FAILS=".gpu_gate_fails"
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
    n=$(( $(cat "$GATE_FAILS" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$GATE_FAILS"
    if [ "$n" -ge 5 ]; then
        echo ">>> GPU busy on $n consecutive nodes -- giving up"
        rm -f "$GATE_FAILS"; exit 1
    fi
    echo ">>> GPU busy on ${THIS_NODE} (strike $n/5) -- requeueing"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 120; exit 0
fi
rm -f "$GATE_FAILS"
echo ">>> GPU OK on ${THIS_NODE} (restart #${SLURM_RESTART_COUNT:-0})"

# Own status file: distinct from the production pipeline's, whose keys are
# already marked "done" under different flags.
STATUS_FILE="pipeline_status_10pct_mean_analytic.json"
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
        echo ">>> Skipping $step (already done)"; return 0
    fi
    echo ">>> Running $step $*"
    python -u "scripts/$step" "$@" || exit $?
    _status_mark_done "$step"
}

# Stage 2: mean-only regressor, no variance head trained.
run_step 02_train_fnn_deepsets.py --nll-warmup-frac 1.0

# Stage 3: Stage-2 mean input, mixed 0.7 mean / 0.3 real kernel labels.
run_step 03_train_recon_deepsets.py \
    --recon-input mix --mix-frac 0.3 --fnn-mean-only

# Stage 4, kernel-in-the-loop, two utility variants -- called directly
# (not run_step: same script twice would collide on its status key).
# --out-suffix keeps their output dirs apart.
python -u scripts/04_optimize_lbfgs_kernel.py \
    --utility recon --out-suffix recon_utility --n-events 64 --chunk 8 --epochs 200

python -u scripts/04_optimize_lbfgs_kernel.py \
    --utility analytic --out-suffix analytic_utility --n-events 64 --chunk 8 --epochs 200

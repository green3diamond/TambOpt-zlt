#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=100g
#SBATCH --time=2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J run16_pipeline
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt/.claude/worktrees/run16-emean

# Run 16 (branch run16-emean, its own worktree so run-15 jobs on the main
# checkout are untouched): run 15's config (sigma=10, interpolated kernel k=8,
# lit/dark hurdle Stage 2, 10% data) with KERNEL_E_MODE="mean" -- a detector's E
# is the kernel-weighted mean point energy instead of the energy density.
# Stage 0 is run 15's (v6_run_00 is a symlink). Stages 1-2 only.
#
# Own status file; gpu_requeue preemptions resume from the last finished step.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

STATUS_FILE="pipeline_status_run16.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"

_status_done() {
    python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$1')=='done' else 1)"
}
_status_mark_done() {
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$1']='done'; json.dump(d, open('$STATUS_FILE.tmp','w'), indent=2); os.replace('$STATUS_FILE.tmp','$STATUS_FILE')"
}
run_step () {
    local step="$1"; shift
    if _status_done "$step"; then echo ">>> Skipping $step (already done)"; return 0; fi
    echo ">>> Running $step $*"
    python -u "scripts/$step" "$@" || exit $?
    _status_mark_done "$step"
}

# Refuse to run against the wrong run or config.
python - <<'EOF' || exit 1
import sys; sys.path.insert(0, ".")
from modules.constants import (RUN_LOCATION, KERNEL_INTERP_K, KERNEL_E_MODE,
                               SURROGATE_HURDLE, SIGMA_SPATIAL, DATASET_FRACTION)
print(f"RUN_LOCATION={RUN_LOCATION}\nsigma={SIGMA_SPATIAL} interp_k={KERNEL_INTERP_K} "
      f"e_mode={KERNEL_E_MODE} hurdle={SURROGATE_HURDLE} dataset_fraction={DATASET_FRACTION}")
ok = (RUN_LOCATION.rstrip("/").endswith("16_sigma10_interp_emean_10pct") and KERNEL_INTERP_K == 8
      and KERNEL_E_MODE == "mean" and SURROGATE_HURDLE and SIGMA_SPATIAL == 10.0
      and DATASET_FRACTION == 0.10)
sys.exit(0 if ok else "constants do not match run 16")
EOF

run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py

# Stage-2 figures (fnn_<species>_target_vs_pred.png / _conditional.png, median
# line), written next to the FNN checkpoints in test_v6_run_02_recentered/.
if ! _status_done plot_fnn_conditional; then
    echo ">>> Running plot_fnn_conditional"
    python -u plots/training/02_nn_target_vs_pred.py --dual --fnn-only || exit $?
    _status_mark_done plot_fnn_conditional
fi

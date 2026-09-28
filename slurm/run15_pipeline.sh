#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=100g
#SBATCH --time=2-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --open-mode=append
#SBATCH -J run15_pipeline
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Run 15 (branch run15-interp-hurdle): sigma=10 with the interpolated kernel
# (KERNEL_INTERP_K=8, leading-edge T) and the lit/dark hurdle Stage 2
# (SURROGATE_HURDLE=True), on DATASET_FRACTION=0.10. RUN_LOCATION =
# 15_sigma10_interp_hurdle_10pct, whose v6_run_00 (Stage 0) was moved from run 14
# (run 14 keeps a symlink to it). Stage 1 is rebuilt: labels depend on the kernel.
#
# Own status file: pipeline_status.json already marks run 14's steps done, which
# would make every step below skip. Delete a key to force that step to rerun;
# gpu_requeue preemptions resume from the last finished step.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

STATUS_FILE="pipeline_status_run15.json"
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
run_once () {
    local key="$1"; shift
    if _status_done "$key"; then echo ">>> Skipping $key (already done)"; return 0; fi
    echo ">>> Running $key"
    "$@" || exit $?
    _status_mark_done "$key"
}

# Refuse to run against the wrong run or config.
python - <<'EOF' || exit 1
import sys; sys.path.insert(0, ".")
from modules.constants import RUN_LOCATION, KERNEL_INTERP_K, SURROGATE_HURDLE, SIGMA_SPATIAL, DATASET_FRACTION
print(f"RUN_LOCATION={RUN_LOCATION}\nsigma={SIGMA_SPATIAL} interp_k={KERNEL_INTERP_K} "
      f"hurdle={SURROGATE_HURDLE} dataset_fraction={DATASET_FRACTION}")
ok = (RUN_LOCATION.rstrip("/").endswith("15_sigma10_interp_hurdle_10pct") and KERNEL_INTERP_K == 8
      and SURROGATE_HURDLE and SIGMA_SPATIAL == 10.0 and DATASET_FRACTION == 0.10)
sys.exit(0 if ok else "constants do not match run 15 -- wrong branch checked out?")
EOF

run_step 01_build_dataset_northeast.py
run_step 02_train_fnn_deepsets.py
run_step 03_train_recon_deepsets.py --recon-input mix --mix-frac 0.3
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

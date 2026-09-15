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
source slurm/gpu_gate.sh
gpu_health_gate

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

# Stage 2: mean-only regressor, no variance head trained.
run_step 02_train_fnn_deepsets.py --nll-warmup-frac 1.0

# Stage 3: Stage-2 mean input, mixed 0.7 mean / 0.3 real kernel labels.
run_step 03_train_recon_deepsets.py \
    --recon-input mix --mix-frac 0.3 --fnn-mean-only

# Stage 4, kernel-in-the-loop, two utility variants. --out-suffix keeps their
# output dirs apart; explicit keys since both calls are the same script.
run_once kernel_recon_utility \
    python -u scripts/04_optimize_lbfgs_kernel.py \
        --utility recon --out-suffix recon_utility --n-events 64 --chunk 8 --epochs 200

run_once kernel_analytic_utility \
    python -u scripts/04_optimize_lbfgs_kernel.py \
        --utility analytic --out-suffix analytic_utility --n-events 64 --chunk 8 --epochs 200

# true_utility/utility_vs_energy work against either optimizer's layout_best.pt
# (same {x, y} dict format), so they apply here too -- no --grid-layout since
# both kernel-in-the-loop runs above use --init's default (center). No
# trajectory video: 04_optimize_lbfgs_kernel.py never writes a trajectory.pt.
K="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_kernel"

for variant in recon_utility analytic_utility; do
    DIR="${K}_${variant}"

    run_once "eval_true_utility_kernel_${variant}" bash -c \
        "python -u plots/layouts/true_utility.py \
             --layout \"\$0/layout_best.pt\" 2>&1 | tee \"\$0/true_utility_eval.txt\"" \
        "$DIR"

    run_once "eval_utility_vs_energy_kernel_${variant}" \
        python -u plots/layouts/utility_vs_energy.py --run-dir "$DIR"
done

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

# Stage 4, component-balanced ensemble (equal θ/φ/E shares).
run_step 04_optimize_lbfgs_balanced.py --chains 6

B="$(python -c 'import sys; sys.path.insert(0, "."); from modules.constants import OPT_FOLDER; print(OPT_FOLDER)')_lbfgs_balanced"

for scheme in grid center; do
    DIR="${B}_${scheme}"
    flag=""; [ "$scheme" = grid ] && flag="--grid-layout"

    run_once "eval_true_utility_balanced_${scheme}" bash -c \
        "python -u plots/layouts/true_utility.py $flag \
             --layout \"\$0/layout_best.pt\" 2>&1 | tee \"\$0/true_utility_eval.txt\"" \
        "$DIR"

    run_once "eval_utility_vs_energy_balanced_${scheme}" \
        python -u plots/layouts/utility_vs_energy.py $flag --run-dir "$DIR"
done

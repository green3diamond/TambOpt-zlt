#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=200g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH -J run19
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Run 19 (branch run19-12km): run 18 at twice the size. 12x12 km plane at 45 deg facing South, 421,200 LHS
# tau decays south of it aimed at LHS points on the plane (2x run 18's showers = half its density), zenith
# 60-100 deg, >= 30 deg from the C8 field (x) axis (scripts/17_make_plane_inputs.py --side 12000, in
# <RUN_LOCATION>/inputs). Same shower-frame kernel as run 18; 100 detectors; the 9-type layout zoo
# (disk and grid-patch sizes scaled to the plane); recon with the relative-time input; no Stage 2.
#
#   sbatch slurm/run19_pipeline.sh gen      Stage 0: showers
#   sbatch slurm/run19_pipeline.sh build    Stage 1: kernel labels, 9 layout types
#   sbatch slurm/run19_pipeline.sh recon    Stage 3: relative-time recon, AdamW weight decay 0.03 (run 18 sweep), then L-BFGS
#   sbatch slurm/run19_pipeline.sh tests    layout tests (plots/analysis/run19/tests/run_tests.sh), after the recon
#
# Status keys in pipeline_status_run19_<mode>.json make requeues resume from the last finished step.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh

MODE=${1:-gen}
STATUS_FILE="pipeline_status_run19_${MODE}.json"
[ -s "$STATUS_FILE" ] || echo '{}' > "$STATUS_FILE"
_status_done() {
    python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS_FILE')).get('$1')=='done' else 1)"
}
_status_mark_done() {
    python -c "import json,os; d=json.load(open('$STATUS_FILE')); d['$1']='done'; json.dump(d, open('$STATUS_FILE.tmp.$$','w'), indent=2); os.replace('$STATUS_FILE.tmp.$$','$STATUS_FILE')"
}
run_once () {
    local key="$1"; shift
    if _status_done "$key"; then echo ">>> Skipping $key (already done)"; return 0; fi
    echo ">>> Running $key"
    "$@" || exit $?
    _status_mark_done "$key"
}

case "$MODE" in
gen)
    run_once 00_generate python -u scripts/00_generate_data_dual_species.py
    ;;
build)
    run_once 01_build python -u scripts/01_build_dataset_northeast.py
    ;;
tests)
    run_once tests bash plots/analysis/run19/tests/run_tests.sh
    ;;
recon)
    run_once 03_recon python -u scripts/03_train_recon_deepsets.py --recon-input kernel --adam-patience 50 \
        --input-mode reltime1 --weight-decay 0.03
    ;;
*)
    echo "unknown mode $MODE"; exit 1 ;;
esac

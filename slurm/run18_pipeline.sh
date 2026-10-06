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
#SBATCH -J run18
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

# Run 18 (branch run18-showerframe): 6x6 km plane at 45 deg facing South (like Malata), 210,600 LHS tau
# decays south of it aimed at LHS points on the plane, zenith 60-100 deg, >= 30 deg from the C8 field
# (x) axis (scripts/17_make_plane_inputs.py, already in <RUN_LOCATION>/inputs). Shower-frame kernel:
# layer depth (l+1) x 500 m, times interpolated between planes, circular showers, widths k = 8
# capped at 50 m. No Stage 2 surrogate: the recon trains on kernel responses only.
#
#   sbatch slurm/run18_pipeline.sh gen      Stage 0: showers
#   sbatch slurm/run18_pipeline.sh build    Stage 1: kernel labels
#   sbatch slurm/run18_pipeline.sh recon    Stage 3: recon on kernel responses (+ plots, two-tier kernel JSON)
#   sbatch slurm/run18_pipeline.sh score    grid/center starts + the 6 kernel-optimized layouts, and the core test
#   sbatch slurm/run18_pipeline.sh build_zoo  Stage 1 for the layout-zoo strategies 5-8, merged into ..._northeast_zoo
#   sbatch slurm/run18_pipeline.sh recon_zoo  Stage 3 on the 9-strategy dataset -> ..._deepsets_zoo
#
# Width caches: kernel_bw_run18.sh (after gen); optimizations: kernel_opt_run18.sh (after recon + widths).
# Status keys in pipeline_status_run18_<mode>.json make requeues resume from the last finished step.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run18_guard.sh

MODE=${1:-gen}
STATUS_FILE="pipeline_status_run18_${MODE}.json"
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
build_zoo)
    # Layout zoo: label only the new strategies 5-8 (clusters, disk, grid patch) into
    # ..._northeast_zoo_new, then append them to the existing 5-strategy build as ..._northeast_zoo.
    run_once 01_build_zoo_new env TAMBOPT_DATASET_TAG=zoo_new \
        python -u scripts/01_build_dataset_northeast.py --strategies 5 6 7 8
    D1="$R/test_v6_run_01_northeast"
    run_once 01b_merge_zoo python -u scripts/01b_merge_strategy_blocks.py \
        --base "$D1" --add "${D1}_zoo_new" --out "${D1}_zoo"
    ;;
recon_zoo)
    # Same recon as `recon`, trained on the 9-strategy dataset; written to ..._deepsets_zoo.
    run_once 03_recon_zoo env TAMBOPT_DATASET_TAG=zoo \
        python -u scripts/03_train_recon_deepsets.py --recon-input kernel --adam-patience 50 --out-suffix zoo
    ;;
recon)
    run_once 03_recon python -u scripts/03_train_recon_deepsets.py --recon-input kernel --adam-patience 50
    # Two-tier utility on kernel responses: fixed thresholds (as run 17's kernel-in-the-loop runs).
    run_once calib python - "$R/two_tier_calibration_kernel_5deg.json" <<'PY'
import json, sys
json.dump(dict(theta_good=5.0, e_good=0.2, s_theta=1.0, s_e=0.05, theta_ref=20.0, e_ref=0.5,
               _note="kernel-input two-tier thresholds, fixed (run 18, as run 17's kernel_opt runs)"),
          open(sys.argv[1], "w"), indent=2)
print("[write]", sys.argv[1])
PY
    ;;
score)
    D="$R/diagnostics/stage4_scores"; mkdir -p "$D"
    python - "$D" <<'PY' || exit 1
import sys, torch; sys.path.insert(0, ".")
import modules.constants as C
from modules.geometry import load_tr_mountain, sample_initial_layout_ne, project_to_mountain_ne
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
for scheme in ("grid", "center"):
    torch.manual_seed(0)
    e, n = sample_initial_layout_ne(m, n_units=C.N_DETECTORS, scheme=scheme)
    e, n = project_to_mountain_ne(m, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    torch.save(dict(x=e.float(), y=n.float()), f"{sys.argv[1]}/start_{scheme}.pt")
PY
    ARGS="--layout start_grid=$D/start_grid.pt --layout start_center=$D/start_center.pt"
    for s in _current _twotier_l0 _twotier_l0p3; do
        for scheme in grid center; do
            K="${R}/test_v6_run_04_optimize_lbfgs_kernel${s}_lr10_${scheme}/layout_best.pt"
            [ -f "$K" ] && ARGS="$ARGS --layout kernel${s}_${scheme}=$K"
        done
    done
    python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 $ARGS --out "$D/scores.json" || exit 1
    CT="$R/diagnostics/core_test"
    python -u "$CT/make_core_layouts.py" || exit 1
    python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 $(sed 's/^/--layout /' "$CT/layouts.txt") \
        --out "$CT/core_test_scores.json" || exit 1
    MPLCONFIGDIR=/tmp/mplcache_$USER python -u "$CT/plot_core_test.py"
    ;;
*)
    echo "unknown mode $MODE"; exit 1 ;;
esac

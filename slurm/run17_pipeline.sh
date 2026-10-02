#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=100g
#SBATCH --time=3-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J run17
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt/.claude/worktrees/run17-plane

# Run 17 (branch run17-plane, own worktree): synthetic 3x3 km plane at 45 deg facing East,
# 105,300 LHS tau decays in a 3x3x3 km cube east of it aimed at LHS points on the plane
# (scripts/17_make_plane_inputs.py, already written to <RUN_LOCATION>/inputs), sigma = 10 m
# adaptive kernel (k = 8), lit/dark Stage 2, all 100k training events.
#
#   sbatch slurm/run17_pipeline.sh prep                 Stage 0-3, conditional plots, calibration
#   sbatch slurm/run17_pipeline.sh opt current          Stage 4, current utility
#   sbatch slurm/run17_pipeline.sh opt twotier 0        Stage 4, two-tier lambda = 0
#   sbatch slurm/run17_pipeline.sh opt twotier 0.3      Stage 4, two-tier lambda = 0.3
#   sbatch slurm/run17_pipeline.sh score                score every Stage-4 layout at 5 and 2.5 deg
#
# Stage 4: Adam only, 10,000 epochs, grid and center starts, 3 chains each. Status keys in
# pipeline_status_run17_<mode>.json make requeues resume from the last finished step.

source slurm/env.sh
source slurm/gpu_gate.sh
gpu_health_gate

MODE=${1:-prep}
STATUS_FILE="pipeline_status_run17_${MODE}${2:+_$2}${3:+_$3}.json"   # one per job: parallel Stage-4 jobs never share a file
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

python - <<'EOF' || exit 1
import sys; sys.path.insert(0, ".")
import modules.constants as C
print(f"RUN_LOCATION={C.RUN_LOCATION}\ngeometry={C.GEOMETRY_PATH_RESOLVED} [{C.GEOMETRY_GROUP}]\n"
      f"primaries={C.TAU_WHOLESKY_PATH}\nsigma={C.SIGMA_SPATIAL} interp_k={C.KERNEL_INTERP_K} "
      f"hurdle={C.SURROGATE_HURDLE} dataset_fraction={C.DATASET_FRACTION}")
ok = (C.RUN_LOCATION.rstrip("/").endswith("17_plane45_interp_hurdle_100k") and C.GEOMETRY_GROUP == "plane45"
      and C.KERNEL_INTERP_K == 8 and C.SIGMA_SPATIAL == 10.0 and C.SURROGATE_HURDLE
      and C.DATASET_FRACTION >= 1.0)
sys.exit(0 if ok else "constants do not match run 17")
EOF
R=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION)')
CALIB="$R/two_tier_calibration_5deg.json"

case "$MODE" in
prep)
    run_once 00_generate python -u scripts/00_generate_data_dual_species.py
    run_once 01_build    python -u scripts/01_build_dataset_northeast.py
    run_once 02_fnn      python -u scripts/02_train_fnn_deepsets.py
    run_once 03_recon    python -u scripts/03_train_recon_deepsets.py --recon-input mix --mix-frac 0.3
    run_once plots_2_3   python -u plots/training/02_nn_target_vs_pred.py --dual
    run_once calibrate   python -u scripts/17_calibrate_two_tier.py --theta-good 5 --e-good 0.2
    ;;
opt)
    UTIL=${2:-current}; LAM=${3:-0.3}
    if [ "$UTIL" = current ]; then
        SUFFIX="_current"
    else
        export TAMBOPT_UTILITY=two_tier TAMBOPT_TWO_TIER_LAMBDA="$LAM" TAMBOPT_TWO_TIER_CALIB="$CALIB"
        SUFFIX="_twotier_l${LAM/./p}"
    fi
    run_once "04${SUFFIX}" python -u scripts/04_optimize_lbfgs_ensemble.py --chains 3 \
        --adam-epochs 10000 --lbfgs-iters 0 --opt_suffix "$SUFFIX"
    ;;
score)
    D="$R/diagnostics/stage4_scores"; mkdir -p "$D"
    python - "$D" <<'EOF' || exit 1
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
EOF
    ARGS="--layout start_grid=$D/start_grid.pt --layout start_center=$D/start_center.pt"
    for s in _current _twotier_l0 _twotier_l0p1 _twotier_l0p3; do
        for scheme in grid center; do
            P="${R}/test_v6_run_04_optimize_lbfgs_ensemble${s}_${scheme}/layout_best.pt"
            [ -f "$P" ] && ARGS="$ARGS --layout ${s#_}_${scheme}=$P"
        done
    done
    python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 $ARGS --out "$D/scores.json"
    ;;
esac

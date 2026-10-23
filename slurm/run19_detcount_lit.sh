#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --mem=64g
#SBATCH --time=0-01:30:00
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J run19_detcount_lit
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 19 full-plane grid vs detector count with lit = E > 0.001 (the kNN gate's hit) -> tests/detcount_lit0p001/,
# then the summary deck (slide 12 uses that figure when it exists).
source slurm/env.sh
source slurm/notify.sh
source slurm/gpu_gate.sh
gpu_health_gate
source slurm/run19_guard.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
T=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION + "/diagnostics/tests")')
python -u scripts/diagnostics/detector_count_grid.py --n-events 5000 --lit-min 0.001 --out "$T/detcount_lit0p001" &&
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/figures/make_summary_2wk_pdf.py
rc=$?; notify_end $rc; exit $rc

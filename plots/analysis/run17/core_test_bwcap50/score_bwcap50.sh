#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH -J r17_bwcap50
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test_bwcap50/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 17 core-test layouts rescored with the kernel widths capped at 50 m (branch run17-bwcap50),
# same events and kernel-trained recon as ../core_test, then the utility comparison figure.
source slurm/env.sh
[ "$(git branch --show-current)" = run17-bwcap50 ] || { echo "main checkout not on run17-bwcap50"; exit 1; }
export TAMBOPT_KERNEL_BW_MAX=50
python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 \
    --recon-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/test_v6_run_03_recentered_deepsets_kernel100 \
    $(sed 's/^/--layout /' /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/layouts.txt) \
    --out /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test_bwcap50/core_test_scores_cap50.json || exit 1
MPLCONFIGDIR=/tmp/mplcache_$USER python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test_bwcap50/plot_bwcap50.py

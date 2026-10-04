#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH -J r17_core_test
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 17 core test: (re)build the layout list, score every layout in layouts.txt on 1,536 held-out kernel showers
# with the kernel-trained recon (*_deepsets_kernel100, the one the kernel optimizer uses), then plot.
source slurm/env.sh
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/make_core_layouts.py || exit 1      # picks up any optimized layouts finished since
python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 --recon-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/test_v6_run_03_recentered_deepsets_kernel100 $(sed 's/^/--layout /' /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/layouts.txt) \
    --out /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/core_test_scores.json || exit 1
MPLCONFIGDIR=/tmp/mplcache_$USER python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/core_test/plot_core_test.py

#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_core_test
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/core_test/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 18 core test: (re)build the layout list, score every layout in layouts.txt on 1,536 held-out kernel
# showers with the run's recon (trained on the shower-frame kernel; the default *_deepsets folder), then plot.
source slurm/env.sh
D=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/core_test
python -u $D/make_core_layouts.py || exit 1      # picks up any optimized layouts finished since
python -u scripts/diagnostics/score_layouts.py --theta-good 2.5 $(sed 's/^/--layout /' $D/layouts.txt) \
    --out $D/core_test_scores.json || exit 1
MPLCONFIGDIR=/tmp/mplcache_$USER python -u $D/plot_core_test.py

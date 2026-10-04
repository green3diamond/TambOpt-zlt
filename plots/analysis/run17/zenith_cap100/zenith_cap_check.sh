#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH -J r17_zen100
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/zenith_cap100/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Zenith <= 100 test: axis tilt, grid recon by zenith band (uncapped / 50 m cap), run 18 geometry,
# then the single-shower array-size figures for two showers with zenith <= 100.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/zenith_cap100/zenith_cap_check.py || exit 1
for w in center off; do
    python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/array_size_shower.py --where $w --max-zenith 100 --out-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/zenith_cap100 || exit 1
done

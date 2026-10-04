#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --time=00:40:00
#SBATCH -J r17_array_size
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# One held-out shower near the plane centre and one ~600 m off it vs array size (100-detector disks at the centre + grid), kernel widths uncapped and capped.
source slurm/env.sh
for w in center off; do
    MPLCONFIGDIR=/tmp/mplcache_$USER python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/array_size_shower.py --where $w "$@" || exit 1
done

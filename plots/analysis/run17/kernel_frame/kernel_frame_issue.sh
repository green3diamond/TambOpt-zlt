#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH --time=00:40:00
#SBATCH -J kernel_frame
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/kernel_frame/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Kernel frame issue figure: global-frame kernel (as used) vs the same kernel in the shower frame, 3 showers.
source slurm/env.sh
MPLCONFIGDIR=/tmp/mplcache_$USER python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/kernel_frame/kernel_frame_issue.py "$@"

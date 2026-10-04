#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --time=00:40:00
#SBATCH -J r17_array_tinterp
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Fixed kernel with arrival times interpolated between planes (t_p + (s_d - s_p) / c): array figures + ablation.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
for w in center off; do
    common="--where $w --max-zenith 100 --min-angle-b 30 --centre-on core --kernel fixed --time-interp --out-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel"
    python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/array_size_shower.py $common || exit 1
    python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/array_size_ablation.py $common || exit 1
done

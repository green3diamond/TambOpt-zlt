#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --time=00:40:00
#SBATCH -J r17_array_fixed
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Single-shower array-size test redone: zenith <= 100, >= 30 deg from the C8 field, disks centred on the
# shower core; kernel fixed (shower frame, layer depth (l + 1) x 500 m) and, for comparison, as used.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
for w in center off; do
    for k in fixed used; do
        python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/array_size_shower.py --where $w --max-zenith 100 --min-angle-b 30 --centre-on core \
            --kernel $k --out-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel || exit 1
    done
done

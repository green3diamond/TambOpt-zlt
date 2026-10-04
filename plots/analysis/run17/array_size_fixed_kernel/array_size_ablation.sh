#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH --time=00:30:00
#SBATCH -J r17_array_ablation
#SBATCH --mail-type=FAIL
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Ablation for the redone array-size test: recon with times / energies / all signals zeroed, lit span and time spread.
source slurm/env.sh
for w in center off; do
    for k in fixed used; do
        python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/array_size_ablation.py --where $w --max-zenith 100 --min-angle-b 30 --centre-on core --kernel $k --out-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel || exit 1
    done
done

#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH -J zoo_pdf
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/figures/slurm-%j-zoo_pdf.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Re-plot the old-vs-zoo comparison, then build run18_zoo_recon.pdf.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/plot_compare.py || exit 1
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/figures/make_zoo_recon_pdf.py

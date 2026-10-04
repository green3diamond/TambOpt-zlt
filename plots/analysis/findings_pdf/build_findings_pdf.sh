#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH -J findings_pdf
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/figures/slurm-%j-findings_pdf.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Rebuild findings_after_run13.pdf (queued after the run 17 kernel-in-the-loop core test so its chapter appears).
source slurm/env.sh
cd /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6
MPLCONFIGDIR=/tmp/mplcache_$USER python -u figures/make_findings_pdf.py

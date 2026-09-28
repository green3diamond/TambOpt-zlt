#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH -c 4
#SBATCH -J axis_impact_density
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Where tau shower axes hit the detector mesh (uniform vs concentrated illumination).
source slurm/env.sh
python -u scripts/diagnostics/axis_impact_density.py "$@"

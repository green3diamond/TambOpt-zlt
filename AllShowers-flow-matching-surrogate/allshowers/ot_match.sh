#!/bin/bash
# Submit from the AllShowers-flow-matching-surrogate folder (log paths are relative to it):
#   sbatch allshowers/ot_match.sh
#SBATCH --job-name=ot_full
#SBATCH --mem=500G
#SBATCH --cpus-per-task=38
#SBATCH --time=10:00:00
#SBATCH -p serial_requeue
#SBATCH --output=logs/ot_full_%j.out
#SBATCH --error=logs/ot_full_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

cd "$SLURM_SUBMIT_DIR"
python allshowers/OT_match.py conf/photons.yaml --with-time

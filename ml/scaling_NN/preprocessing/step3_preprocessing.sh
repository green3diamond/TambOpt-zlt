#!/bin/bash
#SBATCH --job-name=step3_preproc
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/n/home05/zdimitrov/tambo/logs/preprocessing/step3_%j.out
#SBATCH --error=/n/home05/zdimitrov/tambo/logs/preprocessing/step3_%j.err

set -euo pipefail

mkdir -p /n/home05/zdimitrov/tambo/logs/preprocessing

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/conda_env/

echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
which python
python -V

# TODO: Update these paths to your actual data locations
# Input: Step 2 preprocessed data (one file per PDG type)
INPUT_DIR="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_2nd_step"
OUTPUT_DIR="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_3rd_step"

# Define input files (adjust PDG codes as needed)
INPUTS=(
  "${INPUT_DIR}/pdg_-11/histograms.pt"    # e+
  "${INPUT_DIR}/pdg_11/histograms.pt"     # e-
  "${INPUT_DIR}/pdg_-211/histograms.pt"   # pi+
  "${INPUT_DIR}/pdg_211/histograms.pt"    # pi-
  "${INPUT_DIR}/pdg_111/histograms.pt"    # pi0
)

# Check if input files exist
echo "Checking input files..."
for input_file in "${INPUTS[@]}"; do
  if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file not found: $input_file"
    exit 1
  fi
  echo "  Found: $input_file"
done

echo ""
echo "Starting step 3 preprocessing..."
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Number of input files: ${#INPUTS[@]}"
echo ""

# Run step 3 preprocessing
python /n/home05/zdimitrov/tambo/TambOpt/ml/diffusion_scaling_NN/preprocessing/step3_preprocessing.py \
  --inputs "${INPUTS[@]}" \
  --outdir "${OUTPUT_DIR}" \
  --chunk-size 500 \
  --seed 24

echo ""
echo "Step 3 preprocessing completed successfully!"
echo "Output saved to: ${OUTPUT_DIR}"

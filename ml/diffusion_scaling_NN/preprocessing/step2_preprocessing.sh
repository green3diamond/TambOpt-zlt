#!/bin/bash
#SBATCH --job-name=tambo_hist_array
#SBATCH --mem=150G
#SBATCH --time=2-00:00
#SBATCH --output=/n/home04/zdimitrov/tambo/logs/step2_bboxes/step2_bboxes_%A_%a.log
#SBATCH -p arguelles_delgado
#SBATCH --array=0-4
#SBATCH --cpus-per-task=24

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

# Base directories
BASE_INPUT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_1st_step/
BASE_OUTPUT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tambo_simulations/pre_processed_2nd_step_min_50/

# Logs directory
mkdir -p /n/home04/zdimitrov/tambo/logs/step2_bboxes/
mkdir -p "${BASE_OUTPUT_DIR}"

# Subdirectories to process
CONFIGS=(
  "pdg_211"
  "pdg_-211"
  "pdg_11"
  "pdg_111"
  "pdg_-11"
)

SUBDIR=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

if [[ -z "$SUBDIR" ]]; then
  echo "ERROR: No SUBDIR for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

INPUT_FILE="${BASE_INPUT_DIR}/${SUBDIR}/valid_files.txt"

# Create output subdirectory
OUTPUT_SUBDIR="${BASE_OUTPUT_DIR}/${SUBDIR}"
mkdir -p "${OUTPUT_SUBDIR}"

OUTPUT_FILE="${OUTPUT_SUBDIR}/histograms.pt"

echo "========================================="
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subdirectory: ${SUBDIR}"
echo "Input file list: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "Memory: 150G, CPUs: 24"
echo "========================================="

# Check input
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: Input file ${INPUT_FILE} does not exist!"
  exit 1
fi

NUM_FILES=$(wc -l < "${INPUT_FILE}")
echo "Number of files to process: ${NUM_FILES}"

# Memory-efficient processing with batching
# - Reduced workers (20 instead of 24) to leave headroom
# - Batch size of 3000 to prevent memory accumulation
# - Each batch gets saved separately then combined at the end

python /n/home05/zdimitrov/tambo/TambOpt/ml/diffusion_scaling_NN/preprocessing/step2_preprocessing.py \
    "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --bins 32 \
    --sigma 0.0 \
    --min-particles 50 \
    --outlier-method iqr \
    --iqr-multiplier 1.5 \
    --workers 20 \
    --batch-size 3000

EXIT_CODE=$?

if [[ ${EXIT_CODE} -ne 0 ]]; then
  echo "========================================="
  echo "ERROR: Python script failed with exit code ${EXIT_CODE}"
  echo "========================================="
  exit ${EXIT_CODE}
fi

if [[ -f "${OUTPUT_FILE}" ]]; then
  echo "========================================="
  echo "SUCCESS: Created ${OUTPUT_FILE}"
  echo "File size: $(du -h ${OUTPUT_FILE} | cut -f1)"
  
  # Verify the file is valid PyTorch format
  python -c "import torch; d=torch.load('${OUTPUT_FILE}'); print(f'Samples: {d[\"histograms\"].shape[0]}')" 2>/dev/null
  
  if [[ $? -eq 0 ]]; then
    echo "File validated successfully"
  else
    echo "WARNING: File may be corrupted"
  fi
  echo "========================================="
else
  echo "========================================="
  echo "ERROR: Failed to create ${OUTPUT_FILE}"
  echo "========================================="
  exit 1
fi

echo "Histogram creation complete for ${SUBDIR}"
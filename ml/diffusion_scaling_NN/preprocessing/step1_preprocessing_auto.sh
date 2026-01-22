#!/bin/bash
#SBATCH --job-name=tambo_preproc_array
#SBATCH --mem=8G
#SBATCH --time=0-10:00
#SBATCH --output=/n/home04/hhanif/tam/logs/step1_preprocessing/step1_preprocessing_%A_%a.log
#SBATCH -p shared,sapphire
#SBATCH --array=0-6000
#SBATCH --cpus-per-task=8

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

# ============================================
# AUTO-GENERATED CONFIGURATION
# ============================================
CHUNK_SIZE=100

BASE_OUTPUT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_1st_step/

CONFIGS=(
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_11_paths.txt:0:pdg_11"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-11_paths.txt:0:pdg_-11"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_211_paths.txt:1:pdg_211"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-211_paths.txt:1:pdg_-211"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_111_paths.txt:2:pdg_111"
)

# ============================================
# PROCESSING LOGIC
# ============================================

# Function to count non-empty, non-comment lines in a file
count_valid_lines() {
    local file=$1
    grep -v '^#' "$file" | grep -v '^[[:space:]]*$' | wc -l
}

# Build task mapping
TASK_MAP=()
for config_idx in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$config_idx]}"
    IFS=':' read -r PATH_FILE CLASS_ID SUBDIR <<< "$CONFIG"
    
    TOTAL_LINES=$(count_valid_lines "$PATH_FILE")
    NUM_CHUNKS=$(( (TOTAL_LINES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    
    for chunk_idx in $(seq 0 $((NUM_CHUNKS - 1))); do
        TASK_MAP+=("${config_idx}:${chunk_idx}")
    done
done

# Get current task mapping
if [[ $SLURM_ARRAY_TASK_ID -ge ${#TASK_MAP[@]} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range"
    exit 1
fi

CURRENT_TASK="${TASK_MAP[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r CONFIG_IDX CHUNK_IDX <<< "$CURRENT_TASK"

CONFIG="${CONFIGS[$CONFIG_IDX]}"
IFS=':' read -r PATH_FILE CLASS_ID SUBDIR <<< "$CONFIG"

BATCH_START=$((CHUNK_IDX * CHUNK_SIZE))
BATCH_END=$(((CHUNK_IDX + 1) * CHUNK_SIZE))

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SUBDIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $SUBDIR"
echo "Chunk: $CHUNK_IDX"
echo "Batch range: [$BATCH_START:$BATCH_END]"
echo "========================================="

python /n/home04/hhanif/tam/step1_preprocessing.py \
    "${PATH_FILE}" \
    --class-id ${CLASS_ID} \
    --out-dir "${OUTPUT_DIR}" \
    --workers 8 \
    --batch-start ${BATCH_START} \
    --batch-end ${BATCH_END}

echo "Preprocessing complete for $SUBDIR chunk $CHUNK_IDX"

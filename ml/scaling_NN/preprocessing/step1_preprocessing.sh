#!/bin/bash
#SBATCH --job-name=tambo_preproc_array      # Job name
#SBATCH --mem=64G                            # Memory
#SBATCH --time=3-00:00                       # Time limit (D-HH:MM)
#SBATCH --output=/n/home04/hhanif/tam/logs/step1_preprocessing/step1_preprocessing_%A_%a.log
#SBATCH -p arguelles_delgado
#SBATCH --array=0-4                          # 5 jobs: indices 0,1,2,3,4
#SBATCH --cpus-per-task=48

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

# Base output directory (single parent directory)
BASE_OUTPUT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_1st_step/

# List of input path files and their corresponding class IDs
# Format: "path_file:class_id:subdirectory_name"
CONFIGS=(
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_11_paths.txt:0:pdg_11"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-11_paths.txt:0:pdg_-11"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_211_paths.txt:1:pdg_211"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-211_paths.txt:1:pdg_-211"
  "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_111_paths.txt:2:pdg_111"
)

# Pick the config corresponding to this array index
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Safety check
if [[ -z "$CONFIG" ]]; then
  echo "No CONFIG for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# Parse config: split by ':'
IFS=':' read -r PATH_FILE CLASS_ID SUBDIR <<< "$CONFIG"

# Output directory for this PDG/class
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SUBDIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Input paths file: ${PATH_FILE}"
echo "Class ID: ${CLASS_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================="

# Run the preprocessing script
python /n/home04/hhanif/tam/step1_preprocessing.py \
    "${PATH_FILE}" \
    --class-id ${CLASS_ID} \
    --out-dir "${OUTPUT_DIR}" \
    --workers 48 

echo "Preprocessing complete for ${SUBDIR}"

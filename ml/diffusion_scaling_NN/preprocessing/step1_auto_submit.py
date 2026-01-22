#!/usr/bin/env python3
"""
Automated SLURM job submission for preprocessing.
Usage: python auto_submit_preprocessing.py --chunk-size 500
"""
# python /n/home04/hhanif/tam/step1_auto_submit.py --chunk-size 2000
import argparse
import os
import subprocess
import tempfile


def count_valid_lines(filepath):
    """Count non-empty, non-comment lines in a file."""
    if not os.path.exists(filepath):
        return 0
    
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                count += 1
    return count


def generate_slurm_script(chunk_size, total_tasks, configs):
    """Generate the SLURM script content."""
    
    configs_str = "\n".join([f'  "{cfg}"' for cfg in configs])
    
    script = f"""#!/bin/bash
#SBATCH --job-name=tambo_preproc_array
#SBATCH --mem=8G
#SBATCH --time=0-10:00
#SBATCH --output=/n/home04/hhanif/tam/logs/step1_preprocessing/step1_preprocessing_%A_%a.log
#SBATCH -p shared,sapphire
#SBATCH --array=0-{total_tasks - 1}
#SBATCH --cpus-per-task=8

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

# ============================================
# AUTO-GENERATED CONFIGURATION
# ============================================
CHUNK_SIZE={chunk_size}

BASE_OUTPUT_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/pre_processed_1st_step/

CONFIGS=(
{configs_str}
)

# ============================================
# PROCESSING LOGIC
# ============================================

# Function to count non-empty, non-comment lines in a file
count_valid_lines() {{
    local file=$1
    grep -v '^#' "$file" | grep -v '^[[:space:]]*$' | wc -l
}}

# Build task mapping
TASK_MAP=()
for config_idx in "${{!CONFIGS[@]}}"; do
    CONFIG="${{CONFIGS[$config_idx]}}"
    IFS=':' read -r PATH_FILE CLASS_ID SUBDIR <<< "$CONFIG"
    
    TOTAL_LINES=$(count_valid_lines "$PATH_FILE")
    NUM_CHUNKS=$(( (TOTAL_LINES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    
    for chunk_idx in $(seq 0 $((NUM_CHUNKS - 1))); do
        TASK_MAP+=("${{config_idx}}:${{chunk_idx}}")
    done
done

# Get current task mapping
if [[ $SLURM_ARRAY_TASK_ID -ge ${{#TASK_MAP[@]}} ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range"
    exit 1
fi

CURRENT_TASK="${{TASK_MAP[$SLURM_ARRAY_TASK_ID]}}"
IFS=':' read -r CONFIG_IDX CHUNK_IDX <<< "$CURRENT_TASK"

CONFIG="${{CONFIGS[$CONFIG_IDX]}}"
IFS=':' read -r PATH_FILE CLASS_ID SUBDIR <<< "$CONFIG"

BATCH_START=$((CHUNK_IDX * CHUNK_SIZE))
BATCH_END=$(((CHUNK_IDX + 1) * CHUNK_SIZE))

OUTPUT_DIR="${{BASE_OUTPUT_DIR}}/${{SUBDIR}}"
mkdir -p "${{OUTPUT_DIR}}"

echo "========================================="
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $SUBDIR"
echo "Chunk: $CHUNK_IDX"
echo "Batch range: [$BATCH_START:$BATCH_END]"
echo "========================================="

python /n/home04/hhanif/tam/step1_preprocessing.py \\
    "${{PATH_FILE}}" \\
    --class-id ${{CLASS_ID}} \\
    --out-dir "${{OUTPUT_DIR}}" \\
    --workers 8 \\
    --batch-start ${{BATCH_START}} \\
    --batch-end ${{BATCH_END}}

echo "Preprocessing complete for $SUBDIR chunk $CHUNK_IDX"
"""
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate and submit preprocessing SLURM job"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of simulations per batch (default: 500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate script but don't submit job"
    )
    parser.add_argument(
        "--output-script",
        default="/n/home04/hhanif/tam/step1_preprocessing_auto.sh",
        help="Output script filename (default: step1_preprocessing_auto.sh)"
    )
    args = parser.parse_args()
    
    # Configuration matching the original script
    configs = [
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_11_paths.txt:0:pdg_11",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-11_paths.txt:0:pdg_-11",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_211_paths.txt:1:pdg_211",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_-211_paths.txt:1:pdg_-211",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/logs/pdg_111_paths.txt:2:pdg_111",
    ]
    
    # Calculate total tasks needed
    print("=" * 60)
    print(f"Calculating job requirements with chunk size: {args.chunk_size}")
    print("=" * 60)
    
    total_tasks = 0
    task_breakdown = []
    
    for config in configs:
        path_file, class_id, subdir = config.split(':')
        
        if not os.path.exists(path_file):
            print(f"WARNING: Path file not found: {path_file}")
            continue
        
        total_lines = count_valid_lines(path_file)
        num_chunks = (total_lines + args.chunk_size - 1) // args.chunk_size
        
        task_breakdown.append({
            'subdir': subdir,
            'path_file': path_file,
            'total_sims': total_lines,
            'num_chunks': num_chunks,
            'task_start': total_tasks,
            'task_end': total_tasks + num_chunks - 1
        })
        
        total_tasks += num_chunks
        
        print(f"\nDataset: {subdir}")
        print(f"  Path file: {path_file}")
        print(f"  Total simulations: {total_lines}")
        print(f"  Number of chunks: {num_chunks}")
        print(f"  Array indices: {total_tasks - num_chunks} to {total_tasks - 1}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL ARRAY JOBS: {total_tasks}")
    print(f"CHUNK SIZE: {args.chunk_size} simulations per job")
    print("=" * 60)
    
    if total_tasks == 0:
        print("\nERROR: No valid path files found!")
        return 1
    
    # Generate SLURM script
    script_content = generate_slurm_script(args.chunk_size, total_tasks, configs)
    
    # Write script to file
    with open(args.output_script, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(args.output_script, 0o755)
    
    print(f"\nGenerated SLURM script: {args.output_script}")
    
    if args.dry_run:
        print("\n[DRY RUN] Script generated but not submitted.")
        print(f"To submit manually: sbatch {args.output_script}")
        return 0
    
    # Submit job
    print("\nSubmitting job to SLURM...")
    try:
        result = subprocess.run(
            ['sbatch', args.output_script],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        # Extract job ID if present
        if "Submitted batch job" in result.stdout:
            job_id = result.stdout.split()[-1]
            print(f"\n✓ Job submitted successfully!")
            print(f"  Job ID: {job_id}")
            print(f"  Total tasks: {total_tasks}")
            print(f"\nMonitor with: squeue -u $USER")
            print(f"Cancel with: scancel {job_id}")
    
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to submit job:")
        print(e.stderr)
        return 1
    except FileNotFoundError:
        print("\nERROR: 'sbatch' command not found. Are you on a SLURM cluster?")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/bin/bash
#SBATCH -J scan_blobs
#SBATCH -p test
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 4:00:00
#SBATCH -o slurm_logs/slurm-%j-%x.out

# Full-corpus scan of total_deposited/E_primary, to site the Step-0 blob cut.
# CPU only: the corpus is already on disk, nothing is generated here.
set -euo pipefail
cd /n/home05/zdimitrov/tambo/TambOpt
source ~/.bashrc
conda activate multiproc_env
python tests/scan_corpus_blobs.py --chunk 256 "$@"

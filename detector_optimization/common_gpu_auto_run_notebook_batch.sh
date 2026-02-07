#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=100g
#SBATCH --time=24:00:00 			# time in HH:MM:SS
#SBATCH -c 4            			# number of cores
#SBATCH --gres=gpu:1        # requested GPU type

module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization.ipynb
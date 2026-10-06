#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH -J r18_angleonly
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/angleonly/slurm-%j.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt
# Run 18: every earlier layout + growing disks, scored with 'good' = angle only (e_good huge) at 5 and 2.5 deg.
source slurm/env.sh
export MPLCONFIGDIR=/tmp/mplcache_$USER
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/angleonly/make_layouts.py || exit 1
for th in 5 2.5; do
    python -u scripts/diagnostics/score_layouts.py --theta-good $th --e-good 1e9 $(sed 's/^/--layout /' /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/angleonly/layouts.txt) \
        --out /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/angleonly/scores_angleonly_${th}deg.json || exit 1
done
python -u /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/18_plane45S_6km_showerframe_200k/diagnostics/angleonly/plot_angleonly.py

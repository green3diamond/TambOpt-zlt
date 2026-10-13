"""Target-vs-prediction plots for run 19's recon at the end of the Adam phase (epoch 230, val 0.13335),
before L-BFGS overwrites recon.pt. Kernel labels, the training script's val split. Writes into this folder."""
import importlib.util as ilu
import os
import sys
sys.path.insert(0, os.getcwd())
spec = ilu.spec_from_file_location("p02", os.path.join(os.getcwd(), "plots/training/02_nn_target_vs_pred.py"))
p02 = ilu.module_from_spec(spec); spec.loader.exec_module(p02)
here = os.path.dirname(os.path.abspath(__file__))
p02.plot_recon_on_kernel(recon_folder=here, output_path=os.path.join(here, "recon_target_vs_pred_kernel.png"))
print("[write]", here)

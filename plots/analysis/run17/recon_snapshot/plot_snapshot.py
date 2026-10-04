"""Recon plots for a snapshot of the in-training run 17 recon.pt (best-so-far epoch), loaded from
this folder and written into the Stage 3 folder (the pipeline's plot step overwrites them later)."""
import os, sys
sys.path.insert(0, os.getcwd())
import importlib.util as ilu
spec = ilu.spec_from_file_location("p02", os.path.join(os.getcwd(), "plots/training/02_nn_target_vs_pred.py"))
p02 = ilu.module_from_spec(spec); spec.loader.exec_module(p02)
here = os.path.dirname(os.path.abspath(__file__))
out = p02.RECON_DEEPSETS_FOLDER
p02.plot_recon_dual(recon_folder=here, output_path=os.path.join(out, "recon_target_vs_pred.png"))
p02.plot_recon_on_kernel(recon_folder=here, output_path=os.path.join(out, "recon_target_vs_pred_kernel.png"))

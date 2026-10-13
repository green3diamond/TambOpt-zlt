"""As plot_snapshot.py, restricted to val rows where >= 6 detectors are hit (kernel E > 1 particle).
Writes recon_target_vs_pred_kernel_lit6.png / recon_conditional_kernel_lit6.png and prints the lit-count shares."""
import importlib.util as ilu
import os
import sys
import torch
sys.path.insert(0, os.getcwd())
spec = ilu.spec_from_file_location("p02", os.path.join(os.getcwd(), "plots/training/02_nn_target_vs_pred.py"))
p02 = ilu.module_from_spec(spec); spec.loader.exec_module(p02)
here = os.path.dirname(os.path.abspath(__file__))
MIN_LIT = int(sys.argv[1]) if len(sys.argv) > 1 else 6

primary, xy, _E, _T, strat = p02._load_corpus()
species = torch.load(os.path.join(p02.TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
val_idx = p02.shower_level_val_idx(strat, p02.VAL_FRAC, p02.RECON_VAL_SEED, species_ids=species)
s3s = ilu.spec_from_file_location("_s3", os.path.join(os.getcwd(), "scripts", "03_train_recon_deepsets.py"))
s3 = ilu.module_from_spec(s3s); s3s.loader.exec_module(s3)
E_kern, _ = s3.load_kernel_responses(p02.TRAINING_DATASET_FOLDER)
n_lit = (torch.expm1(E_kern[val_idx].float()) > 1).sum(1)
for k in (0, 1, 3, 6, 10, 20):
    print(f"val rows with >= {k:2d} hit detectors: {(n_lit >= k).float().mean():.3f}")
print(f"val rows with 0 hit: {(n_lit == 0).float().mean():.3f}")
keep = val_idx[n_lit >= MIN_LIT]
print(f"plotting {keep.numel():,} of {val_idx.numel():,} val rows (>= {MIN_LIT} hit)")
del E_kern
recon = p02.load_recon(here)
target, pred, N = p02._recon_target_pred_kernel(recon, primary, xy, keep)
out = os.path.join(here, f"recon_target_vs_pred_kernel_lit{MIN_LIT}.png")
p02._draw_recon_scatter(target, pred, N, out, formats=("png",), title_suffix=f" [kernel input, >= {MIN_LIT} hit detectors]")
p02._draw_recon_conditional(target, pred, N, out.replace("target_vs_pred", "conditional"), formats=("png",),
                            title_suffix=f" [kernel input, >= {MIN_LIT} hit detectors]")
print("[write]", out)

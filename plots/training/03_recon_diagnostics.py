"""Re-render Stage-3 diagnostics for a recon run that did not finish normally.

03_train_recon_deepsets.py writes recon_train_log.json, recon_train_curves.png,
recon_target_vs_pred.png and recon_conditional.png only at the very END of its
L-BFGS phase. A run stopped early -- preempted, or killed once L-BFGS has
visibly plateaued (job 45345560 was stopped at closure 321 after iters 200-321
bought 0.00003 val in ~3h) -- leaves the best recon.pt on disk but none of the
figures.

This rebuilds them from what survives:
  * the Adam half of recon_train_log.json, written when Adam finished;
  * the L-BFGS closure trace, recovered from the slurm log with --lbfgs-log
    (see the scraper note below) since it lived only in memory;
  * recon.pt itself, for the scatter/conditional panels.

The scraped L-BFGS trace carries only (iter, loss, val) -- the per-axis mse_*
keys 03's own _plot_curves expects are not printed per closure -- so the curve
figure here is drawn locally rather than by importing that helper.

Run:
    python plots/training/03_recon_diagnostics.py \
        --lbfgs-log <folder>/recon_lbfgs_iter_log_scraped.json
    python plots/training/03_recon_diagnostics.py --skip-scatter   # curves only
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)) + "/TambOpt")

from _pathfix import V6_ROOT  # noqa: F401,E402
import modules  # noqa: F401,E402
from modules.constants import RECON_FOLDER  # noqa: E402

OUTPUT_FOLDER = RECON_FOLDER + "_deepsets"


def plot_curves(adam_log, lbfgs_log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 3 if lbfgs_log else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    ep = [e["epoch"] for e in adam_log]
    axes[0].plot(ep, [e["train"] for e in adam_log], color="C0", label="train")
    axes[0].plot(ep, [e["val"] for e in adam_log], color="C1", label="val")
    axes[0].set_xlabel("Adam epoch"); axes[0].set_ylabel("loss")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].set_title("Adam")

    for name, ls in [("dx", "-"), ("dy", "--"), ("dz", ":"), ("logE", "-.")]:
        axes[1].plot(ep, [e[f"train_{name}"] for e in adam_log],
                     color="C0", linestyle=ls)
        axes[1].plot(ep, [e[f"val_{name}"] for e in adam_log],
                     color="C1", linestyle=ls, label=name)
    axes[1].set_xlabel("Adam epoch"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=8); axes[1].set_title("Adam, per axis")

    if lbfgs_log:
        it = [e["iter"] for e in lbfgs_log]
        axes[2].plot(it, [e["loss"] for e in lbfgs_log], color="C0", label="train")
        axes[2].plot(it, [e["val"] for e in lbfgs_log], color="C1", label="val")
        best = min(lbfgs_log, key=lambda e: e["val"])
        axes[2].axvline(best["iter"], color="k", lw=0.8, ls=":",
                        label=f"best val {best['val']:.6f} @ {best['iter']}")
        axes[2].set_xlabel("L-BFGS closure call")
        axes[2].legend(fontsize=8)
        axes[2].set_title("L-BFGS (train falls, val flat = overfitting)")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[curves] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=OUTPUT_FOLDER)
    ap.add_argument("--lbfgs-log", default=None,
                    help="scraped L-BFGS closure trace (iter/loss/val JSON list)")
    ap.add_argument("--skip-scatter", action="store_true",
                    help="curves only; skips the GPU/corpus-loading panels")
    args = ap.parse_args()

    log_path = os.path.join(args.folder, "recon_train_log.json")
    with open(log_path) as f:
        blob = json.load(f)
    adam_log = [e for e in blob["log"] if e.get("phase") != "lbfgs"]
    print(f"[log] {len(adam_log)} Adam epochs, adam best val="
          f"{blob.get('best_val_total')}")

    lbfgs_log = []
    if args.lbfgs_log and os.path.exists(args.lbfgs_log):
        with open(args.lbfgs_log) as f:
            lbfgs_log = json.load(f)
        best = min(lbfgs_log, key=lambda e: e["val"])
        print(f"[log] {len(lbfgs_log)} L-BFGS closures, best val="
              f"{best['val']:.6f} at iter {best['iter']}")
        # Fold the recovered trace back into the on-disk log so downstream
        # readers see a complete record rather than an Adam-only one.
        blob["lbfgs_iter_log"] = lbfgs_log
        blob["best_val_total"] = min(blob.get("best_val_total", 1e9), best["val"])
        blob["lbfgs_recovered_from_slurm_log"] = True
        with open(log_path, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[log] merged L-BFGS trace into {log_path}")

    plot_curves(adam_log, lbfgs_log,
                os.path.join(args.folder, "recon_train_curves.png"))

    if args.skip_scatter:
        print("[scatter] skipped (--skip-scatter)")
        return

    # plot_recon_only renders BOTH recon_target_vs_pred.png and
    # recon_conditional.png. It can self-load recon.pt, the corpus and the val
    # split -- its RECON_VAL_SEED/VAL_FRAC (1 / 0.10) match 03's SEED/VAL_FRAC,
    # so the split is the same one recon actually validated on. The one thing
    # it CANNOT self-load is the surrogate: its load_fnn() wants a legacy
    # single-model fnn.pt, while this pipeline writes per-species
    # fnn_electron/muon/photon.pt. Hand it the dual wrapper explicitly.
    import torch
    import importlib.util
    from modules.surrogates import load_dual_surrogate
    from modules.constants import FNN_FOLDER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = importlib.util.spec_from_file_location(
        "_plot_tvp", os.path.join(V6_ROOT, "plots", "training",
                                  "02_nn_target_vs_pred.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.plot_recon_only(
        fnn=load_dual_surrogate(FNN_FOLDER, device),
        recon_folder=args.folder,
        output_path=os.path.join(args.folder, "recon_target_vs_pred.png"))
    print("[scatter] done")


if __name__ == "__main__":
    main()

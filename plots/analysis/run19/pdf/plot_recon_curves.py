"""Run 19 recon training: Adam val/train per epoch (stopped by hand at 252, best 230) and the L-BFGS phase."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(P)) + "/test_v6_run_03_recentered_deepsets"
A = {r["epoch"]: r for r in json.load(open(f"{R}/recon_train_log_adam.json"))["log"]}
L = json.load(open(f"{R}/recon_train_log.json"))["lbfgs_iter_log"]
ep = sorted(A)
fig, axs = plt.subplots(1, 3, figsize=(20, 5.6))
axs[0].plot(ep, [A[e]["train"] for e in ep], color="#9b9a95", lw=1.5, label="train")
axs[0].plot(ep, [A[e]["val"] for e in ep], color="#2a78d6", lw=2, label="val")
axs[0].set(ylim=(0.125, 0.145), xlabel="epoch", ylabel="loss", title="Adam (AdamW wd 0.03, lr 3e-5)")
b = min(ep, key=lambda e: A[e]["val"]); axs[0].plot([b], [A[b]["val"]], "o", color="#2a78d6")
axs[0].annotate(f"best {A[b]['val']:.4f} @ {b}", (b, A[b]["val"]), textcoords="offset points", xytext=(-90, -18), fontsize=10)
dr = lambda r: r["val_dx"] + r["val_dy"] + r["val_dz"]
axs[1].plot(ep, [dr(A[e]) for e in ep], color="#2a78d6", lw=2, label="direction (dx + dy + dz)")
axs[1].plot(ep, [A[e]["val_logE"] for e in ep], color="#eb6834", lw=2, label="energy (logE)")
axs[1].set(ylim=(0.045, 0.09), xlabel="epoch", ylabel="val loss part", title="Val loss by part (Adam)")
it = [r["iter"] for r in L]; v = [r["val"] for r in L]
axs[2].plot(it, v, color="#1baf7a", lw=1.5)
bi = min(range(len(v)), key=v.__getitem__)
axs[2].plot([it[bi]], [v[bi]], "o", color="#1baf7a"); axs[2].axhline(v[0], color="#52514e", lw=0.8, ls=":")
axs[2].annotate(f"best {v[bi]:.4f} @ {it[bi]} (start {v[0]:.4f})", (it[bi], v[bi]), textcoords="offset points", xytext=(-150, -18), fontsize=10)
axs[2].set(ylim=(0.130, 0.136), xlabel="L-BFGS closure call", ylabel="val loss", title="L-BFGS (full batch, from the Adam best)")
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9) if ax.get_legend_handles_labels()[0] else None
fig.suptitle("Run 19 recon: DeepSets, reltime1 input, 11.8M training rows · Adam stopped at epoch 252 (val flat since ~165), then L-BFGS", fontsize=13)
fig.tight_layout(); fig.savefig(f"{P}/recon_training.png", dpi=110, facecolor="white", bbox_inches="tight")
print("[write]", f"{P}/recon_training.png")

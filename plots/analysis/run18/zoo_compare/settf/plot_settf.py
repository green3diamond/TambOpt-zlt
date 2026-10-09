"""Attention recon (SetTransformerRecon, reltime1) vs the DeepSets recon with the same setup (relative-time,
AdamW wd 0.03, lr 3e-5, batch 256, run 18 zoo data, Adam only): val loss, its direction part (dx + dy + dz)
and the train-val gap per epoch. Writes settf_vs_deepsets.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
R18 = os.path.dirname(os.path.dirname(os.path.dirname(S)))
RUNS = (("attention (set transformer, reltime1)", "test_v6_run_03_recentered_deepsets_zoo_settf_rt1/recon_train_log.json", "#c23b6b"),
        ("DeepSets (reltime), same setup", "test_v6_run_03_recentered_deepsets_zoo_reltime_wd0.03/recon_train_log_adam.json", "#2a78d6"))
fig, axs = plt.subplots(1, 3, figsize=(19, 5.6))
for lab, f, col in RUNS:
    log = {}
    for r in json.load(open(os.path.join(R18, f)))["log"]:
        log[r["epoch"]] = r                                                  # last entry per epoch (requeues)
    ep = sorted(log)
    val = [log[e]["val"] for e in ep]
    dirn = [log[e]["val_dx"] + log[e]["val_dy"] + log[e]["val_dz"] for e in ep]
    raw = [log[e]["val"] - log[e]["train"] for e in ep]
    # 5-epoch running median: a requeued epoch logs the train loss of a partial pass (spikes in the raw gap).
    gap = [sorted(raw[max(0, i - 2):i + 3])[len(raw[max(0, i - 2):i + 3]) // 2] for i in range(len(raw))]
    b = min(ep, key=lambda e: log[e]["val"])
    axs[0].plot(ep, val, color=col, lw=2, label=f"{lab}: best {log[b]['val']:.4f} @ {b}")
    axs[0].plot([b], [log[b]["val"]], "o", color=col, ms=7)
    bd = min(ep, key=lambda e: dirn[ep.index(e)])
    axs[1].plot(ep, dirn, color=col, lw=2, label=f"{lab}: best {min(dirn):.4f} @ {bd}")
    axs[2].plot(ep, gap, color=col, lw=2, label=lab)
for ax, t, yl in zip(axs, ("Val loss (total)", "Val loss, direction part (dx + dy + dz)", "Overfitting: val − train (5-epoch median)"),
                     ("val loss", "val loss (direction)", "val − train")):
    ax.set(title=t, xlabel="epoch", ylabel=yl); ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
axs[0].set_ylim(0.076, 0.09); axs[1].set_ylim(0.030, 0.036); axs[2].axhline(0, color="#52514e", lw=0.8)
axs[2].set_ylim(-0.005, 0.025)
fig.suptitle("Attention recon vs DeepSets · run 18 zoo data, AdamW wd 0.03, Adam only (≤ 80 epochs, patience 30)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{S}/settf_vs_deepsets.png", dpi=120, facecolor="white", bbox_inches="tight")
print("[write]", f"{S}/settf_vs_deepsets.png")

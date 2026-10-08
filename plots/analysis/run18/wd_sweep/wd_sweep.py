"""Weight-decay sweep for the relative-time recon (run 18 zoo data): val/train loss per epoch from the job logs,
for wd = 0 (the relative-time recon's Adam phase) and wd = 0.01 / 0.03 / 0.1 / 0.3 (AdamW, 80 epochs, patience 30,
no L-BFGS). A requeued job repeats epochs in its (appended) log; the last entry per epoch is kept.
Writes wd_sweep.png and wd_sweep.json."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
L = "/n/home05/zdimitrov/tambo/TambOpt/slurm_logs/"
RUNS = {"0 (no decay)": L + "slurm-51053289-r18_recon_reltime.out", "0.01": L + "slurm-51091359_0-wd_sweep.out",
        "0.03": L + "slurm-51091359_1-wd_sweep.out", "0.1": L + "slurm-51091359_2-wd_sweep.out",
        "0.3": L + "slurm-51091359_3-wd_sweep.out"}
COL = {"0 (no decay)": "#52514e", "0.01": "#1baf7a", "0.03": "#2a78d6", "0.1": "#8a5cd1", "0.3": "#eb6834"}
PAT = re.compile(r"^\[epoch\s+(\d+)/\d+\] train=([\d.]+) val=([\d.]+) \(dx=([\d.]+) dy=([\d.]+) dz=([\d.]+) logE=([\d.]+)\)\s+([\d.]+)s")

res = {}
for wd, f in RUNS.items():
    ep = {}
    for line in open(f):
        m = PAT.match(line)
        if m:
            e = int(m.group(1))
            if wd.startswith("0 (") and e > 80:          # compare over the sweep's 80 epochs
                continue
            ep[e] = [float(x) for x in m.groups()[1:]]
    es = sorted(ep)
    best = min(es, key=lambda e: ep[e][1])
    first = {t: next((e for e in es if ep[e][1] <= t), None) for t in (0.0790, 0.0785, 0.0780)}
    b = ep[best]
    res[wd] = dict(epochs=es, train=[ep[e][0] for e in es], val=[ep[e][1] for e in es], best_epoch=best,
                   best_val=b[1], train_at_best=b[0], direction=b[2] + b[3] + b[4], logE=b[5],
                   gap=b[1] - b[0], first=first, s_per_epoch=sum(ep[e][6] for e in es) / len(es),
                   last_val=ep[es[-1]][1])
    print(f"wd {wd:13s} best {b[1]:.4f} @ {best:2d}  dir {b[2] + b[3] + b[4]:.4f}  logE {b[5]:.4f}  gap {b[1] - b[0]:+.4f}  "
          f"first<=.0790/.0785/.0780: {first[0.079]}/{first[0.0785]}/{first[0.078]}  last {ep[es[-1]][1]:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(18, 6.2))
for wd, r in res.items():
    axs[0].plot(r["epochs"], r["val"], "-", color=COL[wd], lw=2, label=f"wd {wd}")
    axs[0].plot(r["best_epoch"], r["best_val"], "o", color=COL[wd], ms=7)
    g = [v - t for v, t in zip(r["val"], r["train"])]
    g = [sorted(g[max(0, i - 2):i + 3])[len(g[max(0, i - 2):i + 3]) // 2] for i in range(len(g))]   # 5-epoch median
    axs[1].plot(r["epochs"], g, "-", color=COL[wd], lw=2, label=f"wd {wd}")
axs[0].set(ylim=(0.076, 0.090), xlabel="epoch", ylabel="val loss", title="Validation loss (dot = best epoch)")
axs[1].set(ylim=(-0.004, 0.014), xlabel="epoch", ylabel="val − train", title="Overfitting gap (val − train, 5-epoch median: requeues spike the train loss)")
axs[1].axhline(0, color="#52514e", lw=0.8)
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False)
fig.suptitle("Weight-decay sweep · relative-time recon on run 18 zoo data · AdamW, ≤ 80 epochs, no L-BFGS · "
             "wd 0 = the relative-time recon's own Adam phase", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/wd_sweep.png", dpi=120, facecolor="white", bbox_inches="tight")
json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("epochs", "train", "val")} for k, v in res.items()},
          open(f"{OUT}/wd_sweep.json", "w"), indent=1)
print("[write] wd_sweep.png wd_sweep.json")

"""Train a DeepSets reconstruction network on frozen dual-surrogate outputs.

Drop-in replacement for 03_train_recon.py using DeepSetsRecon instead of the
flat MLP. Permutation invariance is structural — no augmentation needed.

Architecture:
    token_i = [x_i, y_i, E_comb_i, T_comb_i]     (4 features, per detector)
    h_i     = encoder(token_i)                     shared encoder
    c       = context_proj(cat[mean h_i, max h_i])  invariant pool (maxmean)
    out     = decoder(c)            → [dir_x, dir_y, dir_z, log_e_norm]

E_comb/T_comb come from one of three sources, chosen with --recon-input:

  sample  (default, historical behaviour) — a stochastic SAMPLE drawn from the
          stage-2 heteroscedastic surrogate's predicted (mean, var) distribution
          (DualSpeciesSurrogate.forward_sample), not the mean point estimate, so
          recon sees the surrogate's learned aleatoric spread directly.
  kernel  — the REAL kernel-computed detector responses Step 1 already wrote
          (E.pt / T.pt), i.e. the same labels Step 2 trained the surrogate to
          predict.
  mix     — a per-row coin flip between the two (--mix-frac = P(kernel)).

Why `kernel`/`mix` exist: with `sample`, recon only ever sees surrogate-generated
responses, but plots/layouts/true_utility.py scores it on KERNEL responses ("kernel
vs surrogate on the SAME recon"). That is a train/eval distribution mismatch, and
it shows up as a large standing artifact gap — on the 6-chain run (job 42695512)
surrogate-U 47.94 vs true-U 22.78 on the UN-optimized grid baseline, a ~2.2-2.4x
uniform overestimate across u_theta/u_phi/u_e that is present before the optimizer
moves anything. Training recon on kernel responses removes that mismatch; `mix`
keeps it robust to both, since stage 4 still feeds it surrogate output.

Normalization: in_mean/in_std are (4,) — one scalar per feature kind,
broadcast over all detector slots. Output stats are (4,) like the flat MLP.

Writes to RECON_FOLDER + "_deepsets" — safe to run alongside 03_train_recon.py.

Run:
    python 03_train_recon_deepsets.py
    python 03_train_recon_deepsets.py --recon-input kernel
    python 03_train_recon_deepsets.py --recon-input mix --mix-frac 0.5
    python 03_train_recon_deepsets.py --epochs 2 --lbfgs-iters 3   # smoke run
"""
import argparse
import json
import os
import sys
import time
from typing import Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

import modules   # noqa: F401
from modules.surrogates  import load_dual_surrogate
from modules.surrogates  import DeepSetsRecon
from modules.constants import (
    RECON_FOLDER, TRAINING_DATASET_FOLDER, FNN_FOLDER,
    N_DETECTORS, T_LOG_SCALE,
)

OUTPUT_FOLDER = RECON_FOLDER + "_deepsets"

# Where the recon's per-detector (E, T) inputs come from — see the module
# docstring. "sample" is the historical default; the CLI can override.
RECON_INPUT   = "sample"     # "sample" | "kernel" | "mix"
MIX_FRAC      = 0.5          # P(row uses kernel labels) when RECON_INPUT == "mix"

# ── Architecture ─────────────────────────────────────────────────────────────
# 3 -> 6 layers each (2026-07-30). Cost is asymmetric: the encoder runs
# PER DETECTOR (x100) while the decoder runs once per event after the pool, so
# deepening the encoder is ~100x more expensive than deepening the decoder.
# n_enc 3->6 raises encoder FLOPs ~2.5x (5 hidden matmuls vs 2); n_dec 3->6 is
# nearly free. Params 298k -> 693k.
HIDDEN   = 256
CONTEXT  = 128
N_ENC    = 4
N_DEC    = 4
POOL     = "maxmean"   # "mean" or "maxmean"; maxmean doubles context_proj input

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
N_EPOCHS   = 300
LR         = 3e-5
GRAD_CLIP  = 10.0
VAL_FRAC   = 0.10
SEED       = 1
NUM_WORKERS = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── L-BFGS ───────────────────────────────────────────────────────────────────
LBFGS_LR           = 1.0
LBFGS_MAX_ITER     = 500
LBFGS_HISTORY_SIZE = 20
# Encoder activations dominate: chunk × n_det × hidden × bytes/layer.
# chunk=4096 keeps peak memory ~3 GB on a 40 GB GPU (vs 32 768 for flat MLP).
LBFGS_CHUNK        = 4_096
RESUME_CKPT_INTERVAL = 25   # save a rolling resume checkpoint every N Adam epochs

RECON_INPUT_FEATURES = 4   # (x, y, E, T) per detector — E, T come from
                            # RECON_INPUT (surrogate draw, real kernel, or a
                            # mix); see the module docstring.


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float,
                       seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split so all strategies for one shower share a side."""
    n_pairs   = int(strategy_ids.shape[0])
    n_strat   = int(strategy_ids.max().item() + 1)
    n_showers = n_pairs // n_strat

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(val_frac * n_showers)))

    is_val = torch.zeros(n_showers, dtype=torch.bool)
    is_val[perm[:n_val]] = True

    all_idx = torch.arange(n_pairs, dtype=torch.long)
    shower_of_pair = all_idx - strategy_ids * n_showers
    val_mask = is_val[shower_of_pair]
    return (torch.nonzero(~val_mask).squeeze(-1),
            torch.nonzero(val_mask).squeeze(-1))


@torch.no_grad()
def compute_fnn_predictions(model: nn.Module,
                             primary: torch.Tensor,
                             xy:      torch.Tensor,
                             device:  torch.device,
                             batch_size: int = 1024
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw one stochastic sample per row from the dual surrogate's predicted
    distribution (not the mean) over the whole corpus. Returns CPU tensors.
    Sampled once here and reused for every epoch — the same "one fixed
    realization per (primary, layout) row" convention the real kernel-labeled
    corpus already uses, not a fresh draw every epoch."""
    model.eval()
    N = int(primary.shape[0])
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        pred = model.forward_sample(
            primary[lo:hi].to(device, non_blocking=True),
            xy[lo:hi].to(device, non_blocking=True))            # (B, 100, 2)
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


def load_kernel_responses(folder: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The REAL kernel labels Step 1 wrote, in the surrogate's own output space.

    Step 1 saves E already log1p'd (`01_build_dataset_northeast.py:117`) but T
    RAW, and Step 2 is what applies `log1p(T * T_LOG_SCALE)` before training
    against it. `DualSpeciesSurrogate.forward` returns
    (log1p(N_tot), log1p(t_tot * T_LOG_SCALE)) — so E needs no transform here and
    T needs Step 2's. Getting that asymmetry wrong would silently feed recon two
    different spaces depending on --recon-input.

    Row-aligned with primary.pt / xy.pt, same (N, N_DETECTORS) shape as
    compute_fnn_predictions' return, so the two are drop-in interchangeable.
    """
    E = torch.load(os.path.join(folder, "E.pt")).float()
    T = torch.load(os.path.join(folder, "T.pt")).float()
    T = torch.log1p(T * T_LOG_SCALE)
    return E, T


def build_recon_inputs(mode: str, mix_frac: float, seed: int,
                       E_sample: torch.Tensor, T_sample: torch.Tensor,
                       E_kernel: torch.Tensor, T_kernel: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve --recon-input into the (E, T) pair recon actually trains on.

    "mix" is a per-ROW coin flip rather than a concatenation of both corpora: it
    keeps the row count (and so the shower-level split, the strategy_ids
    alignment and peak memory) exactly as the other two modes, and every event
    still appears once. Seeded so a preempted run resumes on the same draw.
    """
    if mode == "sample":
        return E_sample, T_sample
    if mode == "kernel":
        return E_kernel, T_kernel
    if mode != "mix":
        raise ValueError(f"unknown --recon-input {mode!r}")
    g = torch.Generator().manual_seed(int(seed))
    use_kernel = torch.rand(E_sample.shape[0], generator=g) < float(mix_frac)
    m = use_kernel.unsqueeze(1)                       # (N, 1) broadcast over slots
    print(f"[mix] {int(use_kernel.sum())}/{use_kernel.numel()} rows "
          f"({use_kernel.float().mean():.1%}) use kernel labels")
    return torch.where(m, E_kernel, E_sample), torch.where(m, T_kernel, T_sample)


_AXIS_KEYS = ("tot", "dx", "dy", "dz", "logE")


def _per_axis_loss(pred: torch.Tensor, tgt: torch.Tensor,
                   reduction: str = "mean"):
    l_dx = F.mse_loss(pred[:, 0], tgt[:, 0], reduction=reduction)
    l_dy = F.mse_loss(pred[:, 1], tgt[:, 1], reduction=reduction)
    l_dz = F.mse_loss(pred[:, 2], tgt[:, 2], reduction=reduction)
    l_lE = F.mse_loss(pred[:, 3], tgt[:, 3], reduction=reduction)
    return l_dx + l_dy + l_dz + l_lE, l_dx, l_dy, l_dz, l_lE


@torch.no_grad()
def _validate(recon: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    sums = [0.0] * 5
    n = 0
    for xy_b, E_b, T_b, tgt_b in loader:
        xy_b  = xy_b .to(device, non_blocking=True)
        E_b   = E_b  .to(device, non_blocking=True)
        T_b   = T_b  .to(device, non_blocking=True)
        tgt_b = tgt_b.to(device, non_blocking=True)
        inp   = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)
        losses = _per_axis_loss(recon(inp), tgt_b)
        B = xy_b.shape[0]
        for i, v in enumerate(losses):
            sums[i] += v.item() * B
        n += B
    n = max(n, 1)
    return {k: sums[i] / n for i, k in enumerate(_AXIS_KEYS)}


def _save_ckpt(path: str,
               recon: DeepSetsRecon,
               epoch: int,
               val: dict,
               in_mean: torch.Tensor, in_std: torch.Tensor,
               tgt_mean: torch.Tensor, tgt_std: torch.Tensor,
               **extra) -> None:
    torch.save({
        "state_dict": recon.state_dict(),
        "epoch": epoch,
        "val_total": val["tot"],
        "val_dx":    val["dx"],   "val_dy":    val["dy"],
        "val_dz":    val["dz"],   "val_logE":  val["logE"],
        "input_features": RECON_INPUT_FEATURES,
        "num_detectors":  N_DETECTORS,
        "input_mean":  in_mean.detach().cpu(),
        "input_std":   in_std .detach().cpu(),
        "target_mean": tgt_mean.detach().cpu(),
        "target_std":  tgt_std .detach().cpu(),
        "config": dict(
            model_type="deepsets",
            hidden=HIDDEN, context=CONTEXT,
            n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
            output_dim=4,
            # What the recon's (E, T) inputs were during training. Stage 4 and
            # true_utility.py feed it different response sources, so a checkpoint
            # that does not record this cannot be told apart from one trained the
            # other way. Old checkpoints have no key -> read as "sample".
            recon_input=RECON_INPUT,
            mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
        ),
        **extra,
    }, path)


def _plot_curves(log, path: str, adam_epochs: int = 0,
                 lbfgs_iter_log=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        AXIS_STYLES = [("dx", "-"), ("dy", "--"), ("dz", ":"), ("logE", "-.")]

        adam_log = [e for e in log if e.get("phase") != "lbfgs"]
        ep = [e["epoch"] for e in adam_log]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train"] for e in adam_log], color="C0", label="train")
        axes[0].plot(ep, [e["val"]   for e in adam_log], color="C1", label="val")
        for name, ls in AXIS_STYLES:
            axes[1].plot(ep, [e[f"train_{name}"] for e in adam_log], color="C0", linestyle=ls)
            axes[1].plot(ep, [e[f"val_{name}"]   for e in adam_log], color="C1", linestyle=ls)

        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"] for e in lbfgs_iter_log], color="C0")
            axes[0].plot(lb_ep, [e["val"]  for e in lbfgs_iter_log], color="C1")
            for name, ls in AXIS_STYLES:
                axes[1].plot(lb_ep, [e[f"mse_{name}"] for e in lbfgs_iter_log], color="C0", linestyle=ls)
                axes[1].plot(lb_ep, [e[f"val_{name}"] for e in lbfgs_iter_log], color="C1", linestyle=ls)
            axes[0].set_yscale("log"); axes[1].set_yscale("log")

        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam→L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (normalized labels)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE")
        axes[1].set_title("per-axis"); axes[1].grid(alpha=0.3)
        axes[1].legend(handles=[
            Line2D([], [], color="C0",                    label="train"),
            Line2D([], [], color="C1",                    label="val"),
            Line2D([], [], color="black", linestyle="-",  label="dx"),
            Line2D([], [], color="black", linestyle="--", label="dy"),
            Line2D([], [], color="black", linestyle=":",  label="dz"),
            Line2D([], [], color="black", linestyle="-.", label="logE"),
        ], ncol=2, fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    global N_EPOCHS, LBFGS_MAX_ITER
    global RECON_INPUT, MIX_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int, default=N_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    ap.add_argument("--recon-input", choices=("sample", "kernel", "mix"),
                    default=RECON_INPUT,
                    help="source of the per-detector (E, T) recon inputs: "
                         "'sample' = a draw from the stage-2 surrogate (default, "
                         "historical); 'kernel' = the real Step-1 labels; 'mix' = "
                         "per-row coin flip. See the module docstring for why "
                         "'sample' alone leaves a standing surrogate-vs-true gap.")
    ap.add_argument("--mix-frac", type=float, default=MIX_FRAC,
                    help="P(row uses kernel labels) when --recon-input mix")
    args = ap.parse_args()
    RECON_INPUT, MIX_FRAC = args.recon_input, float(args.mix_frac)
    N_EPOCHS, LBFGS_MAX_ITER = int(args.epochs), int(args.lbfgs_iters)

    print("=" * 72)
    print("v6/03_train_recon_deepsets.py — DeepSets recon on dual-species preds")
    print("=" * 72)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"training data : {TRAINING_DATASET_FOLDER}")
    print(f"fnn ckpts     : {FNN_FOLDER}  (fnn_electron.pt + fnn_muon.pt)")
    print(f"output        : {OUTPUT_FOLDER}")
    print(f"device        : {DEVICE}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"arch          : hidden={HIDDEN} context={CONTEXT} n_enc={N_ENC} n_dec={N_DEC} pool={POOL}")

    t0 = time.time()
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    print(f"[load] corpus in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}")

    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)

    # The surrogate draw is computed even for --recon-input kernel: it costs one
    # pass and its summary stats beside the kernel's are the diagnostic for how
    # far apart the two response distributions actually are — the quantity the
    # whole surrogate-vs-true artifact gap comes from.
    t0 = time.time()
    E_samp, T_samp = compute_fnn_predictions(dual, primary, xy, DEVICE)
    print(f"[dual] sampled predictions in {time.time() - t0:.1f}s  "
          f"E mean={E_samp.mean():.3g} std={E_samp.std():.3g}  "
          f"T mean={T_samp.mean():.3g} std={T_samp.std():.3g}")

    t0 = time.time()
    E_kern, T_kern = load_kernel_responses(TRAINING_DATASET_FOLDER)
    print(f"[kernel] real labels in {time.time() - t0:.1f}s  "
          f"E mean={E_kern.mean():.3g} std={E_kern.std():.3g}  "
          f"T mean={T_kern.mean():.3g} std={T_kern.std():.3g}")
    if E_kern.shape != E_samp.shape:
        raise RuntimeError(
            f"kernel labels {tuple(E_kern.shape)} do not match the surrogate "
            f"draw {tuple(E_samp.shape)} — E.pt/T.pt are not row-aligned with "
            f"primary.pt, so Step 1 and Step 2/3 disagree on the corpus")
    print(f"[delta] surrogate-minus-kernel  E {(E_samp - E_kern).abs().mean():.4g}"
          f"  T {(T_samp - T_kern).abs().mean():.4g}  (mean |diff|)")

    print(f"[input] recon-input={RECON_INPUT}"
          + (f"  mix_frac={MIX_FRAC}" if RECON_INPUT == "mix" else ""))
    E_pred, T_pred = build_recon_inputs(
        RECON_INPUT, MIX_FRAC, SEED, E_samp, T_samp, E_kern, T_kern)
    del E_samp, T_samp, E_kern, T_kern

    target   = primary[:, :4].clone().float()
    tgt_mean = target.mean(dim=0)
    tgt_std  = target.std(dim=0).clamp(min=1e-8)
    print(f"[target] mean={tgt_mean.tolist()}  std={tgt_std.tolist()}")

    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train={len(train_idx)}  val={len(val_idx)}")

    # Per-feature z-score: one scalar per feature kind, broadcast over all slots.
    in_mean = torch.stack([xy[..., 0].mean(), xy[..., 1].mean(),
                           E_pred.mean(),      T_pred.mean()])         # (4,)
    in_std  = torch.stack([xy[..., 0].std(),  xy[..., 1].std(),
                           E_pred.std(),       T_pred.std()]).clamp(min=1e-8)
    print(f"[norm] per-feat mean={in_mean.tolist()}  std={in_std.tolist()}")

    full_ds  = TensorDataset(xy, E_pred, T_pred, target)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    torch.manual_seed(SEED)
    recon = DeepSetsRecon(
        n_det=N_DETECTORS, input_features=RECON_INPUT_FEATURES, output_dim=4,
        hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
    ).to(DEVICE)
    recon.set_normalization(
        in_mean  = in_mean.to(DEVICE),
        in_std   = in_std.to(DEVICE),
        out_mean = tgt_mean.to(DEVICE),
        out_std  = tgt_std.to(DEVICE),
    )
    n_params = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    print(f"[model] DeepSetsRecon  params={n_params:,}")

    optimizer = torch.optim.Adam(recon.parameters(), lr=LR)

    # gpu_requeue can preempt mid-training; recon_resume.pt lets the job
    # continue from the last RESUME_CKPT_INTERVAL-epoch checkpoint rather than
    # restarting cold.  adam_done=True means Adam finished; jump to L-BFGS.
    resume_path = os.path.join(OUTPUT_FOLDER, "recon_resume.pt")
    adam_done   = False
    start_epoch = 0
    log        = []
    best_val   = float("inf")
    best_epoch = -1
    if os.path.exists(resume_path):
        _r = torch.load(resume_path, map_location=DEVICE)
        if _r.get("adam_done"):
            adam_done = True
            print("[resume] Adam done, jumping to L-BFGS")
        else:
            recon.load_state_dict(_r["state_dict"])
            optimizer.load_state_dict(_r["optimizer"])
            start_epoch, log = _r["epoch"], _r["log"]
            best_val, best_epoch = _r["best_val"], _r["best_epoch"]
            print(f"[resume] epoch {start_epoch}  best_val={best_val:.6f}")

    for epoch in range(start_epoch if not adam_done else N_EPOCHS, N_EPOCHS):
        t_epoch = time.time()
        recon.train()
        sums = [0.0] * 5
        n_tr = 0
        for xy_b, E_b, T_b, tgt_b in train_loader:
            xy_b  = xy_b .to(DEVICE, non_blocking=True)
            E_b   = E_b  .to(DEVICE, non_blocking=True)
            T_b   = T_b  .to(DEVICE, non_blocking=True)
            tgt_b = tgt_b.to(DEVICE, non_blocking=True)

            inp    = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)
            losses = _per_axis_loss(recon(inp), tgt_b)

            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            B = xy_b.shape[0]
            for i, v in enumerate(losses):
                sums[i] += v.item() * B
            n_tr += B
        n_tr = max(n_tr, 1)
        tr = {k: sums[i] / n_tr for i, k in enumerate(_AXIS_KEYS)}

        recon.eval()
        va = _validate(recon, val_loader, DEVICE)

        dt = time.time() - t_epoch
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr['tot']:.4f} val={va['tot']:.4f} "
              f"(dx={va['dx']:.4f} dy={va['dy']:.4f} dz={va['dz']:.4f} "
              f"logE={va['logE']:.4f})  {dt:.1f}s")
        log.append(dict(
            epoch=epoch + 1,
            train=tr['tot'],
            train_dx=tr['dx'], train_dy=tr['dy'], train_dz=tr['dz'], train_logE=tr['logE'],
            val=va['tot'],
            val_dx=va['dx'],   val_dy=va['dy'],   val_dz=va['dz'],   val_logE=va['logE'],
            dt=dt,
        ))

        if va['tot'] < best_val - 1e-6:
            best_val   = va['tot']
            best_epoch = epoch + 1
            _save_ckpt(os.path.join(OUTPUT_FOLDER, "recon.pt"),
                       recon, epoch + 1, va, in_mean, in_std, tgt_mean, tgt_std)

        if (epoch + 1) % RESUME_CKPT_INTERVAL == 0:
            torch.save({
                "state_dict": recon.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1, "log": log,
                "best_val": best_val, "best_epoch": best_epoch,
            }, resume_path)

    if not adam_done:
        with open(os.path.join(OUTPUT_FOLDER, "recon_train_log.json"), "w") as f:
            json.dump({
                "log": log,
                "best_val_total": best_val,
                "best_epoch": best_epoch,
                "config": dict(
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                    grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                    hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
                    recon_input=RECON_INPUT,
                    mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
                ),
            }, f, indent=2)
        _plot_curves(log, os.path.join(OUTPUT_FOLDER, "recon_train_curves.png"))
        print(f"[adam done] best val {best_val:.4f} at epoch {best_epoch}")
        torch.save({"adam_done": True}, resume_path)

    # ── Phase 2: L-BFGS fine-tuning ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2: L-BFGS fine-tuning (full-batch)")
    print("=" * 72)

    adam_ckpt = torch.load(os.path.join(OUTPUT_FOLDER, "recon.pt"), map_location=DEVICE)
    recon.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best  epoch={adam_ckpt['epoch']}  val={adam_best_val:.6f}")

    recon.eval()   # dropout off; grad stays True

    # Pre-stack full training set as (N_train, n_det, 4) on GPU.
    xy_train  = xy[train_idx].to(DEVICE)
    E_train   = E_pred[train_idx].to(DEVICE)
    T_train   = T_pred[train_idx].to(DEVICE)
    tgt_train = target[train_idx].to(DEVICE)
    inp_all   = torch.stack([xy_train[..., 0], xy_train[..., 1],
                             E_train, T_train], dim=-1)    # (N_train, n_det, 4)
    del xy_train, E_train, T_train
    print(f"[lbfgs] full train set on {DEVICE}: {tgt_train.shape[0]} samples")

    lbfgs_optimizer = torch.optim.LBFGS(
        recon.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter_log  = []
    lbfgs_best_val  = adam_best_val
    lbfgs_best_iter = -1
    N_train = inp_all.shape[0]
    t_lbfgs = time.time()

    def closure():
        nonlocal lbfgs_best_val, lbfgs_best_iter
        lbfgs_optimizer.zero_grad()
        sums = [0.0] * 5
        for lo in range(0, N_train, LBFGS_CHUNK):
            hi = min(lo + LBFGS_CHUNK, N_train)
            losses = _per_axis_loss(recon(inp_all[lo:hi]),
                                    tgt_train[lo:hi], reduction="sum")
            (losses[0] / N_train).backward()
            for i, v in enumerate(losses):
                sums[i] += v.item()
        tr = {k: sums[i] / N_train for i, k in enumerate(_AXIS_KEYS)}
        va = _validate(recon, val_loader, DEVICE)

        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(
            iter=it, loss=tr['tot'],
            mse_dx=tr['dx'], mse_dy=tr['dy'], mse_dz=tr['dz'], mse_logE=tr['logE'],
            val=va['tot'],
            val_dx=va['dx'], val_dy=va['dy'], val_dz=va['dz'], val_logE=va['logE'],
        ))
        marker = ""
        if va['tot'] < lbfgs_best_val - 1e-6:
            lbfgs_best_val  = va['tot']
            lbfgs_best_iter = it
            _save_ckpt(os.path.join(OUTPUT_FOLDER, "recon.pt"),
                       recon, N_EPOCHS + 1, va, in_mean, in_std, tgt_mean, tgt_std,
                       phase="lbfgs", lbfgs_iter=it)
            marker = "  <- NEW BEST (saved)"
        print(f"  [lbfgs iter {it:3d}] loss={tr['tot']:.6f} "
              f"val={va['tot']:.6f}{marker}")
        return torch.tensor(tr['tot'], device=DEVICE)

    final_loss = lbfgs_optimizer.step(closure)
    del inp_all, tgt_train
    if os.path.exists(resume_path):
        os.remove(resume_path)

    dt_lbfgs = time.time() - t_lbfgs
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)
    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {len(lbfgs_iter_log)} iters")
        recon.load_state_dict(adam_ckpt["state_dict"])
    print(f"[lbfgs] {len(lbfgs_iter_log)} iterations in {dt_lbfgs:.1f}s")

    overall_best_val = lbfgs_best_val
    lbfgs_log = []
    if not lbfgs_abort and lbfgs_iter_log:
        last = lbfgs_iter_log[-1]
        lbfgs_log.append(dict(
            epoch=N_EPOCHS + 1, phase="lbfgs",
            train=last["loss"],
            train_dx=last["mse_dx"], train_dy=last["mse_dy"],
            train_dz=last["mse_dz"], train_logE=last["mse_logE"],
            val=last["val"],
            val_dx=last["val_dx"], val_dy=last["val_dy"],
            val_dz=last["val_dz"], val_logE=last["val_logE"],
            dt=dt_lbfgs,
        ))

    if lbfgs_best_iter >= 0:
        print(f"[lbfgs] best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter} "
              f"(Adam was {adam_best_val:.6f}, gain={adam_best_val - lbfgs_best_val:.6f})")
    else:
        print(f"[lbfgs] no improvement over Adam  (recon.pt unchanged)")

    full_log = log + lbfgs_log
    with open(os.path.join(OUTPUT_FOLDER, "recon_train_log.json"), "w") as f:
        json.dump({
            "log": full_log,
            "lbfgs_iter_log": lbfgs_iter_log,
            "best_val_total": overall_best_val,
            "best_epoch": best_epoch if overall_best_val == adam_best_val else N_EPOCHS + 1,
            "adam_best_val": adam_best_val,
            "adam_best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
                recon_input=RECON_INPUT,
                mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
            ),
        }, f, indent=2)
    _plot_curves(full_log, os.path.join(OUTPUT_FOLDER, "recon_train_curves.png"),
                 adam_epochs=N_EPOCHS, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[done] best recon val {overall_best_val:.4f}  -> {OUTPUT_FOLDER}")

    # Auto-render target-vs-prediction scatter using best checkpoint.
    try:
        _best = torch.load(os.path.join(OUTPUT_FOLDER, "recon.pt"), map_location=DEVICE)
        recon.load_state_dict(_best["state_dict"])
        recon.eval()
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_plot_tvp",
            os.path.join(V6_ROOT, "plots", "training", "02_nn_target_vs_pred.py"),
        )
        assert _spec is not None and _spec.loader is not None
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.plot_recon_only(fnn=dual, recon=recon, primary=primary, xy=xy,
                             val_idx=val_idx,
                             output_path=os.path.join(OUTPUT_FOLDER,
                                                      "recon_target_vs_pred.png"))
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


if __name__ == "__main__":
    main()

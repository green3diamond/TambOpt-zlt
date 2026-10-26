"""Stage 3: train the reconstruction network (DeepSets; --model settf: set transformer) on Stage-1 responses.

Per detector token (x, y, E, T) -> pooled (maxmean) -> decoder -> (dir_x, dir_y, dir_z, log_e_norm). Permutation
invariant by construction. --input-mode reltime1 (run 19): positions and times relative to the earliest timed detector.
(E, T) inputs from --recon-input:
  kernel  the kernel labels of Stage 1 (E.pt / T.pt, species combined) -- run 18 / 19;
  sample  the Stage 2 surrogate's prediction for each row (default, historical name);
  mix     a per-row coin flip between the two (--mix-frac = P(kernel)).
Shower-level split shared by all strategies and species of a shower (SEED, as Stage 2's SPLIT_SEED); blob rows dropped.
Adam/AdamW (--weight-decay), then chunked full-batch L-BFGS; both resume after a preemption (recon_resume.pt,
lbfgs_resume.pt). Output: RECON_FOLDER + "_deepsets"/recon.pt.

    python scripts/03_train_recon_deepsets.py --recon-input kernel --input-mode reltime1 --weight-decay 0.03 \
        --adam-patience 50                                    # slurm/run19_pipeline.sh recon
    python scripts/03_train_recon_deepsets.py --epochs 2 --lbfgs-iters 3   # smoke run
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
from modules.surrogates  import DeepSetsRecon, SetTransformerRecon
from modules.surrogates.recon import INPUT_MODES, RELTIME_FEATURES, encoder_features
from modules.surrogates.dual import combine_species_outputs
from modules.constants import (
    RECON_FOLDER, TRAINING_DATASET_FOLDER, FNN_FOLDER,
    N_DETECTORS, T_LOG_SCALE, TRAIN_FRACTION,
)

OUTPUT_FOLDER = RECON_FOLDER + "_deepsets"

RECON_INPUT   = "sample"     # "sample" | "kernel" | "mix"
INPUT_MODE    = "raw"        # "raw" | "reltime" | "reltime1" (encoder input transform, see recon.py)
WEIGHT_DECAY  = 0.0          # > 0: AdamW with this decoupled weight decay (per step the weights shrink by LR * WEIGHT_DECAY)
MIX_FRAC      = 0.5          # P(row uses kernel labels) when RECON_INPUT == "mix"

# ── Architecture ─────────────────────────────────────────────────────────────
# The encoder runs once per detector, the decoder once per event: encoder depth is ~100x costlier.
HIDDEN   = 256
CONTEXT  = 128
N_ENC    = 4
N_DEC    = 4
POOL     = "maxmean"   # "mean" or "maxmean"; maxmean doubles context_proj input
# --model settf: SetTransformerRecon (recon.py) -- the encoder above maps to D_MODEL tokens, N_BLOCKS attention
# blocks with a pair-feature bias, masked maxmean pool, the same decoder.
MODEL       = "deepsets"   # "deepsets" | "settf"
D_MODEL     = 128
N_HEADS     = 4
N_BLOCKS    = 3
PAIR_HIDDEN = 32
HIT_E_MIN   = 0.1
HIT_TAU     = 0.5


def _arch_cfg() -> dict:
    """Architecture keys saved in the checkpoint config (build_recon_from_ckpt reads them)."""
    cfg = dict(model_type=MODEL, hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL)
    if MODEL == "settf":
        cfg.update(d_model=D_MODEL, n_heads=N_HEADS, n_blocks=N_BLOCKS, pair_hidden=PAIR_HIDDEN,
                   hit_e_min=HIT_E_MIN, hit_tau=HIT_TAU)
    return cfg

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
N_EPOCHS   = 600
ADAM_PATIENCE = 0      # Adam epochs without val improvement before stopping (0 = off)
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
LBFGS_CHUNK        = 4_096   # rows per forward in the closure (encoder activations set the memory)
LBFGS_PATIENCE     = 50      # closure calls without val improvement before stopping
# The sweep is repeated step() calls of this many iterations, so a preemption resumes with the L-BFGS history intact
# (the optimizer state is only consistent between step() calls, not inside the closure).
LBFGS_STEP_ITERS    = 25
# Resume checkpoint every epoch (5 MB; an epoch takes minutes and preemptions come every 1-3 h).
RESUME_CKPT_INTERVAL = 1

RECON_INPUT_FEATURES = 4   # (x, y, E, T) per detector


def shower_level_split(strategy_ids: torch.Tensor,
                       species_ids: torch.Tensor,
                       val_frac: float,
                       seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split: every strategy and species row of one shower lands on the same side (their combined kernel
    target is identical, so splitting by row would leak). Rows are strategy-major, species-major, event."""
    n_pairs   = int(strategy_ids.shape[0])
    n_strat   = int(strategy_ids.max().item() + 1)
    n_species = int(species_ids.max().item() + 1)
    n_showers = n_pairs // (n_strat * n_species)

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(val_frac * n_showers)))

    is_val = torch.zeros(n_showers, dtype=torch.bool)
    is_val[perm[:n_val]] = True

    all_idx = torch.arange(n_pairs, dtype=torch.long)
    shower_of_pair = all_idx - strategy_ids * (n_species * n_showers) - species_ids * n_showers
    val_mask = is_val[shower_of_pair]
    return (torch.nonzero(~val_mask).squeeze(-1),
            torch.nonzero(val_mask).squeeze(-1))


@torch.no_grad()
def compute_fnn_predictions(model: nn.Module,
                             primary: torch.Tensor,
                             xy:      torch.Tensor,
                             device:  torch.device,
                             batch_size: int = 1024,
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The dual surrogate's prediction per row over the whole corpus, on CPU.

    Computed once here and reused for every epoch — the same "one fixed
    realization per (primary, layout) row" convention the real kernel-labeled
    corpus already uses.

    This used to take `mean_only` and dispatch to `forward_sample` by default,
    drawing `mean + eps*std` from the stage-2 variance head. That head no longer
    exists (the surrogate is mean-only), and in practice it was never trained
    anyway — every run since 13 passed `--nll-warmup-frac 1.0` and then had to
    pass `--fnn-mean-only` here to avoid sampling untrained-head noise. The mean
    is now the only path, so the parameter and the flag are both gone."""
    model.eval()
    N = int(primary.shape[0])
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    predict = model.forward
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        pred = predict(
            primary[lo:hi].to(device, non_blocking=True),
            xy[lo:hi].to(device, non_blocking=True))            # (B, 100, 2)
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


def load_kernel_responses(folder: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The REAL kernel labels Step 1 wrote, COMBINED across species and put in
    the dual surrogate's own output space.

    Two transforms are needed, and skipping either silently trains recon on the
    wrong quantity:

    1. SPECIES COMBINATION. E.pt/T.pt are PER-SPECIES — Step 2 trains one
       surrogate per species against them, so row i holds only that row's own
       species' response (electron rows are frequently all-zero while the muon
       and photon rows of the SAME event carry the signal). What recon consumes
       at every other call site is `DualSpeciesSurrogate.forward`, which returns
       the species-COMBINED event. Feeding the raw per-species rows hands recon
       one third of an event and asks it to reconstruct the whole thing. Combine
       with the very function the surrogate uses (`combine_species_outputs`:
       counts add, T is the leading edge, i.e. min over species that hit) so the
       two sources are the same physical quantity.
    2. UNITS. Step 1 saves E already log1p'd (`01_build_dataset_northeast.py:117`)
       but T RAW; Step 2 is what applies `log1p(T * T_LOG_SCALE)`. So E needs no
       transform and T needs Step 2's before combining.

    Row order is strategy-major then species-major then event (asserted below —
    a silent mismatch here is exactly the failure this docstring exists to
    prevent). The combined event is broadcast back over all species rows, so the
    return stays row-aligned with primary.pt / xy.pt and keeps the same
    (N, N_DETECTORS) shape compute_fnn_predictions returns.
    """
    E = torch.load(os.path.join(folder, "E.pt")).float()
    T = torch.load(os.path.join(folder, "T.pt")).float()
    sp = torch.load(os.path.join(folder, "species_ids.pt")).long()
    st = torch.load(os.path.join(folder, "strategy_ids.pt")).long()
    T = torch.log1p(T * T_LOG_SCALE)

    N, n_det = E.shape
    n_species = int(sp.max().item()) + 1
    n_strat   = int(st.max().item()) + 1
    if N % (n_species * n_strat):
        raise RuntimeError(
            f"{N} rows is not divisible by n_species={n_species} x "
            f"n_strat={n_strat} — the corpus is not the rectangular "
            f"strategy/species/event block this assumes")
    n_ev = N // (n_species * n_strat)

    # Assert the block order rather than trust it: view() below is a reinterpret,
    # so a different ordering would mis-pair species silently and train on
    # nonsense that still has the right shape.
    exp_sp = torch.arange(n_species).repeat_interleave(n_ev).repeat(n_strat)
    exp_st = torch.arange(n_strat).repeat_interleave(n_species * n_ev)
    if not torch.equal(sp, exp_sp) or not torch.equal(st, exp_st):
        raise RuntimeError(
            "species_ids/strategy_ids are not in strategy-major, species-major, "
            "event order; load_kernel_responses' reshape would mis-pair species")

    E4 = E.view(n_strat, n_species, n_ev, n_det)
    T4 = T.view(n_strat, n_species, n_ev, n_det)
    per_species = [torch.stack([E4[:, k], T4[:, k]], dim=-1)      # (S, ev, det, 2)
                   for k in range(n_species)]
    comb = combine_species_outputs(*per_species)                  # (S, ev, det, 2)
    # Same combined event for every species row of that (strategy, event).
    comb = (comb.unsqueeze(1)
                .expand(n_strat, n_species, n_ev, n_det, 2)
                .reshape(N, n_det, 2))
    return comb[..., 0].contiguous(), comb[..., 1].contiguous()


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
               recon: nn.Module,
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
            **_arch_cfg(),
            output_dim=4,
            # The (E, T) source it was trained on (old checkpoints: missing -> "sample").
            recon_input=RECON_INPUT,
            mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
            input_mode=INPUT_MODE,
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
    global N_EPOCHS, LBFGS_MAX_ITER, ADAM_PATIENCE
    global RECON_INPUT, MIX_FRAC, OUTPUT_FOLDER, INPUT_MODE, WEIGHT_DECAY, MODEL, N_BLOCKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int, default=N_EPOCHS)
    ap.add_argument("--adam-patience", type=int, default=ADAM_PATIENCE,
                    help="stop Adam after this many epochs without val improvement (0 = off)")
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
    ap.add_argument("--out-suffix", type=str, default=None,
                    help="append _<suffix> to the output folder, e.g. "
                         "--out-suffix expB1 writes to RECON_FOLDER + "
                         "'_deepsets_expB1' instead of the production "
                         "recon.pt. Use for any run that is not meant to "
                         "become the checkpoint Stage 4 / true_utility.py load.")
    ap.add_argument("--input-mode", choices=INPUT_MODES, default=INPUT_MODE,
                    help="'reltime': positions relative to the first timed hit, linear dt in ns, lit (E > 0) "
                         "and timed (T > 0) flags, and the reference's absolute position/time "
                         "(recon.reltime_features); 'reltime1': the same with the reference at the earliest timed "
                         "detector with >= 1 count")
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                    help="AdamW decoupled weight decay for the Adam phase (0 = plain Adam, as before)")
    ap.add_argument("--model", choices=("deepsets", "settf"), default=MODEL,
                    help="'settf': SetTransformerRecon (attention between detectors with a pair-feature bias)")
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS, help="attention blocks for --model settf")
    args = ap.parse_args()
    MODEL, N_BLOCKS = args.model, int(args.n_blocks)
    RECON_INPUT, MIX_FRAC = args.recon_input, float(args.mix_frac)
    INPUT_MODE = args.input_mode
    WEIGHT_DECAY = float(args.weight_decay)
    N_EPOCHS, LBFGS_MAX_ITER = int(args.epochs), int(args.lbfgs_iters)
    ADAM_PATIENCE = int(args.adam_patience)
    if args.out_suffix:
        OUTPUT_FOLDER = OUTPUT_FOLDER + "_" + args.out_suffix

    print("=" * 72)
    print("v6/03_train_recon_deepsets.py — DeepSets recon on dual-species preds")
    print("=" * 72)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"training data : {TRAINING_DATASET_FOLDER}")
    print(f"fnn ckpts     : {FNN_FOLDER}  (fnn_electron.pt + fnn_muon.pt)")
    print(f"output        : {OUTPUT_FOLDER}")
    print(f"device        : {DEVICE}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"arch          : {_arch_cfg()}")

    t0 = time.time()
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    species_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
    # Degenerate (blob) showers flagged at Stage 1: dropped from train, val and stats; a missing file is an error.
    _blob_path = os.path.join(TRAINING_DATASET_FOLDER, "blob_ids.pt")
    if not os.path.exists(_blob_path):
        raise SystemExit(
            f"blob_ids.pt not found in {TRAINING_DATASET_FOLDER}.\n"
            "This dataset predates the degenerate-shower filter, so its E/T "
            "labels may still contain blobs. Re-run 01_build_dataset_northeast.py.")
    blob_ids = torch.load(_blob_path)
    keep = ~blob_ids.bool()
    print(f"[load] corpus in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}  "
          f"blob={100 * (~keep).float().mean():.2f}%")

    # --recon-input kernel needs no surrogate; with sample / mix its predictions are computed.
    dual = None if RECON_INPUT == "kernel" else load_dual_surrogate(FNN_FOLDER, DEVICE)

    t0 = time.time()
    E_kern, T_kern = load_kernel_responses(TRAINING_DATASET_FOLDER)
    print(f"[kernel] real labels in {time.time() - t0:.1f}s  "
          f"E mean={E_kern.mean():.3g} std={E_kern.std():.3g}  "
          f"T mean={T_kern.mean():.3g} std={T_kern.std():.3g}")
    if dual is None:
        E_samp, T_samp = E_kern, T_kern          # unused by build_recon_inputs("kernel")
        print("[dual] skipped (--recon-input kernel)")
    else:
        t0 = time.time()
        E_samp, T_samp = compute_fnn_predictions(dual, primary, xy, DEVICE)
        print(f"[dual] mean predictions "
              f"in {time.time() - t0:.1f}s  "
              f"E mean={E_samp.mean():.3g} std={E_samp.std():.3g}  "
              f"T mean={T_samp.mean():.3g} std={T_samp.std():.3g}")
    if E_kern.shape != E_samp.shape:
        raise RuntimeError(
            f"kernel labels {tuple(E_kern.shape)} do not match the surrogate "
            f"draw {tuple(E_samp.shape)} — E.pt/T.pt are not row-aligned with "
            f"primary.pt, so Step 1 and Step 2/3 disagree on the corpus")
    if dual is not None:
        print(f"[delta] surrogate-minus-kernel  E {(E_samp - E_kern).abs().mean():.4g}"
              f"  T {(T_samp - T_kern).abs().mean():.4g}  (mean |diff|)")

    print(f"[input] recon-input={RECON_INPUT}"
          + (f"  mix_frac={MIX_FRAC}" if RECON_INPUT == "mix" else ""))
    E_pred, T_pred = build_recon_inputs(
        RECON_INPUT, MIX_FRAC, SEED, E_samp, T_samp, E_kern, T_kern)
    del E_samp, T_samp, E_kern, T_kern

    target   = primary[:, :4].clone().float()
    # Stats over the clean (non-blob) rows only.
    tgt_mean = target[keep].mean(dim=0)
    tgt_std  = target[keep].std(dim=0).clamp(min=1e-8)
    print(f"[target] mean={tgt_mean.tolist()}  std={tgt_std.tolist()}")

    train_idx, val_idx = shower_level_split(strat_ids, species_ids, VAL_FRAC, SEED)
    n_tr0, n_va0 = len(train_idx), len(val_idx)
    train_idx = train_idx[keep[train_idx]]
    val_idx   = val_idx[keep[val_idx]]
    print(f"[split] train={len(train_idx)}  val={len(val_idx)}  "
          f"(dropped {n_tr0 - len(train_idx)} train / {n_va0 - len(val_idx)} val blob rows)")
    # TRAIN_FRACTION < 1: the same slice of showers as Stage 2 (same constant and seed).
    if 0.0 < TRAIN_FRACTION < 1.0:
        _n_orig = int(train_idx.shape[0])
        _n_keep = max(1, int(round(TRAIN_FRACTION * _n_orig)))
        _g_sub = torch.Generator().manual_seed(SEED)
        _perm = torch.randperm(_n_orig, generator=_g_sub)
        train_idx = train_idx[_perm[:_n_keep]]
        print(f"[subsample] kept {_n_keep} of {_n_orig} train pairs (TRAIN_FRACTION={TRAIN_FRACTION})")

    # Per-feature z-score: one scalar per feature kind, broadcast over all slots.
    in_mean = torch.stack([xy[keep, :, 0].mean(), xy[keep, :, 1].mean(),
                           E_pred[keep].mean(),    T_pred[keep].mean()])   # (4,)
    in_std  = torch.stack([xy[keep, :, 0].std(),  xy[keep, :, 1].std(),
                           E_pred[keep].std(),     T_pred[keep].std()]).clamp(min=1e-8)
    if INPUT_MODE != "raw":
        # Stats of the transformed features the encoder sees, on a seeded sample of 300k clean rows.
        _g = torch.Generator().manual_seed(SEED)
        _ok = torch.nonzero(keep).squeeze(-1)                 # clean rows, as the raw stats above
        _rows = _ok[torch.randperm(_ok.numel(), generator=_g)[:300_000]]
        _f = []
        with torch.no_grad():
            for _lo in range(0, _rows.numel(), 8192):
                _r = _rows[_lo:_lo + 8192]
                _x = torch.stack([xy[_r, :, 0], xy[_r, :, 1], E_pred[_r], T_pred[_r]], dim=-1).to(DEVICE)
                _f.append(encoder_features(_x, INPUT_MODE).reshape(-1, RELTIME_FEATURES).cpu())
        _f = torch.cat(_f)
        in_mean, in_std = _f.mean(0), _f.std(0).clamp(min=1e-8)
        del _f
    print(f"[norm] input_mode={INPUT_MODE} per-feat mean={in_mean.tolist()}  std={in_std.tolist()}")

    full_ds  = TensorDataset(xy, E_pred, T_pred, target)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    torch.manual_seed(SEED)
    if MODEL == "settf":
        recon = SetTransformerRecon(
            n_det=N_DETECTORS, input_features=RECON_INPUT_FEATURES, output_dim=4,
            hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, d_model=D_MODEL, n_heads=N_HEADS,
            n_blocks=N_BLOCKS, pair_hidden=PAIR_HIDDEN, hit_e_min=HIT_E_MIN, hit_tau=HIT_TAU, input_mode=INPUT_MODE,
        ).to(DEVICE)
    else:
        recon = DeepSetsRecon(
            n_det=N_DETECTORS, input_features=RECON_INPUT_FEATURES, output_dim=4,
            hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
            input_mode=INPUT_MODE,
        ).to(DEVICE)
    recon.set_normalization(
        in_mean  = in_mean.to(DEVICE),
        in_std   = in_std.to(DEVICE),
        out_mean = tgt_mean.to(DEVICE),
        out_std  = tgt_std.to(DEVICE),
    )
    n_params = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    print(f"[model] {type(recon).__name__}  params={n_params:,}")

    optimizer = (torch.optim.AdamW(recon.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) if WEIGHT_DECAY > 0
                 else torch.optim.Adam(recon.parameters(), lr=LR))
    print(f"[optim] {type(optimizer).__name__} lr={LR} weight_decay={WEIGHT_DECAY}")

    # Resume from recon_resume.pt (adam_done=True: go straight to L-BFGS).
    resume_path = os.path.join(OUTPUT_FOLDER, "recon_resume.pt")
    adam_done   = False
    start_epoch = 0
    log        = []
    best_val   = float("inf")
    best_epoch = -1
    lbfgs_resume = None          # L-BFGS sweep state, if a preemption left one
    if os.path.exists(resume_path):
        # An unreadable resume file means a cold start, not a failure.
        try:
            _r = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        except Exception as exc:
            print(f"[resume] unreadable {resume_path} ({exc!r}) -- starting cold")
            _r = {}
        if not _r:
            pass
        elif _r.get("adam_done"):
            adam_done = True
            lbfgs_resume = _r.get("lbfgs")
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
            # tmp + os.replace: a preemption mid-write cannot leave a truncated file.
            tmp = resume_path + ".tmp"
            torch.save({
                "state_dict": recon.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1, "log": log,
                "best_val": best_val, "best_epoch": best_epoch,
            }, tmp)
            os.replace(tmp, resume_path)

        # Counted from the stored best epoch, so a resumed run stops at the same epoch.
        if ADAM_PATIENCE and epoch + 1 - best_epoch >= ADAM_PATIENCE:
            print(f"[adam] early stop: {ADAM_PATIENCE} epochs with no val improvement "
                  f"(best val={best_val:.6f} at epoch {best_epoch})")
            break

    if not adam_done:
        with open(os.path.join(OUTPUT_FOLDER, "recon_train_log.json"), "w") as f:
            json.dump({
                "log": log,
                "best_val_total": best_val,
                "best_epoch": best_epoch,
                "config": dict(
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, adam_patience=ADAM_PATIENCE, lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                    **_arch_cfg(),
                    recon_input=RECON_INPUT,
                    mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
                    input_mode=INPUT_MODE,
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

    # Pre-stack full training set as (N_train, n_det, 4) in pinned CPU memory; the closure moves it to the GPU
    # LBFGS_CHUNK rows at a time (run 20: the 13.8M-row set is 20 GB, too big for a 40 GB GPU next to its parts).
    xy_train  = xy[train_idx]
    inp_all   = torch.stack([xy_train[..., 0], xy_train[..., 1],
                             E_pred[train_idx], T_pred[train_idx]], dim=-1).pin_memory()   # (N_train, n_det, 4)
    tgt_train = target[train_idx].to(DEVICE)
    del xy_train
    print(f"[lbfgs] full train set in pinned CPU memory, streamed to {DEVICE}: {tgt_train.shape[0]} samples")

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
    lbfgs_no_improve = 0
    done_iters = 0            # cumulative L-BFGS iterations across step() calls
    N_train = inp_all.shape[0]
    t_lbfgs = time.time()

    class _LBFGSStop(Exception):
        """Raised inside the closure to break out of optimizer.step()."""

    # Resume a preempted sweep: restore the CURRENT iterate from the payload (the L-BFGS history belongs to it),
    # not the best-val weights in recon.pt.
    if lbfgs_resume is not None:
        recon.load_state_dict(lbfgs_resume["model"])
        lbfgs_optimizer.load_state_dict(lbfgs_resume["optimizer"])
        lbfgs_iter_log   = lbfgs_resume["iter_log"]
        lbfgs_best_val   = lbfgs_resume["best_val"]
        lbfgs_best_iter  = lbfgs_resume["best_iter"]
        lbfgs_no_improve = lbfgs_resume["no_improve"]
        # state['n_iter'] is cumulative and survives load_state_dict; the driver
        # loop below uses it to spend only the remaining budget.
        done_iters = int(
            lbfgs_optimizer.state[lbfgs_optimizer._params[0]].get("n_iter", 0))
        print(f"[resume] L-BFGS from closure {len(lbfgs_iter_log)} "
              f"(iters done={done_iters}/{LBFGS_MAX_ITER})  "
              f"best_val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter}")

    def closure():
        nonlocal lbfgs_best_val, lbfgs_best_iter, lbfgs_no_improve
        lbfgs_optimizer.zero_grad()
        sums = [0.0] * 5
        for lo in range(0, N_train, LBFGS_CHUNK):
            hi = min(lo + LBFGS_CHUNK, N_train)
            losses = _per_axis_loss(recon(inp_all[lo:hi].to(DEVICE, non_blocking=True)),
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
            lbfgs_no_improve = 0
            _save_ckpt(os.path.join(OUTPUT_FOLDER, "recon.pt"),
                       recon, N_EPOCHS + 1, va, in_mean, in_std, tgt_mean, tgt_std,
                       phase="lbfgs", lbfgs_iter=it)
            marker = "  <- NEW BEST (saved)"
        else:
            lbfgs_no_improve += 1
        print(f"  [lbfgs iter {it:3d}] loss={tr['tot']:.6f} "
              f"val={va['tot']:.6f}{marker}")

        if lbfgs_no_improve >= LBFGS_PATIENCE:
            raise _LBFGSStop
        return torch.tensor(tr['tot'], device=DEVICE)

    def _save_lbfgs_resume():
        """Persist the sweep. Only valid straight after step() returns — see
        LBFGS_STEP_ITERS. Temp file + os.replace so a preemption landing
        mid-save cannot leave a truncated payload (the same failure mode that
        once zeroed pipeline_status.json)."""
        tmp = resume_path + ".tmp"
        torch.save({"adam_done": True,
                    "lbfgs": {"model": recon.state_dict(),
                              "optimizer": lbfgs_optimizer.state_dict(),
                              "iter_log": lbfgs_iter_log,
                              "best_val": lbfgs_best_val,
                              "best_iter": lbfgs_best_iter,
                              "no_improve": lbfgs_no_improve}}, tmp)
        os.replace(tmp, resume_path)

    final_loss = torch.tensor(lbfgs_best_val, device=DEVICE)
    try:
        while (done_iters < LBFGS_MAX_ITER
               and lbfgs_no_improve < LBFGS_PATIENCE):
            budget = min(LBFGS_STEP_ITERS, LBFGS_MAX_ITER - done_iters)
            lbfgs_optimizer.param_groups[0]["max_iter"] = budget
            final_loss = lbfgs_optimizer.step(closure)
            # state["n_iter"] is cumulative across step() calls; the loop bound
            # inside step() is a LOCAL counter, so budget is "this many more".
            done_iters = int(
                lbfgs_optimizer.state[lbfgs_optimizer._params[0]]["n_iter"])
            _save_lbfgs_resume()
    except _LBFGSStop:
        print(f"[lbfgs] early stop: {LBFGS_PATIENCE} closure calls with no val "
              f"improvement (best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter})")
        final_loss = torch.tensor(lbfgs_best_val, device=DEVICE)

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
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                **_arch_cfg(),
                recon_input=RECON_INPUT,
                mix_frac=(MIX_FRAC if RECON_INPUT == "mix" else None),
                input_mode=INPUT_MODE,
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
        if dual is None:
            _mod.plot_recon_on_kernel(recon=recon, primary=primary, xy=xy, val_idx=val_idx,
                                      recon_folder=OUTPUT_FOLDER)
        else:
            _mod.plot_recon_only(fnn=dual, recon=recon, primary=primary, xy=xy,
                                 val_idx=val_idx,
                                 output_path=os.path.join(OUTPUT_FOLDER,
                                                          "recon_target_vs_pred.png"))
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


if __name__ == "__main__":
    main()

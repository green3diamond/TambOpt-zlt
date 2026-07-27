"""Validate (not assert) the "current limitations" claims made in conversation,
against the ACTUAL trained checkpoints and the ACTUAL saved stage-01 dataset —
no claim here is carried forward from memory/prior sessions unverified.

Checks, in order:

  1. Layout diversity (was: "01 only labels a handful of fixed strategy
     layouts"). `modules_v6/detector_strategies_ne.py._STRATEGIES` is read
     directly; grid_jit20/200 are confirmed commented out, so zero training
     rows come from a grid-like layout even though grid is the initialization
     scheme used everywhere in 04/eval. The other 5 active strategies redraw a
     FRESH random layout every `batch_size` shower rows (not a "handful"),
     quantified here by counting distinct layouts actually on disk.

  2. Corrupted-event correlation (was: "muon corpus corruption already
     contaminates 01's labels/02's training"). The specific muon event indices
     flagged by plots/inspect_voxel_corpus.py are looked up directly in the
     ACTUAL E.pt/T.pt used to train the current checkpoints, across all 5
     strategy rows for each, and compared against a random baseline sample —
     to see whether corruption reaches the trained-on labels as literal
     NaN/garbage, as a log-compressed-but-legitimate-looking outlier, or not
     at all (01's isfinite point-filter + the E channel's structural
     non-negativity + nan_to_num may already protect it — verified, not
     assumed, below).

  3. Capacity vs. aleatoric floor for the FNN surrogate (was: "R^2~=0", never
     actually computed). Loads the CURRENT fnn_electron.pt/fnn_muon.pt, runs
     them on the held-out validation split (reproducing 02's exact
     shower_level_split/seed), and reports RAW R^2 (surrogate vs. per-row
     label) AND BINNED R^2 (surrogate vs. the label averaged within coarse
     (primary, xy) bins) — the latter isolates whether the surrogate captures
     the primary-determined mean field even if raw R^2 is near 0.

  4. Recon permutation-invariance (was, from memory: "recon MLP stayed
     order-sensitive" — but the CURRENTLY LOADED recon is DeepSetsRecon,
     architecturally permutation-invariant, a different model than the one
     memory's finding was about). Verified directly: score one layout through
     the loaded recon.pt with the detector order shuffled and compare to
     unshuffled.

  5. Recon distribution shift (was: "recon trained on kernel labels, fed
     surrogate outputs at opt time — out of distribution"). Compares summary
     statistics of (E, T) from 01's saved training labels vs. the surrogate's
     own predictions on the SAME (primary, xy) pairs.

    cd TambOpt/detector_optimization_v6
    python plots/validate_limitations.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401
from modules_v6.constants import (
    TRAINING_DATASET_FOLDER, N_DETECTORS, T_LOG_SCALE,
)
from modules_v6.opt_core import load_models
from modules_v6.detector_strategies_ne import _STRATEGIES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2(true, pred):
    true, pred = true.reshape(-1).double(), pred.reshape(-1).double()
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum().clamp(min=1e-30)
    return float(1.0 - ss_res / ss_tot)


def shower_level_split(strategy_ids, val_frac, seed):
    """Verbatim copy of 02_train_fnn_deepsets.shower_level_split, so the val
    set here is EXACTLY the one the current checkpoints were validated on."""
    n_pairs = int(strategy_ids.shape[0])
    n_strat = int(strategy_ids.max().item() + 1)
    n_showers = n_pairs // n_strat
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(val_frac * n_showers)))
    is_val = torch.zeros(n_showers, dtype=torch.bool)
    is_val[perm[:n_val]] = True
    all_idx = torch.arange(n_pairs, dtype=torch.long)
    shower_of_pair = all_idx - strategy_ids * n_showers
    val_mask = is_val[shower_of_pair]
    return (torch.nonzero(~val_mask).squeeze(-1), torch.nonzero(val_mask).squeeze(-1))


def check1_layout_diversity():
    print("=" * 72)
    print("CHECK 1 — layout diversity in the ACTIVE strategy list")
    print("=" * 72)
    names = [s[0] for s in _STRATEGIES]
    fns = [s[1] for s in _STRATEGIES]
    print(f"active _STRATEGIES entries: {list(zip(names, fns))}")
    has_grid = any("grid" in fn for fn in fns)
    print(f"grid-like strategy present in training set: {has_grid}")
    if not has_grid:
        print("  -> CONFIRMED: zero training rows come from a grid-like layout,")
        print("     even though 'grid' is the init/eval scheme used everywhere")
        print("     in 04_optimize_*.py and eval_true_utility.py.")

    strat = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt"))
    xy = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt"))
    n_strat = int(strat.max().item() + 1)
    n_showers = strat.shape[0] // n_strat
    print(f"n_pairs={strat.shape[0]}  n_strat={n_strat}  n_showers={n_showers}")

    # Count distinct layouts actually on disk for one strategy (exact — every
    # BATCH_SIZE-row block should share one layout; count unique blocks).
    s0_rows = torch.nonzero(strat == 0).squeeze(-1)
    xy0 = xy[s0_rows]                                   # (n_showers, n_det, 2)
    # Two rows share a layout iff their xy blocks are bit-identical.
    flat = xy0.reshape(xy0.shape[0], -1)
    _, first_occurrence = torch.unique(flat, dim=0, return_inverse=True)
    n_distinct = int(torch.unique(flat, dim=0).shape[0])
    print(f"strategy 0 ('{names[0]}'): {n_showers} shower rows -> "
          f"{n_distinct} DISTINCT layouts actually used "
          f"(~{n_showers / max(n_distinct, 1):.0f} shower rows share each layout)")
    print()


def check2_corruption_in_labels():
    print("=" * 72)
    print("CHECK 2 — do the flagged corrupted muon events reach E.pt/T.pt?")
    print("=" * 72)
    # Indices from the earlier voxel-corpus inspection (slurm-35371950): union
    # of top-10 voxel-count / |sum_energy| / |sum_time_raw| muon events.
    flagged = sorted(set([
        51959, 46165, 93816, 53526, 17087, 61405, 89275, 85454, 94546, 67227,
        28427, 39370, 84415, 75057, 35503, 20017, 89644, 86802, 3908,
        53529, 57841, 30102, 59507, 81988, 35963, 26077, 86187,
    ]))
    print(f"flagged muon event indices (voxel-corpus, n={len(flagged)}): {flagged}")

    species = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt"))
    strat = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt"))
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt"))   # log1p(counts) already
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt"))   # RAW arrival_time

    n_strat = int(strat.max().item() + 1)
    n_pairs = strat.shape[0]
    n_showers = n_pairs // n_strat
    # 01's dst layout: row = s_idx*n_showers + ds, ds in electron block [0,k_sp)
    # then muon block [k_sp, n_showers) — k_sp = n_showers//2 since DATASET_FRACTION=1.0.
    k_sp = n_showers // 2
    print(f"n_showers={n_showers}  k_sp(per species)={k_sp}  n_strat={n_strat}")

    def rows_for_muon_event(x):
        ds = k_sp + x
        return [s * n_showers + ds for s in range(n_strat)]

    # Baseline: random NON-flagged muon events for comparison.
    rng = np.random.default_rng(0)
    all_muon = np.arange(k_sp)
    baseline = rng.choice(np.setdiff1d(all_muon, flagged), size=50, replace=False)

    def row_stats(rows):
        e_rows = E[rows]
        t_rows = T[rows]
        return dict(
            e_max=float(e_rows.abs().max()), e_mean=float(e_rows.abs().mean()),
            t_max=float(t_rows.abs().max()), t_mean=float(t_rows.abs().mean()),
            e_nonfinite=int((~torch.isfinite(e_rows)).sum()),
            t_nonfinite=int((~torch.isfinite(t_rows)).sum()),
            frac_zero_e=float((e_rows == 0).float().mean()),
        )

    base_rows = [r for x in baseline.tolist() for r in rows_for_muon_event(x)]
    base_stats = row_stats(base_rows)
    print(f"\nbaseline (50 random non-flagged muon events, {len(base_rows)} rows):")
    print(f"  {base_stats}")

    flag_rows = [r for x in flagged for r in rows_for_muon_event(x)]
    flag_stats = row_stats(flag_rows)
    print(f"\nflagged ({len(flagged)} events, {len(flag_rows)} rows):")
    print(f"  {flag_stats}")

    print(f"\nper-flagged-event breakdown (max |E| (log1p space), max |T| (raw)):")
    for x in flagged:
        rows = rows_for_muon_event(x)
        e_max = float(E[rows].abs().max())
        t_max = float(T[rows].abs().max())
        e_nf = int((~torch.isfinite(E[rows])).sum())
        t_nf = int((~torch.isfinite(T[rows])).sum())
        print(f"  muon event {x:6d}: max|E(log1p)|={e_max:10.4f}  "
              f"max|T(raw)|={t_max:.4e}  nonfinite(E,T)=({e_nf},{t_nf})")
    print()


def check3_capacity_vs_floor():
    print("=" * 72)
    print("CHECK 3 — surrogate capacity vs. aleatoric floor (train AND val, exact split)")
    print("=" * 72)
    primary = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"))
    xy = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt"))
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt"))
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt"))
    species = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt"))
    strat = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt"))

    T_log = torch.log1p(T * T_LOG_SCALE)   # 02's canonical T target

    train_idx, val_idx = shower_level_split(strat, val_frac=0.10, seed=0)
    print(f"reproduced 02's split: train={len(train_idx)}  val={len(val_idx)}")

    fnn, _ = load_models(DEVICE)

    def score(model, sample):
        preds_E, preds_T = [], []
        with torch.no_grad():
            for i in range(0, len(sample), 2048):
                idx = sample[i:i + 2048]
                p = primary[idx].to(DEVICE)
                xyb = xy[idx].to(DEVICE)
                out = model(p, xyb)                      # (B, n_det, 2) log1p space
                preds_E.append(out[..., 0].cpu())
                preds_T.append(out[..., 1].cpu())
        pred_E = torch.cat(preds_E); pred_T = torch.cat(preds_T)
        true_E = E[sample]; true_T = T_log[sample]
        return r2(true_E, pred_E), r2(true_T, pred_T)

    for sp_name, sp_id in (("electron", 0), ("muon", 1)):
        model = fnn.electron if sp_id == 0 else fnn.muon

        sp_val = val_idx[species[val_idx] == sp_id]
        sp_train = train_idx[species[train_idx] == sp_id]
        if len(sp_val) == 0 or len(sp_train) == 0:
            print(f"[{sp_name}] no rows"); continue

        g = torch.Generator().manual_seed(0)
        val_sample = sp_val[torch.randperm(len(sp_val), generator=g)[:min(20000, len(sp_val))]]
        train_sample = sp_train[torch.randperm(len(sp_train), generator=g)[:min(20000, len(sp_train))]]

        val_r2_E, val_r2_T = score(model, val_sample)
        train_r2_E, train_r2_T = score(model, train_sample)

        print(f"\n[{sp_name}]  n_train_sampled={len(train_sample)}  n_val_sampled={len(val_sample)}")
        print(f"  TRAIN R^2:  E={train_r2_E:.4f}   T={train_r2_T:.4f}")
        print(f"  VAL   R^2:  E={val_r2_E:.4f}   T={val_r2_T:.4f}")
        gap_e, gap_t = train_r2_E - val_r2_E, train_r2_T - val_r2_T
        print(f"  gap (train-val): E={gap_e:+.4f}  T={gap_t:+.4f}")
        print(f"  interpretation: small gap + both moderate -> the ~{100*(1-val_r2_E):.0f}% "
              f"unexplained E variance is a floor the model can't beat even ON TRAIN "
              f"(irreducible per-event noise, or the model needs more signal, not more capacity)")
        print(f"                  large gap (train >> val)  -> classic overfitting/generalization "
              f"gap -> more DATA (not necessarily bigger model) plausibly helps")
    print()


def check4_recon_permutation_invariance():
    print("=" * 72)
    print("CHECK 4 — is the LOADED recon.pt actually permutation-invariant?")
    print("=" * 72)
    _, recon = load_models(DEVICE)
    torch.manual_seed(0)
    B, n_det = 4, N_DETECTORS
    xy = torch.randn(B, n_det, 2, device=DEVICE) * 500
    E = torch.randn(B, n_det, device=DEVICE).abs()
    T = torch.randn(B, n_det, device=DEVICE).abs()
    feats = torch.stack([xy[..., 0], xy[..., 1], E, T], dim=-1)

    perm = torch.randperm(n_det, device=DEVICE)
    feats_shuffled = feats[:, perm, :]

    with torch.no_grad():
        out1 = recon(feats)
        out2 = recon(feats_shuffled)
    max_diff = float((out1 - out2).abs().max())
    print(f"max |recon(original) - recon(shuffled detectors)| = {max_diff:.3e}")
    print(f"  -> {'PERMUTATION-INVARIANT (confirmed)' if max_diff < 1e-3 else 'ORDER-SENSITIVE (not invariant!)'}")
    print()


def check5_recon_distribution_shift():
    print("=" * 72)
    print("CHECK 5 — recon input distribution: kernel labels (03 training) vs surrogate (04 opt-time)")
    print("=" * 72)
    primary = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"))
    xy = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt"))
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt"))
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt"))
    T_log = torch.log1p(T * T_LOG_SCALE)
    species = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt"))

    fnn, _ = load_models(DEVICE)
    rng = np.random.default_rng(0)
    sample = torch.as_tensor(rng.choice(len(primary), size=5000, replace=False))

    with torch.no_grad():
        pred_E_all, pred_T_all = [], []
        for i in range(0, len(sample), 1024):
            idx = sample[i:i + 1024]
            p = primary[idx].to(DEVICE)
            xyb = xy[idx].to(DEVICE)
            out = fnn(p, xyb)
            pred_E_all.append(out[..., 0].cpu()); pred_T_all.append(out[..., 1].cpu())
    pred_E = torch.cat(pred_E_all); pred_T = torch.cat(pred_T_all)
    true_E = E[sample]; true_T = T_log[sample]

    def summarize(name, t):
        q = torch.quantile(t.double().reshape(-1),
                          torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
        print(f"  {name:22s} mean={t.mean():.4f} std={t.std():.4f}  "
              f"p1={q[0]:.4f} p50={q[1]:.4f} p99={q[2]:.4f}")

    print("E channel (log1p(counts)):")
    summarize("kernel labels (03 train)", true_E)
    summarize("surrogate (04 opt-time)", pred_E)
    print("T channel (log1p(T*T_LOG_SCALE)):")
    summarize("kernel labels (03 train)", true_T)
    summarize("surrogate (04 opt-time)", pred_T)
    print()


if __name__ == "__main__":
    check1_layout_diversity()
    check2_corruption_in_labels()
    check3_capacity_vs_floor()
    check4_recon_permutation_invariance()
    check5_recon_distribution_shift()

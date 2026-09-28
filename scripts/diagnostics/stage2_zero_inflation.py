"""Measure the σ=10 target collapse that wrecked run 14's Stage 2 fit.

Run 14 (`14_sigma10_10pct`, SIGMA_SPATIAL=10) trains to val MSE 0.62/0.70/0.61
against run 13's 0.37/0.47/0.40 on the same loss (both ran
`--nll-warmup-frac 1.0`, so both numbers are plain z-scored MSE). The
target-vs-pred figure shows the mechanism: run 13's density ridge follows y=x,
run 14's is FLAT — the network predicts ~1-3 whatever the target is.

The hypothesis under test is that the labels collapsed onto a point mass at
zero, via two INDEPENDENT routes:

  E: `local_intensity` has no threshold; it sums e_p * exp(-d^2/2 sigma^2) over
     every point (showers/kernel.py). At d=100 m that factor is 0.135 at
     sigma=50 but 2e-22 at sigma=10, which underflows to hard zero in float32.
     At ~120 m detector spacing almost every off-core detector reads exactly 0.
     Intrinsic to wanting a 10 m response — it has to be MODELLED, not fixed.

  T: `HIT_DEPOSIT_MIN` is an ABSOLUTE cut on energy * kernel acceptance. The
     constant's own comment records that 1e-3 was swept to put T > 0 at ~35% of
     detectors *at sigma=50*. Cutting sigma to 10 shrank the acceptance ~25x and
     nothing rescaled the threshold. This one is a stale constant, not physics.

Four measurements, matching the four things the fix needs to know:

  (1) zero fractions per channel, run 14 vs run 13 — how bad, and how much of it
      is E versus T;
  (2) a HIT_DEPOSIT_MIN sweep at the CURRENT sigma — which threshold restores
      T > 0 at ~35% of detectors (the sigma^2 argument predicts ~4e-5);
  (3) label stats over hit entries only versus over everything, which is what
      `compute_normalization` does today — how badly the z-scoring is
      conditioned by the zeros;
  (4) run 14's own val MSE split into hit and non-hit entries. This is the one
      that decides whether a zero-inflated head is worth building: if the hits
      already fit well and the zeros carry the loss, a hit/no-hit head is
      justified; if the hits fit badly too, the problem is somewhere else.

Read-only. Loads existing corpora and checkpoints, writes nothing but stdout and
an optional JSON summary. Run it BEFORE the NLL removal lands — (4) loads a
4-output (mu+logvar) decoder that the mean-only refactor deletes.

    python scripts/diagnostics/stage2_zero_inflation.py --json out.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _V6)
from _pathfix import V6_ROOT  # noqa: F401,E402 — idempotent, registers v6 root

import numpy as np  # noqa: E402
import torch  # noqa: E402

import modules  # noqa: F401,E402 — package import; keeps modules on the path
from modules.constants import (  # noqa: E402
    SPECIES_NAMES, T_LOG_SCALE, N_DETECTORS, SIGMA_SPATIAL, HIT_DEPOSIT_MIN,
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    DUAL_SHOWER_CACHE_PATH, RUN_LOCATION,
)
from modules.surrogates import compute_normalization  # noqa: E402

# 02's filename starts with a digit, so it cannot be imported by name. Load it by
# path and reuse `shower_level_split` + its split constants rather than
# re-deriving them here: measurement (4) is only comparable to the training log
# if the val rows are the SAME val rows.
_spec02 = importlib.util.spec_from_file_location(
    "train02", os.path.join(_V6, "scripts", "02_train_fnn_deepsets.py"))
_train02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_train02)
shower_level_split = _train02.shower_level_split
VAL_FRAC = _train02.VAL_FRAC
SEED = _train02.SEED

RUNS_ROOT = os.path.dirname(RUN_LOCATION.rstrip("/"))
STAGE1_SUBDIR = "test_v6_run_01_northeast"
STAGE2_SUBDIR = "test_v6_run_02_recentered"

# Species id -> name. Step 1 writes species_ids.pt as an index into SPECIES_NAMES.
SPECIES_TAGS = list(enumerate(SPECIES_NAMES))


def _fmt_pct(x):
    return f"{100.0 * float(x):6.2f}%"


def _stats(x: torch.Tensor) -> dict:
    """mean/std/max plus the z-distance of the max under that mean/std.

    z_max is the point of the whole exercise: it says how far the tail sits from
    the bulk once the zeros have dragged the mean down and the std up. A tail at
    28 sigma is a target no MSE head can chase.
    """
    mu, sd, mx = float(x.mean()), float(x.std()), float(x.max())
    return dict(mean=mu, std=sd, max=mx,
                z_max=(mx - mu) / sd if sd > 0 else float("nan"))


# ── (1) + (3) label-side measurements ────────────────────────────────────────

def measure_labels(run_dir: str, label: str) -> dict:
    """Zero fractions and hits-only vs all-entries label stats, per species.

    E.pt is stored as log1p(E), so `E == 0` is exactly `raw E == 0` — log1p is
    strictly monotone with log1p(0) = 0. T.pt is stored RAW (02 applies the
    log1p(T * T_LOG_SCALE) transform at train time), so T == 0 is the kernel's
    "no point cleared HIT_DEPOSIT_MIN" sentinel directly.
    """
    d = os.path.join(run_dir, STAGE1_SUBDIR)
    if not os.path.isdir(d):
        print(f"[skip] {label}: no {STAGE1_SUBDIR} under {run_dir}")
        return {}

    print("=" * 78)
    print(f"({1}) + (3)  labels — {label}")
    print(f"          {d}")
    print("=" * 78)

    t0 = time.time()
    E = torch.load(os.path.join(d, "E.pt"), map_location="cpu").float()
    T = torch.load(os.path.join(d, "T.pt"), map_location="cpu").float()
    species_ids = torch.load(os.path.join(d, "species_ids.pt"), map_location="cpu").long()
    blob = torch.load(os.path.join(d, "blob_ids.pt"), map_location="cpu").bool()
    print(f"[load] {time.time()-t0:.1f}s  E={tuple(E.shape)}  "
          f"blobs={int(blob.sum())} ({_fmt_pct(blob.float().mean())})")

    out = {}
    for sp_val, sp_name in SPECIES_TAGS:
        idx = torch.nonzero((species_ids == sp_val) & ~blob).squeeze(-1)
        if idx.numel() == 0:
            continue
        Es, Ts = E[idx], T[idx]

        e_hit = Es > 0
        t_hit = Ts > 0
        # The constants.py comment asserts E > 0 is a strictly weaker condition
        # than T > 0 (E sums sub-threshold deposits that never clear the trigger),
        # and that ~40% of detectors sit in the E>0/T==0 band at sigma=50.
        # t_and_not_e should be identically zero; if it is not, that invariant
        # broke and the whole T story needs rethinking.
        e_and_not_t = (e_hit & ~t_hit).float().mean()
        t_and_not_e = (t_hit & ~e_hit).float().mean()

        # log-T space is what the trainer actually regresses (02 mutates the T
        # norm slots after this transform), so report T stats there.
        Tlog = torch.log1p(Ts * T_LOG_SCALE)

        rec = dict(
            n_rows=int(idx.numel()),
            frac_E_zero=float((~e_hit).float().mean()),
            frac_T_zero=float((~t_hit).float().mean()),
            frac_E_pos_T_zero=float(e_and_not_t),
            frac_T_pos_E_zero=float(t_and_not_e),
            E_all=_stats(Es), E_hits=_stats(Es[e_hit]) if e_hit.any() else None,
            T_all=_stats(Tlog), T_hits=_stats(Tlog[t_hit]) if t_hit.any() else None,
        )
        out[sp_name] = rec

        print(f"\n  [{sp_name}]  rows={rec['n_rows']:,}")
        print(f"    frac(E == 0)            {_fmt_pct(rec['frac_E_zero'])}")
        print(f"    frac(T == 0)            {_fmt_pct(rec['frac_T_zero'])}")
        print(f"    frac(E > 0 & T == 0)    {_fmt_pct(rec['frac_E_pos_T_zero'])}"
              "   <- sub-threshold deposit, no trigger")
        print(f"    frac(T > 0 & E == 0)    {_fmt_pct(rec['frac_T_pos_E_zero'])}"
              "   <- must be 0.00%; nonzero breaks the E-weaker-than-T invariant")
        for chan, key_all, key_hit in (("log1p(E)", "E_all", "E_hits"),
                                       ("log-T   ", "T_all", "T_hits")):
            a, h = rec[key_all], rec[key_hit]
            print(f"    {chan}  all : mean={a['mean']:8.4f} std={a['std']:7.4f} "
                  f"max={a['max']:8.3f}  max at {a['z_max']:6.1f}σ")
            if h:
                print(f"    {chan}  hits: mean={h['mean']:8.4f} std={h['std']:7.4f} "
                      f"max={h['max']:8.3f}  max at {h['z_max']:6.1f}σ")
    print()
    return out


# ── (2) HIT_DEPOSIT_MIN sweep at the current sigma ───────────────────────────

def sweep_hit_threshold(n_showers: int, batch: int, thresholds, seed: int) -> dict:
    """frac(T > 0) as a function of `hit_deposit_min`, at the CURRENT sigma.

    Calls `GetCounts_planeaware` once per threshold rather than reimplementing
    its `energy_kernel > thr` line. That is ~len(thresholds)x the kernel work on
    a deliberately small shower sample, and it is worth it: a local copy of the
    kernel math would be exactly the kind of drift that produces a confidently
    wrong recalibration. Layouts come from the real Step-1 strategies so the
    geometry matches the corpus.
    """
    from modules.data.dataset_builder import (_load_corpus_metadata, _load_clouds,
                                              _build_chunk_list)
    from modules.data import compute_labels_batch  # noqa: F401 — parity reference
    from modules.showers import GetCounts_planeaware
    from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS
    from modules.geometry import load_tr_mountain, SurfaceUpMap

    print("=" * 78)
    print(f"(2)  HIT_DEPOSIT_MIN sweep at SIGMA_SPATIAL={SIGMA_SPATIAL:g}")
    print(f"     current HIT_DEPOSIT_MIN = {HIT_DEPOSIT_MIN:g}"
          f"   sigma^2-scaled guess = {HIT_DEPOSIT_MIN * (SIGMA_SPATIAL / 50.0) ** 2:.3g}")
    print("=" * 78)

    device = torch.device("cpu")
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)
    print(f"[load] mountain + surface in {time.time()-t0:.1f}s")

    meta = _load_corpus_metadata(mountain, DUAL_SHOWER_CACHE_PATH,
                                 max_showers=n_showers)
    print(f"[meta] {meta.n_showers} showers ({meta.k_sp} per species block)")

    rng = np.random.default_rng(seed)
    thr = torch.tensor(sorted(thresholds), dtype=torch.float64)
    hit_counts = torch.zeros(len(thr), dtype=torch.float64)
    n_slots = 0.0
    # Parity check against the production path: compute_labels_batch uses the
    # module-level HIT_DEPOSIT_MIN default, so its frac(T>0) must equal this
    # sweep's value at that same threshold. If the two disagree the sweep is
    # measuring something other than what Step 1 labels.
    parity_hits, parity_slots = 0.0, 0.0

    t0 = time.time()
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        e_det, n_det_xy = fn(mountain, n_det=N_DETECTORS, rng=rng, **kwargs)
        e_det = e_det.float().to(device)
        n_det_xy = n_det_xy.float().to(device)
        # Mirror compute_labels_batch's frame mapping exactly: the kernel reads
        # North as its transverse axis and East as depth.
        up = surface(n_det_xy, e_det)
        z_cont = (EAST_ENTRY - e_det) / LAYER_EAST_DX
        dummy_flux = torch.tensor([0.0], device=device)

        # Chunks must come from _build_chunk_list, not from a flat range: the
        # corpus is species-major, so a chunk carries BOTH a file offset
        # (s_i * per_sp) and a kept-array offset (s_i * k_sp), and those differ
        # whenever max_showers subsamples. A hand-rolled ("diag", 0, 0, lo, hi)
        # silently reads the wrong species block's positions past the first.
        _, chunk_list = _build_chunk_list(meta.k_sp, meta.per_sp,
                                          load_chunk=batch, batch_size=batch)
        for chunk in chunk_list:
            clouds, _, _ = _load_clouds(DUAL_SHOWER_CACHE_PATH, meta, chunk,
                                        EAST_ENTRY, LAYER_EAST_DX)
            clouds = clouds.to(device)
            with torch.no_grad():
                for i, th in enumerate(thr):
                    _, T = GetCounts_planeaware(
                        clouds, n_det_xy, up, z_cont,
                        SmearN_fn=None, fluxB_e=dummy_flux,
                        TimeAverage_vectorized_fn=None,
                        sigma=SIGMA_SPATIAL, hit_deposit_min=float(th))
                    hit_counts[i] += float((T > 0).sum())
                    if abs(float(th) - HIT_DEPOSIT_MIN) < 1e-30:
                        parity_hits += float((T > 0).sum())
                        parity_slots += T.numel()
                n_slots += float(T.numel())
            del clouds
        print(f"  [{s_name}] done  ({time.time()-t0:.1f}s cumulative)")

    fracs = (hit_counts / max(n_slots, 1.0)).tolist()
    print(f"\n  slots sampled: {int(n_slots):,}   "
          f"(showers x detectors x {len(_STRATEGIES)} strategies)")
    print(f"\n  {'hit_deposit_min':>16s}   {'frac(T > 0)':>12s}    target ~35%")
    best, best_gap = None, 1e9
    for th, fr in zip(thr.tolist(), fracs):
        mark = "  <- current" if abs(th - HIT_DEPOSIT_MIN) < 1e-30 else ""
        print(f"  {th:16.3g}   {_fmt_pct(fr):>12s}{mark}")
        if abs(fr - 0.35) < best_gap:
            best, best_gap = th, abs(fr - 0.35)
    print(f"\n  closest to 35%: hit_deposit_min = {best:.3g}"
          f"  (frac = {_fmt_pct(fracs[thr.tolist().index(best)])})")
    if parity_slots:
        print(f"  parity: frac(T>0) at current threshold = "
              f"{_fmt_pct(parity_hits / parity_slots)}")
    print()
    return dict(sigma=SIGMA_SPATIAL, current=HIT_DEPOSIT_MIN,
                thresholds=thr.tolist(), fracs=fracs, recommended=best,
                n_slots=int(n_slots))


# ── (4) split run 14's val MSE into hit / non-hit ────────────────────────────

def _load_mean_rows(ckpt: dict, primary_dim: int, device):
    """Build the current mean-only surrogate from ANY Stage 2 checkpoint.

    Run 12/13/14 checkpoints carry a 4-output decoder (mu_E, mu_T, logvar_E,
    logvar_T); the current class has 2. Rather than resurrect the old class to
    read them, slice the final decoder Linear down to its first two output rows.
    That is exactly equivalent, not an approximation: the old `_forward_z` did
    `mu_z = out_n[..., :2]`, so rows 0-1 of that layer ARE the mean head and
    rows 2-3 fed only the deleted variance path. Every other layer is unchanged
    between the two versions.
    """
    from modules.surrogates import DeepSetsSurrogate
    cfg = ckpt.get("config", {})
    model = DeepSetsSurrogate(
        n_det=N_DETECTORS, primary_dim=primary_dim,
        hidden=int(cfg.get("hidden", 256)), context=int(cfg.get("context", 64)),
        n_enc=int(cfg.get("n_enc", 3)), n_dec=int(cfg.get("n_dec", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    sd = dict(ckpt["state_dict"])
    want = model.state_dict()
    n_sliced = 0
    for k, v in list(sd.items()):
        if k in want and v.shape != want[k].shape:
            # Only ever the decoder's final Linear, and only along dim 0.
            assert v.shape[0] > want[k].shape[0] and v.shape[1:] == want[k].shape[1:], \
                f"unexpected shape mismatch at {k}: ckpt {tuple(v.shape)} vs model {tuple(want[k].shape)}"
            sd[k] = v[:want[k].shape[0]]
            n_sliced += 1
    model.load_state_dict(sd)
    if n_sliced:
        print(f"[ckpt] legacy 4-output decoder: sliced {n_sliced} tensors "
              f"to their mean rows")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def split_val_mse(run_dir: str, species: str) -> dict:
    """Run 14's OWN checkpoint, its val MSE decomposed by hit mask.

    Reproduces train_species' pipeline exactly — per-species norm stats over
    non-blob rows, the log1p-T mutation of the T slots, `shower_level_split`,
    then the blob drop — so the total here reconciles with the training log's
    final val number. Anything else and the hit/non-hit split would be a
    decomposition of a different quantity.
    """
    from modules.surrogates import DeepSetsSurrogate

    d1 = os.path.join(run_dir, STAGE1_SUBDIR)
    ckpt_path = os.path.join(run_dir, STAGE2_SUBDIR, f"fnn_{species}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[skip] (4): no checkpoint at {ckpt_path}")
        return {}

    print("=" * 78)
    print(f"(4)  val MSE split by hit mask — {species} @ {os.path.basename(run_dir)}")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary = torch.load(os.path.join(d1, "primary.pt"), map_location="cpu").float()
    xy = torch.load(os.path.join(d1, "xy.pt"), map_location="cpu").float()
    E_all = torch.load(os.path.join(d1, "E.pt"), map_location="cpu").float()
    T_all = torch.load(os.path.join(d1, "T.pt"), map_location="cpu").float()
    strat = torch.load(os.path.join(d1, "strategy_ids.pt"), map_location="cpu").long()
    species_ids = torch.load(os.path.join(d1, "species_ids.pt"), map_location="cpu").long()
    blob = torch.load(os.path.join(d1, "blob_ids.pt"), map_location="cpu").bool()

    sp_val = SPECIES_NAMES.index(species)
    idx = torch.nonzero(species_ids == sp_val).squeeze(-1)
    primary, xy = primary[idx], xy[idx]
    E_all, T_all = E_all[idx], T_all[idx]
    strat, blob = strat[idx], blob[idx]

    keep = ~blob
    norm_stats = compute_normalization(primary[keep], xy[keep], E_all[keep], T_all[keep])
    # T hit mask must be taken BEFORE the log1p transform maps 0 -> 0; it is the
    # same set either way, but taking it first keeps the intent explicit.
    t_hit_raw = T_all > 0
    T_all = torch.log1p(T_all * T_LOG_SCALE)
    _n = T_all.shape[1]
    norm_stats["out_mean"][_n:] = float(T_all.mean().item())
    norm_stats["out_std"][_n:] = max(float(T_all.std().item()), 1e-6)

    _, val_idx = shower_level_split(strat, VAL_FRAC, SEED)
    val_idx = val_idx[keep[val_idx]]
    print(f"[split] val pairs={len(val_idx)}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[ckpt] epoch={ckpt.get('epoch','?')} val_total={ckpt.get('val_total','?')}")
    model = _load_mean_rows(ckpt, primary.shape[1], device)
    model.eval()

    out_mean = model.out_mean.to(device)
    out_std = model.out_std.to(device)

    # Accumulate sum of squared z-errors and counts separately for hit and
    # non-hit entries, per channel. Batched to keep the corpus off the GPU.
    acc = {c: {"hit_se": 0.0, "hit_n": 0.0, "zero_se": 0.0, "zero_n": 0.0}
           for c in ("E", "T")}
    BATCH = 4096
    with torch.no_grad():
        for lo in range(0, len(val_idx), BATCH):
            b = val_idx[lo:lo + BATCH]
            p_b = primary[b].to(device)
            xy_b = xy[b].to(device)
            E_b = E_all[b].to(device)
            T_b = T_all[b].to(device)
            # forward() returns (B, n_det, 2) INTERLEAVED per detector, so it
            # must be cat'd channel-wise, not reshaped — a reshape would pair
            # E targets against alternating E/T predictions. Same flattening as
            # 02's mse_mean_only.
            mean_raw = model(p_b, xy_b)                       # (B, n_det, 2)
            mean_flat = torch.cat([mean_raw[..., 0], mean_raw[..., 1]], dim=1)
            target = torch.cat([E_b, T_b], dim=1)
            mean_n = (mean_flat - out_mean) / out_std
            target_n = (target - out_mean) / out_std
            se = (mean_n - target_n) ** 2

            masks = {"E": (E_b > 0, se[:, :_n]),
                     "T": (t_hit_raw[b].to(device), se[:, _n:])}
            for chan, (hit, se_c) in masks.items():
                acc[chan]["hit_se"] += float(se_c[hit].sum())
                acc[chan]["hit_n"] += float(hit.sum())
                acc[chan]["zero_se"] += float(se_c[~hit].sum())
                acc[chan]["zero_n"] += float((~hit).sum())

    res = {}
    print(f"\n  {'chan':>4s} {'frac hits':>10s} {'MSE hits':>10s} "
          f"{'MSE zeros':>10s} {'MSE all':>10s} {'zeros share':>12s}")
    for chan, a in acc.items():
        n_tot = a["hit_n"] + a["zero_n"]
        se_tot = a["hit_se"] + a["zero_se"]
        mse_hit = a["hit_se"] / a["hit_n"] if a["hit_n"] else float("nan")
        mse_zero = a["zero_se"] / a["zero_n"] if a["zero_n"] else float("nan")
        mse_all = se_tot / n_tot if n_tot else float("nan")
        share = a["zero_se"] / se_tot if se_tot else float("nan")
        res[chan] = dict(frac_hits=a["hit_n"] / n_tot, mse_hits=mse_hit,
                         mse_zeros=mse_zero, mse_all=mse_all, zeros_loss_share=share)
        print(f"  {chan:>4s} {_fmt_pct(a['hit_n']/n_tot):>10s} {mse_hit:10.4f} "
              f"{mse_zero:10.4f} {mse_all:10.4f} {_fmt_pct(share):>12s}")
    total = 0.5 * (res["E"]["mse_all"] + res["T"]["mse_all"])
    res["total_mse"] = total
    print(f"\n  total = 0.5*(E + T) = {total:.4f}"
          f"   (training log's final val for cross-check)")
    print("\n  Reading: a LOW 'MSE hits' with a HIGH 'zeros share' means the head"
          "\n  already fits real signal and is being dragged by the point mass at"
          "\n  zero — the case FOR a hit/no-hit head. A high 'MSE hits' means the"
          "\n  zeros are not the whole story and Part C needs rethinking.\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=["14_sigma10_10pct",
                                                  "13_full_750k_no_sampling"],
                    help="run folder names under the shared runs root")
    ap.add_argument("--ckpt-run", default="14_sigma10_10pct",
                    help="run whose Stage 2 checkpoint measurement (4) loads")
    ap.add_argument("--ckpt-species", default="electron")
    ap.add_argument("--sweep-showers", type=int, default=600,
                    help="showers (across species blocks) for the threshold sweep")
    ap.add_argument("--sweep-batch", type=int, default=64)
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip (2); it is the only part that touches the shower cache")
    ap.add_argument("--skip-ckpt", action="store_true")
    ap.add_argument("--json", default="", help="also write the summary here")
    args = ap.parse_args()

    print("=" * 78)
    print("stage2_zero_inflation.py — why run 14's Stage 2 collapsed")
    print("=" * 78)
    print(f"runs root       : {RUNS_ROOT}")
    print(f"SIGMA_SPATIAL   : {SIGMA_SPATIAL:g}")
    print(f"HIT_DEPOSIT_MIN : {HIT_DEPOSIT_MIN:g}")
    print(f"T_LOG_SCALE     : {T_LOG_SCALE:g}\n")

    summary = {"sigma_spatial": SIGMA_SPATIAL,
               "hit_deposit_min": HIT_DEPOSIT_MIN, "labels": {}}

    for name in args.runs:
        summary["labels"][name] = measure_labels(os.path.join(RUNS_ROOT, name), name)

    if not args.skip_sweep:
        # Decades around the sigma^2-scaled prediction (1e-3 * (10/50)^2 = 4e-5),
        # plus the current value so the sweep is anchored to a known point.
        thresholds = [1e-7, 4e-7, 1e-6, 4e-6, 1e-5, 4e-5, 1e-4, 4e-4, 1e-3]
        summary["threshold_sweep"] = sweep_hit_threshold(
            args.sweep_showers, args.sweep_batch, thresholds, SEED)

    if not args.skip_ckpt:
        summary["val_mse_split"] = split_val_mse(
            os.path.join(RUNS_ROOT, args.ckpt_run), args.ckpt_species)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()

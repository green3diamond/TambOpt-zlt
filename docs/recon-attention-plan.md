# Attention recon (set transformer with pair bias) for run 18

## Context
The relative-time DeepSets recon stops improving at a median of ~2.45° and ~14–15% of showers within 1° (grid, 100 detectors). The 1° optimisations do not move past the grid either.

Timing is not the gap:
- every detector with ≥ 0.1 count already has an interpolated leading-edge time (`KERNEL_T_INTERP = True`; 100% of E > 0.1 slots are timed);
- the untimed "lit" slots are Gaussian tails below 0.001.

DeepSets pools each detector separately and only sees dt relative to a single reference detector. An attention encoder can relate every pair of hit detectors (Δt against Δx, Δy, i.e. the plane front).

Goal: build this as a new recon type, train it on the same run 18 zoo data with the same schedule as the relative-time DeepSets recon (plus wd 0.03), and compare.

Legend: 🟩 exists, reused · 🟨 exists, changed · 🟥 new

## Step 0: plane-front gate (cheap, decides whether to build)
**Script:** 🟥 `<run18>/diagnostics/zoo_compare/planefit_check.py`, a copy of the `lit_geometry.py` loop (same 112 layouts × 1,536 showers, `CachedKernel` labels).

**Per pair:** a weighted least-squares plane-front fit on the timed detectors (≥ 0.1 count).
- Unknowns t0, sx, sy: t_i = t0 + (x_i·sx + y_i·sy)/c, with weights log1p E.
- Direction: n_h = −c·(sx, sy), n_z from |n| = 1. Pick the sign of n_z for showers coming up the plane, as in `primary_to_physical_labels`. Fit only where at least 3 timed detectors are not collinear.

**Output:**
- the angle error against the truth next to the recon's, binned by the number of timed detectors (3, 4–5, 6–10, 11+);
- `planefit_check.json` and a plot.

**Decision:**
- If the fit beats the recon clearly where ≥ 6 detectors are timed, the recon leaves information unused: build the attention recon.
- If not, the floor comes from the data, so stop here and report.

**Run:** one GPU job, gpu_requeue + `gpu_health_gate`, about 5 min.

## Architecture (B batch, N = 100, d = 128, h = 4 heads, L = 3 blocks)
```
raw x (B, N, 4) = [x, y, log1p E, log1p T]                            🟩
├─ 🟩 A  reltime_features → (B, N, 9) → 🟩 normalise (in_mean/in_std)
│     └─ 🟨 B  token MLP _mlp(9, 256, 128, n_enc=4) → z⁰ (B, N, d)    (output width 128 is new)
├─ 🟥 C  pair_features → (B, N, N, 6)
│     └─ 🟥 D  pair MLP _mlp(6, 32, h, 2) → P (B, h, N, N)            (computed once, used by every block)
├─ 🟥 E  soft hit weight logw (B, N) = clamp(log σ((log1p E − log1p 0.1)/τ), −30), τ = 0.5
│        bias = (P + logw[:, None, None, :]) reshaped to (B·h, N, N)
├─ 🟥 F  3 × nn.TransformerEncoderLayer(d, h, 4d, dropout 0, gelu, norm_first, batch_first)(z, src_mask=bias)
├─ 🟨 G  masked maxmean: mean = Σ w z / Σ w (w = e^logw); max = amax(z + logw)  → (B, 2d)
├─ 🟨 H  context_proj Linear(2d → 128)                                (exists, input 2·256 → 2·128)
└─ 🟩 I  decoder _mlp(128, 256, 4, n_dec=4) → × out_std + out_mean
```

**Pair features [b, i, j], all j − i:**

| Feature | Definition |
|---|---|
| Δx, Δy | in km |
| dist | √(Δx² + Δy² + 1e-6), in km |
| Δt | (t_j − t_i) in µs × both_timed |
| both_timed | 1 if both detectors have a time |
| Δ log1p E | log1p E_j − log1p E_i |

**Properties:**
- Permutation-invariant: no positional encoding, and pooling is a mean and a max.
- Smooth in x and y, so `04_optimize_lbfgs_kernel.py` gradients still flow.
- Dark detectors can't be attended to or pooled.
- About 0.9M parameters (DeepSets: 431k).

## Expected code changes

### 1. `modules/surrogates/recon.py`
🟥 `pair_features`, 🟥 `soft_hit_logweight` and 🟥 the `SetTransformerRecon` class; reuse 🟩 `reltime_features`, 🟩 `RELTIME_FEATURES` and 🟩 `_mlp` (`modules/surrogates/blocks.py`).
```python
PAIR_FEATURES = 6


def pair_features(x: torch.Tensor) -> torch.Tensor:
    """(B, n, 4) raw [x, y, log1p E, log1p(t * 1e6)] -> (B, n, n, 6) pair features, [b, i, j] = j relative to i:
    dx, dy, dist [km], dt [us] (0 unless both timed), both_timed, d log1p E."""
    pos, E, T = x[..., :2], x[..., 2], x[..., 3]
    timed = (T > 0).to(x.dtype)
    t_us = torch.expm1(T.clamp(min=0.0))
    d = (pos[:, None, :, :] - pos[:, :, None, :]) / 1000.0
    dist = (d.pow(2).sum(-1) + 1e-6).sqrt()
    both = timed[:, :, None] * timed[:, None, :]
    dt = (t_us[:, None, :] - t_us[:, :, None]) * both
    dE = E[:, None, :] - E[:, :, None]
    return torch.stack([d[..., 0], d[..., 1], dist, dt, both, dE], dim=-1)


def soft_hit_logweight(E: torch.Tensor, e_min: float = 0.1, tau: float = 0.5) -> torch.Tensor:
    """log of a smooth 'hit' weight from log1p E: ~0 above e_min counts, -> -30 below (finite, so an
    event with nothing hit still gets a uniform softmax)."""
    return F.logsigmoid((E - math.log1p(e_min)) / tau).clamp(min=-30.0)


class SetTransformerRecon(nn.Module):
    """DeepSetsRecon with attention between detectors before the pool. ..."""

    def __init__(self, n_det=N_DETECTORS, input_features=4, output_dim=4, hidden=256, context=128,
                 n_enc=4, n_dec=4, d_model=128, n_heads=4, n_blocks=3, pair_hidden=32,
                 hit_e_min=0.1, hit_tau=0.5, input_mode="reltime"):
        super().__init__()
        self.n_det, self.input_features, self.output_dim = n_det, input_features, output_dim
        self.n_heads, self.hit_e_min, self.hit_tau, self.input_mode = n_heads, hit_e_min, hit_tau, input_mode
        enc_in = RELTIME_FEATURES if input_mode == "reltime" else input_features
        self.encoder = _mlp(enc_in, hidden, d_model, n_enc)
        self.pair_mlp = _mlp(PAIR_FEATURES, pair_hidden, n_heads, 2)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(n_blocks))
        self.context_proj = nn.Linear(2 * d_model, context)
        self.decoder = _mlp(context, hidden, output_dim, n_dec)
        # same in_/out_ mean/std buffers + set_normalization as DeepSetsRecon

    def forward(self, x):
        B, N, _ = x.shape
        logw = soft_hit_logweight(x[..., 2], self.hit_e_min, self.hit_tau)          # (B, N)
        f = reltime_features(x) if self.input_mode == "reltime" else x
        z = self.encoder((f - self.in_mean) / self.in_std)                          # (B, N, d)
        bias = self.pair_mlp(pair_features(x)).permute(0, 3, 1, 2)                  # (B, h, N, N)
        bias = (bias + logw[:, None, None, :]).reshape(B * self.n_heads, N, N)
        for blk in self.blocks:
            z = blk(z, src_mask=bias)
        w = logw.exp()[..., None]
        pooled = torch.cat([(w * z).sum(1) / w.sum(1).clamp(min=1e-6),
                            (z + logw[..., None]).amax(1)], dim=-1)                 # (B, 2d)
        out_n = self.decoder(self.context_proj(pooled))
        return out_n * self.out_std + self.out_mean
```
**Other changes in this file:**
- 🟨 `build_recon_from_ckpt`: add an `elif mtype == "settf":` branch that builds `SetTransformerRecon` from the config keys (d_model, n_heads, n_blocks, pair_hidden, hit_e_min, hit_tau, plus the shared ones). Loading, normalisation and freezing stay as they are.
- 🟨 `modules/surrogates/__init__.py`: export `SetTransformerRecon`.

### 2. `scripts/03_train_recon_deepsets.py`
- 🟨 New constants next to the architecture block: `MODEL = "deepsets"`, `D_MODEL = 128`, `N_HEADS = 4`, `N_BLOCKS = 3`, `PAIR_HIDDEN = 32`.
- 🟨 New flags: `--model {deepsets,settf}` and `--n-blocks`.
- 🟨 Construction (~line 589): `recon = (SetTransformerRecon(..., d_model=D_MODEL, n_heads=N_HEADS, n_blocks=N_BLOCKS, pair_hidden=PAIR_HIDDEN, input_mode=INPUT_MODE) if MODEL == "settf" else DeepSetsRecon(...))`.
- 🟨 `_save_ckpt` config (~line 357): `model_type=MODEL`, plus the settf keys when MODEL is settf. The `[optim]` and `[model]` log lines print MODEL.
- 🟩 Unchanged:
  - the reltime normalisation stats (the encoder input is the same 9 features);
  - the data loading;
  - AdamW with wd, the patience, L-BFGS and checkpoint resume;
  - the curves.

### 3. 🟨 `slurm/run18_pipeline.sh`
New mode `recon_settf`:
```
run_once 03_recon_settf env TAMBOPT_DATASET_TAG=zoo python -u scripts/03_train_recon_deepsets.py \
    --recon-input kernel --adam-patience 50 --input-mode reltime --weight-decay 0.03 \
    --model settf --out-suffix zoo_settf
```
It writes `<run18>/test_v6_run_03_recentered_deepsets_zoo_settf/`.

### 4. 🟩 Callers: no change
`load_models` / `build_recon_from_ckpt` dispatch on `model_type`. Scoring (`score_pad.py`, `lit_geometry.py`, `detector_count_grid.py`), `04_optimize_lbfgs_kernel.py` and the PDF scripts all take a `--recon-dir`.

### 5. 🟥 `tests/test_settf_recon.py`
- **Permutation invariance:** a random detector permutation gives the same output (atol 1e-5).
- **Dark slots:** extra dark slots (E = 0, T = 0) change the output by less than 1e-3.
- **Gradients:** the gradient with respect to x, y is finite and nonzero.
- **Checkpoint round trip:** `_save_ckpt` dict, then `build_recon_from_ckpt`, gives an identical output.

## Order and cost
1. Step 0 gate (5 min GPU). Report the result before building anything.
2. Code + tests on a new branch `run18-settf-recon` (from `run18-reltime-recon`), in the main checkout; no jobs are running from it.
3. Smoke: `--epochs 1 --lbfgs-iters 0 --out-suffix settf_smoke` on gpu_requeue. Check the parameter count (~0.9M), the s/epoch and the memory.
4. Full training: `sbatch slurm/run18_pipeline.sh recon_settf`. The DeepSets zoo recon took ~15.6 h, so expect 1.5–3× that.
5. Score with `run_snapshot.sh`-style scoring (all 4 test sets, thetas 1 / 2.5 / 5, detcount, lit_resolution). Then `compare_tags.py reltime_lbfgs25 settf`, and rerun `opt1deg/run.sh` with the settf recon added.
6. New PDF section "Attention recon" in `figures/make_zoo_recon_pdf.py`, before the weight-decay chapter. Commit the script copies.

## Verification
- **Step 0:** the plane fit with exact times on a 100-detector grid should give a sensible error (a few degrees or better) when ≥ 6 detectors are timed. Pair counts match `lit_geometry` (172,032).
- **Unit tests:** `pytest tests/test_settf_recon.py`, run locally with `multiproc_env`.
- **Smoke run:** the loss falls over the first epoch, and the checkpoint loads through `load_models(recon_dir=...)`.
- **Final:**
  - settf val loss compared with the reltime DeepSets (0.0782) and wd 0.03 (0.0769) recons;
  - test layouts within 1 / 2.5 / 5° and the median angle error against `reltime_lbfgs25`;
  - success = clearly above the 1,536-shower noise (about ±0.9 pts at 1°) on grid_n100 and the per-family medians.

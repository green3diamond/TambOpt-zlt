# Why run 14 (sigma=10 m) collapses in Stage 2

Session of 2026-09-28, branch `stage2-mean-only`. Run 14
(`14_sigma10_10pct`, `SIGMA_SPATIAL = 10`, `DATASET_FRACTION = 0.10`) trains to a
far worse Stage 2 fit than run 13. This doc records what was measured, which
hypotheses survived, and — importantly — which one did not.

All numbers come from `scripts/diagnostics/stage2_zero_inflation.py`
(job **49009683**, `slurm_logs/slurm-49009683-stage2_zero_diag.out`, summary JSON
`slurm_logs/stage2_zero_diag_49009683.json`). Electron unless stated.

## TL;DR

- Run 14's val MSE is **0.62 / 0.70 / 0.61** (e/mu/photon) against run 13's
  **0.37 / 0.47 / 0.40**. Normalised MSE of 1.0 = predicting the mean, so run 13
  explains ~63% of target variance and run 14 only ~38%.
- The labels collapsed onto a point mass at zero: at sigma=10, **78.5% of E entries
  and 95.2% of T entries are exactly 0**, against 21.5% / 62.0% at sigma=50.
- Root cause is **Gaussian underflow, not a stale trigger constant**.
  `exp(-d^2 / 2 sigma^2)` is 0.135 at d=100 m for sigma=50 but 2e-22 for sigma=10, which
  underflows to hard zero in float32. At ~120 m detector spacing, most detectors
  read exactly 0.
- **Rescaling `HIT_DEPOSIT_MIN` does not work.** Lowering it four decades moves
  the hit rate 5.87% -> 8.89%, against the ~35% it was calibrated for. A threshold
  cannot readmit a value that already underflowed to 0.0.
- **A zero-inflated (hit/no-hit) head is NOT the fix for E**, contrary to the
  initial hypothesis. The model already predicts the zeros almost perfectly and
  fails on the *hits*: zeros carry only **4.5%** of E's loss. It would help T a
  lot (T's hits are mode-collapsed) but leaves E's real problem untouched.

## Measurement 1 — zero fractions

`E.pt` is stored as `log1p(E)`, so `E == 0` is exactly `raw E == 0`. `T.pt` is raw,
so `T == 0` is the kernel's "no point cleared `HIT_DEPOSIT_MIN`" sentinel.

| run | species | E==0 | T==0 | T>0 | E>0 & T==0 |
|---|---|---|---|---|---|
| 13 (sigma=50) | electron | 21.45% | 61.95% | **38.05%** | 40.50% |
| 13 | muon | 25.45% | 66.24% | 33.76% | 40.78% |
| 13 | photon | 24.02% | 63.84% | 36.16% | 39.82% |
| 14 (sigma=10) | electron | 78.50% | 95.19% | **4.81%** | 16.68% |
| 14 | muon | 75.06% | 93.43% | 6.57% | 18.36% |
| 14 | photon | 78.48% | 94.74% | 5.26% | 16.26% |

Two sanity checks passed, which is why these numbers can be trusted:

- `frac(T>0 & E==0) = 0.00%` in every cell — the "E>0 is strictly weaker than
  T>0" invariant asserted at `modules/constants.py:59-61` holds.
- Run 13's `E>0 & T==0` of **40.50%** reproduces that same comment's documented
  "~40% of detectors have E > 0 with no single detected point" at sigma=50.

Run 13's T>0 rate of 38% is also exactly the ~35% `HIT_DEPOSIT_MIN = 1e-3` was
swept for. The constant was correctly calibrated; sigma=10 is what broke it.

## Measurement 2 — `HIT_DEPOSIT_MIN` cannot be rescaled back

`HIT_DEPOSIT_MIN` is an absolute cut on `energy * kernel_acceptance`
(`modules/showers/kernel.py:104`). Since sigma=10 shrank the acceptance ~25x, the
obvious fix was to scale the threshold by sigma^2 (1e-3 -> 4e-5). Measured at the
current sigma over 300,000 detector slots (600 showers x 5 layout strategies):

| `hit_deposit_min` | frac(T > 0) |
|---|---|
| 1e-7 | **8.89%** |
| 1e-6 | 8.21% |
| 4e-5 (the sigma^2-scaled guess) | 7.06% |
| 1e-4 | 6.73% |
| 1e-3 (current) | **5.87%** |

Four orders of magnitude buys three percentage points and saturates nowhere near
38%. **This hypothesis is dead.** The deposits are not sitting just below the
threshold — they have underflowed to exactly 0.0 in float32, and no positive
threshold admits them.

This also means E and T share one root cause. E never used the threshold at all
(`local_intensity` is an unthresholded sum), so its zeros were always underflow;
measurement 2 shows T's are too.

## Measurement 3 — the zeros wreck the z-scoring

`compute_normalization` (`scripts/01_build_dataset_northeast.py:130`) computes
mean/std over **all** entries including the zeros. Electron:

| | all entries | hits only |
|---|---|---|
| run 14 log1p(E) | mean 0.1053, std 0.7899 | mean 0.4897, std 1.6476 |
| run 14 log-T | mean 0.1216, std 0.5518 | mean 2.5273, std 0.4985 |

The log-T "all" statistics are essentially a description of the zeros. And the
z-scored tail runs to **26.4 sigma** for run 14 E (vs 6.8 sigma for run 13); restricted to
hits it is 12.4 sigma. The target handed to an MSE head is badly conditioned.

## Measurement 4 — the decisive one, and it refutes the hit-head fix

Run 14's own `fnn_electron.pt`, its val MSE decomposed by hit mask. Reconstructed
total is **0.6217**, matching the checkpoint's stored `val_total` 0.62165 — so
this is a decomposition of exactly the number in the training log.

| chan | frac hits | MSE on hits | MSE on zeros | MSE all | zeros' share of loss |
|---|---|---|---|---|---|
| E | 21.28% | **3.2763** | 0.0416 | 0.7299 | **4.49%** |
| T | 4.73% | **6.8388** | 0.1996 | 0.5134 | 37.03% |

The working hypothesis had been "the head fits the hits and is dragged to the
mode by the point mass at zero." **The measurement says the opposite.** The model
predicts the zeros nearly perfectly (E: 0.0416) and fails on the hits (E: 3.2763,
i.e. over 3x *worse* than predicting the global mean).

The two channels fail in different ways. Expressing hit variance in the same
global-z units:

| chan | MSE on hits | variance of hits | variance explained |
|---|---|---|---|
| E | 3.2763 | 4.3504 | **24.7%** |
| T | 6.8388 | 0.8160 | **-738%** |

- **E is a genuine regression failure.** There is real signal on lit detectors and
  the model captures a quarter of it. Gating the zeros away changes E's total from
  0.7299 to 0.6971 — nothing. A hit head does not address this.
- **T is mode collapse.** MSE on hits is 8.4x the variance of the hits themselves:
  the model predicts "dark" on lit detectors. A perfect hit gate plus predicting
  only the hit-conditional mean would take T from 0.5134 to **0.0386**.

Best case for a hit head — perfect classifier, E unimproved — is a total of
~0.37. That matches run 13's number but leaves E's 24.7% untouched.

## What this means for the fix

The original plan (rescale `HIT_DEPOSIT_MIN`, renormalise on hits, add a
zero-inflated head) is **not** supported by the data. Measurement 2 kills the
first part and measurement 4 kills the main justification for the third.

The honest reading is that at sigma=10 with ~120 m spacing the array does not
geometrically sample the showers: most detectors are dark, and on the minority
that fire, the response is so sharp a function of position that the surrogate
captures a quarter of its variance. That is a sampling-geometry problem, not an
architecture one.

Three candidate directions, none yet chosen:

1. **Hit head for T only** — well evidenced, worth ~0.28 of the 0.62 total,
   explicitly does nothing for E.
2. **Revisit sigma or detector spacing** — addresses the actual cause. sigma=10 may
   simply not be learnable at 120 m spacing.
3. **Investigate E's hit regression directly** — why only 24.7%? Candidates: the
   26 sigma tail, the conditioning of `log1p(E)` at sigma=10, or an irreducible
   aleatoric floor that has risen with the sharper kernel.

## Confounds and caveats

- Run 14 is `DATASET_FRACTION = 0.10`, run 13 is the full corpus, so the two
  differ in statistics as well as in sigma. Run 14's train/val gap is 0.038-0.065
  versus run 13's ~0.01, so some of the difference is overfitting — but the gap
  is small next to the 0.37 -> 0.62 move, and measurement 4 locates the failure
  on the hits regardless of sample size.
- Both runs' val MSE numbers are comparable because both trained with
  `--nll-warmup-frac 1.0` (plain z-scored MSE). Run 12's negative values are full
  NLL and must **not** be compared against them.
- Measurement 2 samples 600 showers across 5 strategies. An earlier 12-shower
  smoke run gave the same saturation with the absolute rates ~3 points higher;
  the conclusion is unchanged at 50x the statistics.
- Measurement 4 loads a pre-`042b482` 4-output (mu + logvar) checkpoint by slicing
  the final decoder `Linear` to its first two rows. This is exact, not an
  approximation: the old `_forward_z` returned `mu_z = out_n[..., :2]`, so those
  rows *are* the mean head.

## Related

- `042b482` — Stage 2 reduced to mean-only regression; the NLL/variance head was
  never trained in any surviving run and has been removed. Independent of the
  findings here.
- `modules/constants.py` `SIGMA_SPATIAL`, `HIT_DEPOSIT_MIN` — both carry comments
  that should be read alongside this doc.
- `docs/THEORY.md` 10.4 — the measured aleatoric floor at sigma=50 (E 0.306,
  T 0.414 in z-MSE units), the natural comparison point for direction 3 above.

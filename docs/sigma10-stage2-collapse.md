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
  **Confirmed by the bias test (Measurement 4 result):** E's hit error is 0.4%
  bias, T's is 78.5% bias.
- **Interpolating between macro-particles (Measurement 5)** helps E modestly
  (explainable fraction 35% → 47%) and not T. Most of the sigma=10 scatter is
  real shower structure at the scale of one detector, not generator sampling.

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

### Measurement 4 — interpretation notes (review discussion, 2026-09-28)

A later read-through found that the conclusion above is stronger than the
measurement supports. Summary of that discussion:

**How the numbers fit together.** All MSEs are z-scored over all detector slots
(1.0 = predict the global mean everywhere). The total is a weighted average of
the two groups:

```
MSE_all = frac_hits · MSE_hits + frac_zeros · MSE_zeros
E:  0.2128 · 3.2763 + 0.7872 · 0.0416 = 0.6972 + 0.0327 = 0.7299   (zeros' share 4.5%)
T:  0.0473 · 6.8388 + 0.9527 · 0.1996 = 0.3235 + 0.1902 = 0.5137   (zeros' share 37%)
val_total = (0.7299 + 0.5134) / 2 = 0.6217
```

**Correction: "over 3x worse than predicting the global mean" is wrong.** It
compares a hits-only MSE (3.28) against 1.0, which is the baseline for *all*
slots. On the hits alone, predicting the global mean scores the hit variance
plus the squared offset of the hit mean: E 4.35 + 0.49² ≈ **4.59**, T 0.82 +
4.36² ≈ **19.8**. On E the model beats the global mean (3.28 vs 4.59). The
correct statement is the variance-explained table: 24.7% for E, −738% for T,
relative to predicting the hit mean.

**The zeros can hurt the hits without showing up in the zeros' own error.** An
MSE-trained head predicts the mixture mean at each input:

```
trained on hits + zeros:  ŷ(x) = P(hit|x) · E[y | hit, x]   (pulled toward 0 where hits are uncertain)
trained on hits only:     ŷ(x) = E[y | hit, x]
```

At sigma=10 many slots (edge-of-footprint detectors) are lit only sometimes for
the same (q, xy). On those, predictions are biased low whenever the detector *is*
lit, and that cost is booked under "MSE on hits", not under zeros. So "zeros
carry 4.5% of the loss" does **not** rule out the zeros damaging E. The "E
unimproved" assumption in the best case above is unsupported. T's −738% is the
extreme form of the same effect (only 4.7% of T slots are hits).

It is the *target* that matters, not the *count* of zeros: at convergence the
zeros' residuals are tiny, so they barely push on the gradient, and reweighting
alone would not help. Training the hit regressor on hits only changes its target.

**What separating can and cannot fix:**

```
MSE_hits = bias from the mixture target        -> removed by a hits-only head
         + noise in y given (hit, x)            -> aleatoric; no model removes it
         + approximation error
```

**Caveats for a hurdle (classifier × hits-only regressor) model:**

- The layout optimizer still needs the expected reading, `P(hit|x) × hit-value(x)`.
  A **soft** product reproduces the joint mean at worst and adds an accurate hit
  value plus an explicit hit probability. A **hard** gate costs a full hit-sized
  error on every misclassification. At sigma=10 hit vs dark is partly random given
  (q, xy), so the "perfect classifier" best case is not reachable. This is the
  likely reason hard masks and BCE gates regressed total MSE earlier
  (THEORY.md §10.1).

**Decisive test, on the existing checkpoint (no retraining):**

1. Bias on hits: `mean(y − ŷ)` over hit slots. Shrinkage predicts it is
   significantly positive, with bias² a large share of 3.28 (E) / 6.84 (T).
2. Bin hit slots by ŷ and compare mean ŷ to mean y per bin. Shrinkage predicts
   ŷ < y, worst where ŷ is small.
3. If 1–2 point to shrinkage, fine-tune the decoder on hit slots only and
   re-measure the hit MSE.

If bias dominates, a hits-only head should help E as well as T and "a
zero-inflated head is not the fix for E" should be revised. If the hits are
unbiased but scattered, the E error is mostly aleatoric and the original
conclusion stands.

**Result of test 1 (production checkpoint, same val rows; from the
`stage2_hurdle_check.py` smoke run, which reproduced this section's table
exactly):**

| chan | MSE on hits | bias on hits `mean(y − ŷ)` | bias² share of hit MSE |
|---|---|---|---|
| E | 3.2763 | 0.115 | **0.4%** |
| T | 6.8388 | 2.316 | **78.5%** |

- **E:** hits are essentially unbiased, so E's hit error is scatter and a
  hits-only head cannot remove it. The original conclusion stands for E.
- **T:** most of the hit error is a downward bias: lit arrival times are pulled
  toward "dark". This is the shrinkage a hits-only head removes. Even a 1-epoch
  hurdle head on 2% of the data brought T's hit MSE from 6.84 to 0.55.

The full 15-epoch baseline-vs-hurdle comparison (`stage2_hurdle_check.py`,
branch `stage2-hurdle-check`) was cancelled before it ran.

## Measurement 5 — interpolating between macro-particles (2026-09-28)

AllShowers points are macro-particles (each stands for many real particles), so
at sigma=10 part of a detector's hit-or-miss reading could be the generator's
point sampling rather than physics. Branch `sigma10-interp-kernel` adds an
opt-in kernel (`compute_labels_batch(interp_k=k)`): each point's Gaussian widens
to its k-th nearest in-plane neighbour distance h (floored at sigma) and is
area-normalised, `(sigma/h)² · exp(−d²/2h²)`. It equals the fixed kernel where
points are dense and conserves each point's deposit
(`tests/test_interp_kernel.py`, 5/5).

**Point spacing and labels** (job 49027230, 256 held-out events × 7 strategies,
`slurm_logs/interp_kernel_check.json`):

| | electron | muon | photon |
|---|---|---|---|
| Median in-plane r_8 of points near the array | 6.3 m | 4.5 m | 4.7 m |
| Points near the array with r_8 > sigma | 33.5% | 28.2% | 23.4% |
| E == 0: raw → k = 8 | 78.0% → 7.9% | 74.9% → 7.7% | 77.2% → 7.2% |
| T == 0: raw → k = 8 | 94.9% → 52.6% | 93.3% → 54.0% | 94.3% → 48.6% |
| Deposit collected by the layouts, k = 8 / raw | 0.62 | 1.06 | 1.09 |

**Aleatoric floor** (jobs 49027231 raw / 49027232 k = 8; 64 primaries × 32
showers each, same seed). Normalised by the generated labels' own variance,
the only fair comparison: the corpus-std numbers for k = 8 use raw-kernel
labels as the denominator and are not meaningful.

| | fixed sigma = 10 | interpolated, k = 8 |
|---|---|---|
| E floor (max R²) | 0.650 (35%) | **0.529 (47%)** |
| T floor (max R²) | 0.509 (49%) | 0.566 (43%) |
| total | 0.579 | 0.548 |
| (primary, strategy, detector) groups that fire | 4% | 59% |

Reading:

- **Near the array, most points are already denser than sigma.** Median k-NN
  spacing is 5–6 m; only a quarter to a third of points get widened. The
  dominant cause of the zeros is detectors outside the dense footprint, not
  gaps between neighbouring points.
- **The E == 0 drop is mostly tiny values.** Widened isolated points put small,
  non-underflowing tails on far detectors. T, which needs a deposit above
  `HIT_DEPOSIT_MIN`, is the meaningful count: detected slots rise from ~5% to
  ~47%.
- **E improves, T does not.** The explainable fraction of E rises from 35% to
  47% (floor −19%), so part of E's scatter was point sampling. T's floor gets
  slightly worse: a widened point assigns its own arrival time to detectors
  tens of metres away, where the true front arrives earlier or later. An
  interpolated T would need the time front interpolated too, not the leading
  edge of a smeared point.
- The electron collected deposit falls 38%. Raw electron labels were partly
  lucky near-hits of sparse points, which the area-normalised kernel spreads
  out. That is consistent with sampling noise inflating raw electron readings.

**Verdict.** Interpolation is a real but modest gain for E and none for T.
Most of sigma=10's scatter is not generator sampling. It is the shower's own
structure at the scale of one detector, which no kernel choice removes. Worth
keeping as an option for E; not by itself a fix for the sigma=10 collapse.

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

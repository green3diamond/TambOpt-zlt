# Stage 3: analytic validation + targeted improvements

## Done

- **B0** — blob-row filter + `--out-suffix` flag, `scripts/03_train_recon_deepsets.py` (`5a792d8`).
- **Part A** — `plots/training/04_analytic_recon.py`, run at scale, verdict below (`9c930d6`, `78b39ad`).
- **Kernel-in-the-loop Stage 4** — `scripts/04_optimize_lbfgs_kernel.py`, added beyond this
  doc's original scope at the user's request (`5bde42d`, `f7fc35a`, `f14e7da`). Differentiability
  verified on synthetic data; the optimization loop itself has not been run on the real corpus.

## Left to do

1. **Re-baseline `recon_regimes.py`** — B0 changed which rows recon trains/validates on; no run
   since has produced the corrected kernel-regime R². Every item below should be measured
   against that number, not the pre-B0 archived 0.876/0.880/0.896/0.479.
2. **B1** — explicit hit mask + masked pooling. Best-evidenced, do first.
3. **B2** — give recon the Up/z_cont geometry it's missing.
4. **B3, B5, B6** — cheap follow-ons (energy loss reweighting, unit-norm direction, per-channel
   loss weighting).
5. **B4** — decay-vertex auxiliary output. Downgraded by A2 (see Results); do last if at all.
6. **Run the kernel-in-the-loop optimizer** — `--time-one-step` first, then a real run.

---

## Results (Part A, `plots/training/04_analytic_recon.py`)

Stable across two sample sizes (500 and 20,000 events, matching `recon_regimes.py`'s own
scale) — same signs and magnitudes at both, so this is a settled result, not sampling noise.

| | 500 events | 20,000 events | recon (kernel-regime) |
|---|---|---|---|
| A0 median R² (East only / East+transverse) | 0.086 / 0.30 | 0.083 / 0.347 | — |
| A1 dir_x/y/z pearson r | 0.23 / 0.29 / 0.04 | 0.24 / 0.27 / 0.03 | 0.936 / 0.938 / 0.947 |
| A2 median dist. error vs true median | 2520 m / 2840 m | 2578 m / 2914 m | — |
| A3 log_e pearson r / RMSE (dex) | 0.064 / 0.452 | 0.037 / 0.463 | — / 0.341 |
| A4 pearson r: N_lit vs log₁₀E / axis-dist | −0.216 / 0.017 | −0.154 / −0.008 | — |

**Verdict:** the naive plane-front + LDF analytic reconstruction is consistently worse than
recon on both direction and energy — this is not an easy win recon is leaving on the table. It
does **not** weaken B1 (a different, independent line of evidence), but it **downgrades B4**:
A2's explicit geometric fit for the decay vertex has essentially no localization power (median
error ≈ 88% of the true distance), so there's no strong prior that an auxiliary recon head would
recover that information either.

One clear positive: A0 shows real transverse timing structure beyond the East depth-ramp
(+0.26 median R² gain), so recon's use of the T channel isn't just picking up a trivial depth
correlation.

Three real bugs were found and fixed while getting these numbers (all in `78b39ad`): an
unnormalized conditioning check that rejected 100% of events; numerically unstable uncentered
positions in the plane solve (R² ≈ −1e15); and an aggregate-R² blow-up from a handful of
near-zero-T-variance outlier events, fixed by excluding them and reporting median alongside mean.

---

## Background

Stage 3 trains a DeepSets reconstructor `(East, North, E, T)×100 → (n̂_x, n̂_y, n̂_z, log̃E)`.
Archived kernel-regime quality: dir_x/y/z R² **0.876 / 0.880 / 0.896**, log_e R² **0.479**.
Nothing in the pipeline had ever been compared against a reconstruction-independent
reference — `true_utility.py`'s GENUINE/ARTIFACT verdict included — hence Part A.

**A validated analytic reconstruction already existed**, just not in this pipeline:
`/n/home05/zdimitrov/tambo/tambo_dir_reco/` (`TamboDirReco.jl`, own git repo). `src/reco.jl`'s
`plane_chisq_per_hit` (`:22-40`) fits `(θ, ϕ, t0)` from hit times; `timing_llh_per_hit` (`:42`)
adds quadratic front curvature `pred_delay = a·r²`; `Q_ldf`/`ldf_llh_per_hit` (`:68-74`) give
energy from a lateral distribution, including an un-hit-module term Stage 3 has no analogue
for. Part A ports these three ideas to Python/torch (not a Julia bridge — this pipeline is
100% Python, and the plane-wave model is exactly linear here, needing no Minuit dependency).

**The decay vertex is already a label.** `primary.pt` cols 5-7 are `(rel_E, rel_N, rel_U)`, the
ENU decay vertex minus array centre (`modules/constants.py:66-74`). Stage 3 targets
`primary[:, :4]` only — ground truth for B4's auxiliary target already exists on disk.

**The recon is inverting a noise process, not physics — this is B1's motivation.** R² is
0.903/0.966/0.977 on surrogate *samples* but 0.876/0.880/0.896 on the real kernel, and
*negative* on surrogate *means*. `/n/home05/zdimitrov/diag_stage3.py` explains why: the sample
regime reaches realistic sparsity only by driving **32–35% of slots negative** — a physically
impossible response that acts as a free, unambiguous "dark" flag the kernel can never supply.
The surrogate mean lights **96.8%** of detectors against the kernel's **60.1%**, erasing the
darkness pattern that carries geometric direction information.

**Energy R² 0.479 is close to arithmetic — B3's motivation.** Target range is 2 decades
(`LOG_E_MIN=5, LOG_E_MAX=7`), corpus std **0.4725 dex**, recon RMSE **0.341 dex** → R² 0.48
almost by construction. The corpus spectrum also falls steeply (top bin has **4.45×** fewer
events than the 5.5–5.75 peak); balanced corpus files exist on disk
(`tau_wholesky_balanced_1e5_1e8_{max,100k}.h5`) but switching to them forces a full Step-0
rebuild, which is why B3 (loss reweighting) tests the same hypothesis cheaply instead.

**Timing has no propagation term.** `place_clouds_enu` never touches the time column, and the
kernel adds no `|r_p − r_det|/c` — `T` is the particle's time at its observation plane, not
arrival at the detector's 3-D position. Second-order (~0.17 µs against a 1.1–141 µs span), but
it means a textbook plane fit is mildly mis-specified — which is exactly what A0 measured
(and found the transverse signal survives regardless).

**T is a leading edge**, not a weighted mean — `amin` over points clearing `HIT_DEPOSIT_MIN`,
raw seconds, `0` = no hit (`modules/showers/kernel.py:97-107`).

> `docs/THEORY.md` is stale in several places: T described as a weighted mean (it's a leading
> edge), σ=200 m (is 50), `LAYER_EAST_DX`=150 (is 500), `T_LOG_SCALE`=1e8 (is 1e6), a 5-D
> primary (is 8-D), 2 species (is 3), and §10.3's "recon permutation-dependence is open" (it's
> fixed — `notebooks/recon_grid_score_diagnostics_deepsets.ipynb` Test 8 measures exactly
> 0.0000 permutation delta). Trust `modules/constants.py` and `modules/showers/kernel.py`.
> Worth a doc-fix commit, separate from this work.

---

## Part A — what the script does

`plots/training/04_analytic_recon.py`: self-contained, read-only w.r.t. the pipeline. Reuses
`shower_level_split`/`load_kernel_responses` from Stage 3 (imported by path, `recon_regimes.py`'s
own trick) so it scores the same val rows recon validates on, against kernel ground truth (not
the FNN surrogate — the point is a reference independent of anything neural).

- **A0** (gate) — fits `t_i` against East alone, then against full (East, North, Up), and
  reports the additional variance explained by the transverse terms. Answered: yes, real signal
  survives beyond the depth ramp (+0.26 median R²).
- **A1** (direction) — weighted plane-front least-squares fit, deposit `E` as the multiplicity
  weight (mirrors `reco.jl`'s `sqrt(nhits)` noise model). Linear, closed-form — no Minuit needed.
- **A2** (origin) — front curvature → source distance, compared against the true decay vertex.
  Confirmed uninformative, as expected going in (curvature term is orders of magnitude below
  the plane term at this array's scale).
- **A3** (energy) — LDF Poisson fit including the un-hit-module term, calibrated on train,
  scored on val. Decisive metric is RMSE in dex, not raw R², given the 2-decade energy range.
- **A4** (multiplicity) — `N_lit` vs energy and vs axis-distance; fills a gap nobody had
  measured (checked every notebook and run-dir dump before writing this).

Run: `python plots/training/04_analytic_recon.py --n-val 20000 --n-train-calib 20000` on a
GPU node (same memory footprint as `recon_regimes.py`/`aleatoric_floor.py`).

---

## Part B — Stage-3 changes, ranked

### B1 — Explicit hit mask + masked pooling *(strongest evidence, do first)*

There is no mask anywhere in `DeepSetsRecon` — no mask argument, no masked-mean denominator.
`h.mean(dim=1)` always divides by 100 and `h.max(dim=1)` competes over all 100 slots, including
dark ones (whose z-scored tokens are large negative values in E and T). With ~64% of slots dark
corpus-wide, typically half the pooled tokens are zero-response and in the p10 tail ~87 of 100 are.

The regime table is the argument: recon scores *higher* on surrogate samples than on the real
kernel precisely because samples hand it an unambiguous dark flag (negative slots) that the
kernel cannot. An explicit `1[T > 0]` feature plus a masked mean supplies that information
legitimately, in the regime `true_utility.py` actually scores. Add the live count as a context
feature too.

Note THEORY §10.1 found zero-inflation handling backfired for the *surrogate* — but that was
loss reweighting on an equivariant model, not pooling on an invariant one. It does not transfer
automatically; measure.

### B2 — Give recon the geometry it is missing

Recon sees `(East, North)`; the kernel's response is defined by `(North, Up, z_cont)`
(`dataset_builder.py:187-192`). `Up = g(North, East)` is deterministic so nothing is *lost*, but
the network must internalize a 256×256 terrain raster over ~960 m of relief using capacity that
should be reconstructing showers. `docs/surrogate-axis-features.md` proposes the same fix for
the surrogate (never landed); the same argument applies to recon.

**Changes the recon input contract**: `RECON_INPUT_FEATURES`, both `torch.stack([...], dim=-1)`
assemblies in Stage 3, `objective.py:253-256`, and `plots/layouts/true_utility.py` must move
together, gated on checkpoint `config` for backward compatibility.

### B3 — Reweight the energy loss by inverse spectrum density

Weight each row's `log_e` loss by inverse spectrum density (cheaper than rebuilding on the
balanced corpus). If the energy plateau is a sampling artifact this moves it; if not, that's
evidence for the arithmetic ceiling, and the answer is to widen `LOG_E_MAX` or accept 0.34 dex.

### B4 — Predict the decay vertex as an auxiliary output *(downgraded — see Results)*

Extend the target to `primary[:, 5:8]`, weighted low enough to regularize, not dominate. Free
at inference (U reads only the first four outputs, `objective.py:237-260`). Was meant to test
the energy/distance degeneracy directly, but A2 already showed an explicit geometric fit can't
localize the vertex from this array's timing — no strong prior an auxiliary head would do
better. Still cheap to try; reprioritize below B1/B2 if time is limited.

### B5 — Direction output is unconstrained

`(n̂_x, n̂_y, n̂_z)` are free targets from a bare `Linear(hidden, 4)`; nothing enforces `|n̂| = 1`.
Normalization happens post hoc at `objective.py:102-106`, whose own comment warns the stored
components aren't guaranteed normalized — the MSE pays for radial error that's then discarded.

### B6 — Loss weighting is accidental

`_per_axis_loss` is an unweighted sum of four raw-unit MSEs, so the effective weighting is an
accident of target variance: `dz` (var 0.0445) contributes far less gradient than `dx`
(var 0.3263), while `logE` is the channel we most want improved. Normalize per channel by
target variance.

---

## Verification

- **B1/B2**: `py_compile` on all touched files, then the documented smoke run
  (`--epochs 2 --lbfgs-iters 3`) proving the new input width flows through Stage 3, Stage 4 and
  `true_utility.py` without a shape error; confirm an old `recon.pt` still loads via `config`.
- **Any B item**: `recon_regimes.py` kernel-regime R² against the (re-baselined, post-B0) number.
  That, not val loss, is the decision metric — the mix-fraction change already made val loss
  incomparable across runs.
- Stage-3 experiments should use **`--out-suffix`** (now implemented) so they don't overwrite
  production `recon.pt`. At ~390 s/epoch on the 8.58M-row corpus, run comparisons on a
  subsampled corpus — never a 600-epoch run.

## Not in scope

- Another mix-fraction change to Stage 3.
- Adding `|r_p − r_det|/c` to the kernel — real, but invalidates `E.pt`/`T.pt` and forces a
  Step-1/2/3 rebuild.
- Switching to the balanced corpus (full Step-0 rebuild); B3 tests the same hypothesis cheaply.
- The rigid-translation artifact (clean grid U=+171.7, +200 m North → +135.8, +500 m → +70.0,
  +1000 m → +33.9, `recon_grid_score_diagnostics_deepsets.ipynb`). Unexplained and live, but a
  Stage-4/corpus-coverage question — `ACTIVE_STRATEGIES` no longer contains any grid strategy,
  so the highest-scoring layout family is out of distribution.
- The stale `objective.py` reconstructability-gate thresholds (`TODO(08)`).

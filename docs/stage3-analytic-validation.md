# Stage 3: analytic validation + targeted improvements

## Status (branch `stage3-analytic-validation`)

- [x] **B0** — blob-row filter landed in `scripts/03_train_recon_deepsets.py` (`5a792d8`).
- [x] **`--out-suffix` flag** — added to `scripts/03_train_recon_deepsets.py` (`5a792d8`).
- [x] **Part A script written AND run at scale — results in, this gates B1/B4 below.**
      `plots/training/04_analytic_recon.py` (`9c930d6`, fixes in `78b39ad`). Three real bugs
      found and fixed while running it on a GPU node: unnormalized conditioning check
      (rejected 100% of events), uncentered-position numerical instability in the plane
      solve (R² ≈ −1e15), and an aggregate-R² blow-up from near-zero-T-variance outlier
      events (fixed in `78b39ad`). Confirmed at `--n-val 500` and then at `--n-val 20000`
      (matching `recon_regimes.py`'s own scale) — same signs and magnitudes at both sizes,
      so this is a stable result, not small-sample noise:

      | | 500 events | 20,000 events |
      |---|---|---|
      | A0 median R² (East only / East+transverse) | 0.086 / 0.30 | 0.083 / 0.347 |
      | A0 median delta (transverse gain) | +0.22 | **+0.264** |
      | A1 dir_x/y/z R² | −0.07 / −0.27 / −16.1 | −0.02 / −0.42 / −18.1 |
      | A1 dir_x/y/z pearson r | 0.23 / 0.29 / 0.04 | 0.24 / 0.27 / 0.03 |
      | A2 median dist. error vs true median | 2520 m vs 2840 m | 2578 m vs 2914 m |
      | A3 log_e R² / pearson r / RMSE (dex) | 0.004 / 0.064 / 0.452 | 0.001 / 0.037 / **0.463** |
      | A4 pearson r: N_lit vs log₁₀E / axis-dist | −0.216 / 0.017 | −0.154 / −0.008 |

      **Verdict:** the naive plane-front + LDF analytic reconstruction is clearly and
      consistently WORSE than recon on both channels it was meant to challenge — direction
      pearson-r 0.24-0.30 (recon: 0.936-0.947) and energy RMSE 0.46 dex (recon: 0.341 dex).
      A2 (curvature → vertex) is confirmed uninformative as the doc's own prior predicted
      (error ≈ 88% of the true distance — no localization power). The one clear positive:
      A0 shows real transverse timing structure beyond the East depth-ramp (+0.26 median R²),
      so recon's use of the T channel is not just picking up a trivial depth correlation.
      **This means Part A does not surface an easy win recon is leaving on the table** — the
      network is already extracting substantially more direction/energy signal than a
      straightforward physics-based fit recovers. It does NOT mean recon is optimal, and does
      NOT weaken the independent evidence for B1 (the surrogate-sample-vs-kernel R² asymmetry,
      diag_stage3.py) — but B4 (predict the decay vertex as an auxiliary target) is now weakly
      motivated at best, since even the analytic method with an explicit vertex-fitting step
      (A2) can't localize it, and there is no reason to expect an auxiliary head would recover
      information the physics itself does not clearly provide from this array's geometry.
- [x] **Kernel-in-the-loop Stage 4** — `scripts/04_optimize_lbfgs_kernel.py` + a
      `differentiable` flag on `compute_labels_batch`/`KernelDualLabels` (`5bde42d`), a
      primary_batch/clouds consistency check on `KernelDualLabels` (`f7fc35a`), and a second,
      reconstruction-independent (analytic) evaluation of the init/best layout (`f14e7da`).
      Added beyond this doc's original scope, at the user's request. Differentiability
      verified end-to-end on synthetic clouds; the optimization loop itself (real corpus,
      real `recon.pt`, a full run) has not yet been executed.
- [x] **A0-A4 at full scale** — done, see table above. Re-run only if Stage 3 changes enough
      to plausibly move the direction/energy comparison (e.g. after B1 lands).
- [ ] **B1-B6** — not started. B1 still well-motivated (independent evidence); B2/B3/B5/B6
      unaffected by this result; B4 downgraded — see verdict above.
- [ ] Re-baseline `recon_regimes.py` after B0 (val loss / R² numbers before this branch are
      not comparable to a run after it).

## Context

Stage 3 trains a DeepSets reconstructor `(East, North, E, T)×100 → (n̂_x, n̂_y, n̂_z, log̃E)`.
Kernel-regime quality (archived mix-0.5 run): dir_x/y/z R² **0.876 / 0.880 / 0.896**,
log_e R² **0.479**.

Nothing in the pipeline has ever been compared against a reconstruction-independent
reference — `true_utility.py`'s GENUINE/ARTIFACT verdict included. The goal is a number:
*"physics gets R² = X on these events; the network gets Y."*

### Findings that shape this plan

**A validated analytic reconstruction already exists.**
`/n/home05/zdimitrov/tambo/tambo_dir_reco/` (`TamboDirReco.jl`, own git repo). `src/reco.jl`:
`plane_chisq_per_hit` (`:22-40`) fits `(θ, ϕ, t0)` from hit times; `timing_llh_per_hit` (`:42`)
adds quadratic front **curvature** `pred_delay = a·r²` — your spherical-front idea, already
written; `Q_ldf(r; β, κ=0.35, r_ref=100m)` (`:68`) + `ldf_llh_per_hit` (`:74`) give energy from
a lateral distribution; `run_minuit_on_joint_llh` (`:263`) fits all 8 parameters jointly.
It also does two things Stage 3 does not: **models un-hit modules explicitly**
(`get_unhit_modules` feeds a separate zero-count term) and **weights detectors by hit
multiplicity** (`σ = 1km·√nhits`).

**The decay vertex is already a label.** `primary.pt` is `(N, 8)`: cols 0-2 direction, 3
`log_e_norm`, 4 EM/hadronic class, **5-7 `(rel_E, rel_N, rel_U)` = ENU decay vertex minus
array centre** (`modules/constants.py:66-74`). Stage 3 targets `primary[:, :4]` only. The
origin you want to reconstruct has ground truth on disk, per row, and is discarded.

**The recon is inverting a noise process, not physics.** R² is 0.903/0.966/0.977 on surrogate
*samples* but 0.876/0.880/0.896 on the real kernel, and *negative* on surrogate *means*.
`/n/home05/zdimitrov/diag_stage3.py` explains why: the sample regime reaches realistic
sparsity only by driving **32–35% of slots negative** — a physically impossible response that
acts as a free, unambiguous "dark" flag the kernel can never supply. Meanwhile the surrogate
mean lights **96.8%** of detectors against the kernel's **60.1%** (37.2% false-positive at
99.2% recall), erasing the darkness pattern that carries geometric direction information.

**Energy R² 0.479 is close to arithmetic.** Target range is 2 decades
(`LOG_E_MIN=5, LOG_E_MAX=7`), corpus std **0.4725 dex**, recon RMSE **0.341 dex** → R² 0.48
almost by construction. Compounding it: the corpus spectrum falls steeply (top bin has
**4.45×** fewer events than the 5.5–5.75 peak), and `scripts/make_balanced_tau_subset.py`
already built balanced files (`tau_wholesky_balanced_1e5_1e8_{max,100k}.h5`, 2026-08-25) but
`TAU_WHOLESKY_PATH` was commented back to the unbalanced `n750k.h5`
(`modules/constants.py:177-178`). The energy channel also plateaus by epoch ~150 — neither
L-BFGS (321 closures) nor more Adam epochs move it.

**Timing has no propagation term.** `place_clouds_enu` rewrites only cols 0/1/2; the time
column is never touched and the kernel adds no `|r_p − r_det|/c`. So `T` is the particle's
time *at its observation plane* (zero = injection at the decay vertex), not arrival at the
detector's 3-D position. The omitted term is ~σ/c ≈ 0.17 µs against a T span of 1.1–141 µs —
second-order, but it means the front structure comes entirely from C8's own development and a
textbook plane fit is mildly mis-specified. A0 measures whether it survives.

**T is a leading edge** — `amin` over points clearing `HIT_DEPOSIT_MIN`, raw seconds, `0` = no
hit (`modules/showers/kernel.py:97-107`). The right observable for a front fit.

### Already resolved — do not re-litigate

THEORY §10.3 lists recon permutation-dependence as open. **It is fixed and measured**:
`notebooks/recon_grid_score_diagnostics_deepsets.ipynb` Test 8 gives a permutation delta of
exactly **0.0000** on grid, rings_R800 and center; Test 9 puts canonical order 0σ from the
15-random-order mean. `DeepSetsRecon`'s `mean`/`max` over `dim=1` are exactly invariant.

> `docs/THEORY.md` is stale in physics-relevant places: T as weighted mean, σ=200 m (is 50),
> `LAYER_EAST_DX`=150 (is 500), `T_LOG_SCALE`=1e8 (is 1e6), 5-D primary (is 8-D), 2 species
> (is 3). `docs/VERSIONS.md:186` says `(log10 E − 5)/3` (is `/2`). §10.3 is stale per above.
> Trust `modules/constants.py` and `modules/showers/kernel.py`. Worth a doc-fix commit.

---

## Part A — Analytic reconstruction (new standalone file) — **done, see Status for results**

**New file: `plots/training/04_analytic_recon.py`.** Self-contained, read-only w.r.t. the
pipeline, no Stage-3 changes. Follows `plots/training/aleatoric_floor.py` conventions (the
`_pathfix.py` walk-up for CWD-independent imports, argparse, JSON + PNG output).

**Python/torch port of the `reco.jl` fits, not a Julia bridge.** TambOpt is 100% Python;
`TamboDirReco.jl` consumes `Tambo.Frame` hit tables rather than `(x,y,E,T)` tensors and needs
`Tambo.jl`/`TamboReader`/`Minuit2`. Porting three scoring functions is less work than an
adapter plus a Julia dependency, and keeps the fit differentiable in torch in case Part B
later wants it as a feature.

Reads Step-1 outputs directly and reuses `shower_level_split(strat_ids, 0.10, 1)` from Stage 3
— the same import-by-path trick `recon_regimes.py:10-19` uses — so it scores **the same val
rows recon validated on**. Report MSE / R² / pearson-r in `recon_regimes.py:58-62`'s exact
format so numbers sit directly beside the baseline.

### A0 — Does the timing support a front fit? (gate)

Depth is `z_cont = (EAST_ENTRY − East)/LAYER_EAST_DX` with 500 m planes, so `T` across the
array is dominated by a near-deterministic ramp in East. The direction signal beyond that is
the *transverse* (North, Up) structure from C8's front curvature.

Fit `t_i` against detector position for fired detectors and report: variance explained by the
East ramp alone; **additional** variance from the transverse terms (this number decides A1);
residual RMS vs raw spread as a per-event histogram — a minority of well-fit events is a
different diagnosis from uniform failure.

If the transverse term explains nothing, the honest conclusion is that recon's direction
signal is *which* detectors fire, not *when*. That stops A1/A2 cheaply and redirects to Part B.

### A1 — Direction: plane-front χ²

Port `plane_chisq_per_hit` (`reco.jl:22-40`); fit `(θ, ϕ, t0)` per event (core is degenerate
with `t0`, as the Julia comment notes). Detector 3-D position: East/North from `xy.pt`, `Up`
from `SurfaceUpMap` — the same call `dataset_builder.py:187` makes. Carry over the Julia noise
model: weight by multiplicity, using deposit `E` as the `nhits` proxy (THEORY §10.4 measures a
single fired detector's time as nearly pure shower noise, so the fit must aggregate). Require
≥4 fired detectors and report the qualifying fraction. Batched torch `lstsq` seed + short
Gauss-Newton polish so the val split runs in one GPU pass.

### A2 — Origin: front curvature → decay vertex

Port `timing_llh_per_hit`'s `pred_delay = a·r²`. Fit curvature `a`, convert to a source
distance, compare against the true `primary[:, 5:8]`.

Bounded experiment with an honest prior: at ~1 km decay distance and a ~1 km array the
curvature term sits orders of magnitude below the plane term, and per-detector timing is
shower-noise dominated. Report the fitted-`a` and vertex-error distributions. A clean negative
says the array cannot localize the vertex from timing — worth knowing before Stage 4 is asked
to optimize for it, and it bounds how much B4 can recover.

### A3 — Energy: LDF fit

Port `Q_ldf` and `ldf_llh_per_hit` **including the un-hit module term** — the piece Stage 3
has no analogue for, and real information (a dark detector bounds the footprint). Axis from
A1, core from the LDF fit, density at `r_ref` calibrated to `log_e` on the **train** split,
scored on val.

Given the arithmetic above, the decisive comparison is not raw R² but **RMSE in dex**: beat
0.341 and the network is leaving energy on the table. Also report LDF error *at fixed decay
distance* vs marginal over distance — that directly tests whether energy is degenerate with
decay distance, which is the hypothesis behind B4.

### A4 — Multiplicity vs energy and vs distance (fills a documented gap)

Nobody has ever plotted `N_lit` against primary energy or axis-to-detector distance — I
searched every notebook, `plots/`, and run-dir dump. It is load-bearing: the
reconstructability gate spans p10=13 / p50=48 / p90=82 and `objective.py:40-60` carries an
explicit `TODO(08)` saying its thresholds are stale, tuned on a two-component corpus before
photon (~2× electron's points) was summed in, so `r` is re-flattening toward 1.0 — the exact
zero-gradient failure the 0.05→1.0 change fixed, arriving from the other side.

Cheap (no fitting, no generation), and it tells you where reconstruction actually fails.

---

## Part B — Stage-3 changes, ranked

### B0 — `blob_ids.pt` is never filtered in Stage 3 *(a real bug, ~5 lines)* — **DONE (`5a792d8`)**

Step 2 drops degenerate showers from train, val **and** its norm stats, and hard-errors if the
file is missing (`02_train_fnn_deepsets.py:376-394, 939-942`). `03_train_recon_deepsets.py`
never loads it — grep returns nothing. Run 11 flagged 12,390 muon and 25 photon pairs. Those
rows are in recon's training set, its z-score statistics, and its validation metric.

Cheapest item by a wide margin, and it contaminates the metric every other experiment is
judged on. Do it first and re-baseline.

### B1 — Explicit hit mask + masked pooling *(strongest evidence)* — **not started**

There is no mask anywhere in `DeepSetsRecon` — no mask argument, no masked-mean denominator.
`h.mean(dim=1)` always divides by 100 and `h.max(dim=1)` competes over all 100 slots,
including dark ones (whose z-scored tokens are large negative values in E and T). With ~64% of
slots dark corpus-wide, typically half the pooled tokens are zero-response and in the p10 tail
~87 of 100 are.

The regime table is the argument: recon scores *higher* on surrogate samples than on the real
kernel precisely because samples hand it an unambiguous dark flag (negative slots) that the
kernel cannot. An explicit `1[T > 0]` feature plus a masked mean supplies that information
legitimately, in the regime `true_utility.py` actually scores. Add the live count as a context
feature too.

Note THEORY §10.1 found zero-inflation handling backfired — but that was *loss reweighting* on
the equivariant surrogate, not *pooling* on an invariant recon. It does not transfer; measure.

### B2 — Give recon the geometry it is missing — **not started**

Recon sees `(East, North)`; the kernel's response is defined by `(North, Up, z_cont)`
(`dataset_builder.py:187-192`). `Up = g(North, East)` is deterministic so nothing is *lost*,
but the network must internalize a 256×256 terrain raster over ~960 m of relief using capacity
that should be reconstructing showers.

`docs/surrogate-axis-features.md` already proposes exactly this (add `u_d`, along-axis depth
`s_d`, impact parameter `rho_d`) for the **surrogate**, with a measured symptom —
under-dispersion, log-std 2.27 vs the kernel's 3.17, corr 0.79, "it smooths a sharp function."
Status: a plan, never landed. The same argument applies to recon.

**Changes the recon input contract**: `RECON_INPUT_FEATURES`, both `torch.stack([...], dim=-1)`
assemblies in Stage 3, `objective.py:253-256`, and `plots/layouts/true_utility.py` must move
together or a 4-feature tensor gets fed silently into a 6-feature model. Gate on checkpoint
`config` for backward compatibility.

### B3 — Reweight the energy loss by inverse spectrum density — **not started**

The corpus under-samples the top half-decade 4.5×, exactly where AllShowers already
extrapolates (`constants.py:80-84`: its training set ran out at 4.9e7). Balanced corpora exist
on disk but switching `TAU_WHOLESKY_PATH` forces a full Step-0 rebuild.

Cheaper and testable now: weight each row's `log_e` loss by inverse spectrum density. If the
energy plateau is a sampling artifact this moves it; if not, that is evidence for the
arithmetic ceiling and the answer is to widen `LOG_E_MAX` or accept 0.34 dex.

### B4 — Predict the decay vertex as an auxiliary output — **downgraded by A2's result**

Extend the target to include `primary[:, 5:8]`. Two reasons: it **tests the energy/distance
degeneracy** (if total deposit confounds energy with decay distance, a network that cannot
infer distance cannot invert energy), and it is free at inference — U reads only the first four
outputs (`objective.py:237-260`), so Stage 4 is unaffected if the head is extended rather than
replaced. Weight the auxiliary terms low enough to regularize, not dominate.

**Gated on A3 as written; A2's result (see Status) weakens the case.** A2 fits the front
curvature explicitly to localize the vertex and gets a median error ≈ 88% of the true
distance — essentially no localization power, at both 500- and 20,000-event scale. If the
physics itself (via an explicit geometric fit) cannot pin down the vertex from this array's
timing, there is no strong prior that an auxiliary recon head would recover that information
either. Still cheap to try since it's free at inference, but it is no longer the well-motivated
item it looked like before A2 ran — reprioritize below B1/B2 if time is limited.

### B5 — Direction output is unconstrained — **not started**

`(n̂_x, n̂_y, n̂_z)` are free targets from a bare `Linear(hidden, 4)`; nothing enforces
`|n̂| = 1`. Normalization happens post hoc at `objective.py:102-106`, whose own comment warns
the stored components are not guaranteed normalized. The MSE is isotropic in ℝ³ while the
physical quantity lives on S², so radial error is paid for in the loss and then discarded.

### B6 — Loss weighting is accidental — **not started**

`_per_axis_loss` (`:277-283`) is an unweighted sum of four raw-unit MSEs, so the effective
weighting is an accident of target variance: `dz` (var 0.0445) contributes far less gradient
than `dx` (var 0.3263), while `logE` is the channel we most want improved. Normalize per
channel by target variance.

---

## Sequencing

1. **B0** — done. Everything else is measured against a contaminated baseline until this lands.
2. **A0** — done, positive (real transverse timing signal beyond the depth ramp).
3. **A1 / A3 / A4** — done. Analytic direction/energy both clearly worse than recon;
   A2 (run alongside) confirms no vertex-localization power. See Status for numbers.
4. **B1** — next up. Best-evidenced change (surrogate-sample-vs-kernel R² asymmetry),
   independent of the A-phase result, measured against recon_regimes.py's baseline.
5. **B2**, then **B3** / **B5** / **B6** as cheap follow-ons.
6. **B4** — downgraded by A2's result (see its own section); do last if at all.

Stage-3 experiments need an **`--out-suffix` flag** (not implemented; add it) so they do not
overwrite production `recon.pt`. At ~390 s/epoch on the 8.58M-row corpus, A/B comparisons run
on a subsampled corpus — never 600-epoch runs.

## Verification

- **A0-A4**: `python plots/training/04_analytic_recon.py --folder <recon_dir>` on
  `-p gpu_test` (no shower generation, so minutes). Success = per-channel R² and dex-RMSE for
  the analytic method on the same val split, printed beside the recon numbers. A0's
  transverse-variance number must be reported before A1-A3 are trusted.
- **B0**: row count drops by the blob fraction; re-run `recon_regimes.py` and state the
  shifted baseline explicitly, since all later comparisons hang off it.
- **B1/B2**: `py_compile` on all touched files, then the documented smoke run
  (`--epochs 2 --lbfgs-iters 3`) proving the new input width flows through Stage 3, Stage 4 and
  `true_utility.py` without a shape error; confirm an old `recon.pt` still loads via `config`.
- **End-to-end**: `recon_regimes.py` kernel-regime R² against 0.876 / 0.880 / 0.896 / 0.479.
  That, not val loss, is the decision metric — the mix-fraction change already made val loss
  incomparable across runs.

## Not in scope

- The running 600-epoch job 45423856, or another mix-fraction change.
- Adding `|r_p − r_det|/c` to the kernel — real, but invalidates `E.pt`/`T.pt` and forces a
  Step-1/2/3 rebuild.
- Switching to the balanced corpus (full Step-0 rebuild); B3 tests the same hypothesis cheaply.
- The rigid-translation artifact (clean grid U=+171.7, +200 m North → +135.8, +500 m → +70.0,
  +1000 m → +33.9, `recon_grid_score_diagnostics_deepsets.ipynb`). Unexplained and live, but a
  Stage-4/corpus-coverage question — note that `ACTIVE_STRATEGIES` no longer contains any grid
  strategy, so the highest-scoring layout family is out of distribution.
- Kernel-in-the-loop Stage 4 (THEORY §11.3) and the stale `objective.py` gate thresholds
  (`TODO(08)`).

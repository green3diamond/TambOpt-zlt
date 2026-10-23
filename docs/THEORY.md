# TambOpt: how the pipeline works (2026-10-23)

Goal: place N detectors (100) on a mountain face so that tau-induced air showers are reconstructed as well as possible
(direction, energy). The layout is optimised by gradient descent through a differentiable detector response and a
trained reconstruction network.

## 1. Pipeline

| Stage | Script | Output |
|---|---|---|
| Inputs | `17_make_plane_inputs.py` (synthetic planes) or the Malata mesh + `tau_wholesky.jl` | geometry `.h5`, tau decays `.h5` |
| 0 Showers | `00_generate_data_dual_species.py` | AllShowers point clouds per event × species (e, μ, γ); 5 % held out |
| 1 Labels | `01_build_dataset_northeast.py` | per row (shower × layout): E, T per detector per species from the kernel |
| 2 Surrogate | `02_train_fnn_deepsets.py --species s` | hurdle DeepSets per species: P(lit), value (only for surrogate-based Stage 4) |
| 3 Recon | `03_train_recon_deepsets.py --recon-input kernel` | DeepSets: detector (E, N, E, T) set → primary (direction, log E) |
| 3c Fine-tune | `03c_finetune_recon_noise.py` | recon on background / counted inputs (optional) |
| 4 Optimise | `04_optimize_lbfgs_kernel.py` | layout: Adam (+ chunked L-BFGS), best on a separate scoring set |
| Score | `plots/analysis/run19/*/score_*.py` | layouts on held-out showers (true kernel, true recon) |

Run folder and every path: `modules/constants.py` (`RUN_LOCATION`, overridable with `TAMBOPT_RUN_LOCATION`).

## 2. Physics model

**Primaries.** Tau decays with a position, direction and energy, log10(E/GeV) ∈ [5, 7]. Encoded as 8 numbers:
direction (3), normalised log E, EM/hadronic class, decay vertex relative to the array centre (3).

**Showers.** AllShowers (CORSIKA 8 surrogate) gives super-particles per species on 24 planes, 500 m apart along the
axis (12 km). Per point: horizontal transverse offset t1, plane index, time, kinetic energy. The vertical transverse
offset t2 was never recorded.

**Placement (shower frame).** Each shower sits in its own frame at its decay vertex, along its true direction. A
detector is mapped into it: depth along the axis (plane l at (l + 1) × 500 m) and distance from the axis.

**Kernel (detector response)** (`modules/showers/kernel.py`, `compute_labels_batch`):
- **Ring:** each point is a ring of radius |t1| around the axis (circular showers); detectors read by axis distance.
- **Gaussian:** σ = 10 m, widened to the 8th-nearest-neighbour distance, capped at 50 m (`KERNEL_INTERP_K`, `_MAX`).
- **E** = Σ KE × Gaussian weight (peak 1, total 628 m²): GeV collected over ~628 m², **not particles**.
- **T** = earliest point with deposit > 10⁻³ (`HIT_DEPOSIT_MIN`), carried along the axis at c (no 1.67 µs steps).
- Species combine as E = Σ_s E_s, T = min_s T_s.
- Differentiable in the detector positions.

**Detectors.** Parameterised by (East, North), snapped onto the mesh surface (`project_to_mountain_ne`).

**Particles from E** (`modules/optimize/background.py`, `counting.py`). Expected detected particles in a 2 m²
detector: μ_s = E_s · A / (2πσ² KE_s(r)) / ring_s(r), r = axis distance; KE_s(r), ring_s(r) from 16 CORSIKA 8 π⁻
showers (`c8_scaling.json`); photons × ε_γ = 0.02. Optional Poisson counting: N_s ~ Poisson(μ_s), E′ = N_s / k_s.

**Background.** Cosmic muons, 1.826·10⁻⁶ /m²/ns × 2 m² × window. Each muon: random KE from (KE + 2.6 GeV)^−2.7
(0.1 GeV – 1 TeV, mean 3.85 GeV), E += KE · 2πσ²/A; T = min(T, random time in the window). Added after counting,
independent of the shower (`TAMBOPT_BG_UNITS=cosmic`; `count` = +1 per muon, the run 19 noise results).
Optional kNN 2-of-3 gate: a detector is read out only if ≥ 2 of {itself, its 2 nearest} have E > 10⁻³.

## 3. Training data (Stage 1)

Every shower is labelled under layouts drawn from 11 strategies (`modules/layouts/strategies.py`): grids
(jitter 20 / 200 m), centre Gaussians (σ 200 / 400 / 800 m), rings (0.3 / 0.8 / 1.8 km), uniform random, clusters,
disks, grid patches. The recon is trained on these kernel labels, so it has seen spread-out and compact layouts.

## 4. Recon (Stage 3)

DeepSets (permutation invariant). Input `reltime1`: positions relative to the earliest timed detector, linear Δt, lit
flag. Target: direction and log E. AdamW (weight decay 0.03), then L-BFGS. Run 19: val 0.132, ~10–12 % of showers
within 1° on the best layouts (resolution floor ~2.5° median, set by the kernel times, not the recon).

## 5. Layout optimisation (Stage 4)

`04_optimize_lbfgs_kernel.py` (`slurm/kernel_opt_run19.sh <init> <utility> [λ] [lr] [steps]`). Per step: a fresh
batch of Stage 1 showers (`--batch-events`, 2,048), utility, gradient to (E, N), Adam; best layout on a fixed scoring
set; optional chunked L-BFGS polish. Starts: `grid` or `center`.

**Response (what gives E, T for a layout):**

| `--response` | How | Use |
|---|---|---|
| `kernel` (default) | the kernel itself on the placed showers, differentiable | exact, slower |
| `surrogate` | Stage 2 hurdle per species; lit if P(lit) ≥ `--p-lit-min` | fast, approximate; no μ, so no `swgo_trigger` |

Options: `BG_WINDOW` (background), `GATE=knn23`, `FREEZE_UNSEEN` (detectors without signal stay), `LBFGS_CHUNKS`.

**Utility** (`TAMBOPT_UTILITY`, `modules/optimize/objective.py`; a = angle error, e = |Δlog10 E|):

| Mode | Per shower | Trigger |
|---|---|---|
| `current` | weighted 1/(err² + ε) terms, capped | soft ≥ 5 detectors with E > 1 (kernel units ≪ 1 particle) |
| `two_tier` | g + λ(1 − g) q; g = soft(a < 5°) · soft(e < 0.2), q = 1/(1 + (a/20)² + (e/0.5)²) | none |
| `three_tier` | 0.9 if a < 1°, else 0.3 if a < 5°, else 0.1 · detected | "detected" = ≥ 3 detectors with E > 1, consolation only |
| `swgo_trigger` | r_b · (g + λ(1 − g) q), g angle only (energy with `TRIG_ENERGY=1`) | r_b = P(≥ k detectors with ≥ 1 particle), Poisson-binomial on μ; missed shower = 0 |

`swgo_trigger` (branch `swgo-trigger`, now on master): `TRIG_K` (3), `TRIG_AREA` (2 m²). μ comes from the kernel
with gradients; the recon input stays noise-free, so the current recon applies.

## 6. Setups

**12 × 12 km plane (run 19, current default).** `<v6>/19_plane45S_12km_showerframe_400k`.
- Plane: 45°, facing South, 12 km side (`plane45s12.h5`), as Malata's slope; taus travel North into it.
- Decays: Latin hypercube, 0.5–6.5 km in front of the plane, 12 km wide and high; target uniform on the plane;
  zenith 60–100°; ≥ 30° from the C8 magnetic-field axis (East); decay–target ≤ 11.5 km (the C8 showers span 12 km).
  `17_make_plane_inputs.py --side 12000 --box-depth 6000 --max-dist 11500 --facing south --zenith 60 100
  --min-angle-b 30 --n-events 421200 --group plane45s12 --seed 19`.
- 421,200 showers (5 % held out), 100 detectors (~1.2 km apart on a grid), recon `reltime1`.
- Findings: even coverage wins under the two-tier utility; with a trigger (`swgo_trigger` check) the best is a
  1–2 km disk (`docs/swgo-clustering-and-counting.md` § 5).

**Malata (the real site).** `data/malata.h5`, group `malata`, 266 detector faces; site-local ENU with origin at the
mesh `location` (lon −71.97°, lat −15.58°); bbox North [−956, 716], East [−499, 777], Up [2748, 3712] m (~1.4 km).
- Taus: `decay_locations/tau_wholesky.jl` (whole-sky throw, Earth-skimming), cuts zenith 60–100° from +Up and decay
  ≥ 1 km from the mesh (`--mindist`); ~3 % of thrown taus survive. Julia writes (3, N): `modules/showers/tau.py`
  handles it.
- Last run with the old global-frame kernel (runs ≤ 16). To run it now: `TAMBOPT_GEOMETRY_PATH=<malata.h5>`,
  `TAMBOPT_GEOMETRY_GROUP=malata`, `TAU_WHOLESKY_PATH` (constants.py) to the cut corpus, a new `RUN_LOCATION`, then
  Stages 0–3.
- 100 detectors at ~120 m spacing.

## 7. Hazards

- **AllShowers checkpoint ↔ code:** the checkpoint needs `with_time` in `generator.py` and the post-LayerNorm
  `transformer.py`; other versions silently give blobs (`tests/README.md`).
- **Blob showers:** 0.5 % of muon showers are degenerate (median point energy > 10³ GeV), flagged at Stage 1.
- **E is not particles:** thresholds on E (> 1, > 10⁻³) are far below one particle in a 2 m² detector; use μ.
- **Showers near the C8 field axis** (East) come out shifted sideways; inputs exclude ±30°.
- **Any change to the kernel** (σ, k, cap, frame, ring) changes the labels: Stage 1 and everything after must rerun.

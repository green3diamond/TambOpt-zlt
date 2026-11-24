# TambOpt audit against TamboSim: what was tried and what could help

Status of the items in `TambOpt Audit Against TamboSim.pdf` (17 Nov 2026), as of 23 Nov 2026. The audit reviewed
`master` (69c04f6), not `cleanup` or `timing-model`, so parts of it are out of date. Items marked "checked 23 Nov"
were verified in the code or with a test; the other rows describe the work done so far. Since 24 Nov `master`
(1adfb5f) contains `cleanup` and `rerun-mean5k` (with `timing-model`), so the audited fixes are now on `master`.

## Already tried or largely covered

| Audit item | What was done | Result |
|---|---|---|
| No timing gradient (`delta_l` detached) | `timing-model` branch: carry along the axis attached, mean-delay time model (`TAMBOPT_TIME_MODEL=mean`) | **Done**: all main-text optimizations of papers 1 and 2 rerun with it (`_m5k`: mean time without noise, 5,120 showers per step); batch gradients agree (cosine 0.66-0.76, flow matching); swgo_trigger on Malata: batch-to-batch cosine 0.17-0.25 (per-batch signal-to-noise ~0.5, enough for SGD; `trig_grad_r_check.py`) |
| Signal unit (kinetic energy, not counts) | Counting model: expected counts mu per species, Poisson draws, mu >= 1 hits, trigger on mu (merged 23 Oct) | **Done**: counts for hits, trigger, triggered selection and the analytical fit; the DeepSets reads E **by design** (E proportional to mu at fixed species and r, so no information lost) |
| Thresholds far below one particle | Hits and trigger use mu >= 1 | **Partly done**: selection (>= 3 detectors with mu >= 1) and the analytical fit. **Open, major**: the DeepSets input is not masked, so its accuracies are about **2x optimistic** (see "Sub-threshold detectors in the reconstruction" below) |
| Optimized and scored with the same reconstruction (layouts can exploit quirks of the DeepSets network) | CORSIKA 8 columns scored with a DeepSets fine-tuned from the same network, and with the analytical fit (no training) | **Largely addressed**: no sign of exploitation. Optimized layouts tie with ugrid under both networks (Malata, % within 1 deg: 35.3-36.9 vs 37.0 FM, 30.0-32.1 vs 31.7 C8-ft) and do not stand out under the analytical fit (23.2-28.2 vs grid 26.1). Network swap on the same FM showers keeps the order (Malata ugrid / optimized / grid 37.0 / 35.3-36.9 / 34.8 -> 24.9 / 22.9-23.8 / 22.9); the optimized edge over the grid shrinks from 0.5-2.1 to 0-0.9 points on Malata, none on the 12 km plane (ugrid / current from ugrid / three-tier from grid 13.2 / 12.5 / 11.8 -> 9.3 / 9.4 / 10.0). Full confirmation: a DeepSets retrained from scratch |
| No validation of generated showers | Paper 1 Fig. 8 (CORSIKA 8 vs flow matching vs surrogate at the detectors); CORSIKA 8 scoring in paper 2; `scripts/paper2/time_residual.py` | **Done**: generator fidelity checked against CORSIKA 8. Whether the CORSIKA 8 setup itself matches TamboSim is the physics-realism items below (shower energy, magnetic field, atmosphere) |
| Utility credit for undetected showers | SWGO trigger utility is the main one in paper 2 | **No change needed**: the original utility's 0.27 credit is a property of that utility. Two-tier (no detection term) and three-tier ("detected" = E > 1 in kernel units) are **by design**: the utilities compare optimizing on energy with optimizing on particle counts (SWGO trigger) |
| Detector area (2 m^2 vs TamboSim's 2x2 m box: 4 m^2 top face plus sides) | Papers use 2 m^2 only (the 20 m^2 runs were dropped from paper 2) | **Open**: counts and background 2x low against TamboSim; 2 m^2 is stated in paper 2. The 20 m^2 runs are **removed**; box shape and incidence angle not modelled |
| Run 19 phase space | 12 km plane framed as a geometry study; Malata uses real tau decays | **No change needed**: framed; no physical rate weighting. **Open** for Malata: headline numbers are unweighted means over the injection spectrum (no `oneweight`). The decays themselves are realistic: `decay_locations/tau_wholesky.jl` (TamboSim) injects nu_tau with E^-1.5 from 3e5 to 1e9 GeV, zenith 60-100 deg, all azimuths, propagates them through the Earth (TauRunner) with a forced CC interaction at the site, tracks the tau with PROPOSAL, and keeps decays in air, before the observation mesh (forward ray still crosses it) and at least 1 km from it. Only pdg, energy, direction and position are saved (no weights), so the event mix follows E^-1.5 and the forced interaction, not a physical rate |

## Optimizer gradients and surrogate paths (checked 23 Nov): **No problem**

- **Trigger gradient, distance part** (`axis_distance` under `no_grad`, `background.py:137`): **No problem** (checked
  23 Nov, `trig_grad_r_check.py`, Malata, 4 x 5,120 showers, grid and the _m5k swgotrig layout). With r attached the
  gradient changes by 2-3 % (cosine 0.9996), U is identical; for scale, the batch-to-batch cosine is 0.17-0.25. The `no_grad` stays (no fix needed).
- **Non-differentiable snap to the mesh** (`placement.py:42-65`): **No problem** (checked 23 Nov). The snap moves
  only detectors more than `max_gap` (2x the mean centroid spacing) off the mesh, and runs after each Adam step and
  after each L-BFGS chunk. Every chunk builds a new `LBFGS` (`04_optimize_lbfgs_kernel.py:695`), so **a snapped
  position never enters a curvature pair**; for Adam it acts as a projection onto the mesh.
- **Two surrogate-response paths disagree** (`04_optimize_lbfgs_ensemble.py` via `dual.py` vs the kernel script's
  `SurrogateResponse`): **No problem** (checked 23 Nov). All surrogate runs (`_sur`, `sur5*`) come from
  `04_optimize_lbfgs_kernel.py --response surrogate` (`kernel_opt_run19.sh`, `run20_quick_tests.sh`); the ensemble
  script is used only by two diagnostics.

## Sub-threshold detectors in the reconstruction (checked 23 Nov): **Open**, quantified (~2x optimistic, ranking holds)

The DeepSets gets E and T from every detector with E > 1e-3 (`score_all.py:170`), also where mu < 1; the trigger
threshold itself was always mu >= 1, and the trigger multiplication does not remove these detectors.

- Count (`subthreshold_timed_count.py`, Malata, 5,000 FM showers): per triggered shower ~18 timed detectors with
  mu >= 1 and ~65 with mu < 1 (84 % of them mu < 0.1), the same for grid, disk and optimized layouts.
- Effect (`COUNTING=poisson` in `score_all.py`: Poisson-drawn counts, E and T zeroed where nothing arrives; the
  count-trained recon of run 20b, `test_v6_run_03_recentered_deepsets_count`, also noise fine-tuned), Malata FM,
  % within 1 deg, paper -> counts: grid 34.8 -> 16.8, ugrid 37.0 -> 16.5, optimized 35.3-36.9 -> 15.8-17.5,
  tri33 31.7 -> 14.0, disk r400 24.2 -> 12.4, disk r150 11.7 -> 4.6 (all 12 head layouts).
- So the published DeepSets accuracies (FM and C8-ft) are about 2x optimistic. The order of the layout classes holds
  (even coverage > clusters > small disks), but grid, ugrid and optimized become indistinguishable (optimized
  15.8-17.5 vs grid 16.8). kNN gate still E > 1e-3.

Paper 2 text (no new experiments; **added to paper 2** 24 Nov, methodology and conclusion, Overleaf 21ac3a0 and 19264c2): call the DeepSets accuracies an idealized readout (upper bound), the analytical fit
the realistic counterpart, and quote the count check once; it supports Q2 (no optimized layout beats the grid).

- `chapters_2/02_methodology.tex`, Scoring subsection, new paragraph after the trigger paragraph (line 125), before
  "In the tables each cell gives...":

  > The reconstruction network reads the noise-free kernel energy and arrival time of every detector reached by the
  > shower, including detectors that expect well below one particle; its accuracies are therefore upper bounds for an
  > idealized readout. The analytical fit, which uses only detectors with at least one expected particle, gives the
  > realistic counterpart. As a check, scoring the Malata layouts with Poisson-drawn particle counts (energy and time
  > set to zero where no particle arrives) and a network trained on such inputs roughly halves the share within
  > 1 deg (from 35-37 % to 16-17 %), keeps the order of the layout classes, and makes the grid, the uniform grid
  > and the optimized layouts indistinguishable.

- Conclusion, after "...the reconstruction prefers coverage over local clusters.":

  > The reconstruction network reads every detector reached by the shower, including those expecting well below one
  > particle, so its accuracies are those of an idealized readout; with Poisson-drawn particle counts and a network
  > trained on them they are about half as large, with the same preference for even coverage and no remaining
  > difference between the uniform grid and the optimized layouts.

  Optionally in its future-work sentence: "and a reconstruction trained on realistic particle counts".
- Abstract: when headline accuracies are added, call them idealized readout or use the analytical / count numbers.

## Counting model: energy vs particle counts (checked 23 Nov)

The kernel gives kinetic energy E_s per species; the counting model turns it into expected particles
mu_s = E_s / KE_s(r) (per-particle energy from `c8_scaling.json`: 16 CORSIKA 8 pi- showers, planes pooled). At fixed
species and r, E and mu are proportional, so the reconstruction loses nothing by reading E. mu enters paper 2 through
the trigger (utility, rate, triggered-shower selection of the Malata accuracies) and the analytical reconstruction.

Check on 100 raw CORSIKA 8 showers with the same cuts (`netscratch/.../tambo_simulations_test_raw_v2`, 50 pi+ and
50 e-, 1e5-1e8 GeV; `scripts/diagnostics/c8_raw_scaling_check.py`, plots in `c8_calibration`):

- pi+ (hadronic decays, ~65 %): matches the table for e, mu and gamma at all r, no energy dependence. Table is fine.
- e- (EM decays, ~18 %): about 2x more e and gamma per GeV than the table, so mu is underestimated.
- Muons vary 3-5x with plane depth near the axis (< 1.5x beyond 1 km); electrons dominate the counts inside ~1 km.
- Axis reference (audit, suspected): the table and this check measure r from each plane's energy-weighted centre,
  the kernel from the nominal axis. Confirmed in the code (`c8_ring_radius_check.py:57`, `dataset_builder.py:140`).
  **No problem** (checked 23 Nov, `c8_axis_offset_check.py`): centre offset median 0.1-0.6 m, 90 % < 5 m, max 86 m;
  mu changes < 1 % beyond 10 m from the axis for all species, up to ~10 % only inside 3 m.

**No change needed**: keep the table; state the EM and depth effects as a systematic.

## Physics realism: **Open**, needs new CORSIKA and Stages 0-4

1. **Shower energy and decay class**: the shower gets the tau energy instead of the visible energy (about 2x too
   bright); class 50/50 instead of decay-channel fractions; tau -> mu nu nu gets a full shower. Largest effect on
   rates and trigger brightness.
2. **Magnetic field**: training CORSIKA field rotated 90 deg and partly in the wrong units, so the +-30 deg cut
   removes the wrong directions. Probably the source of the 12-21 deg azimuth shift seen earlier.
3. **Atmosphere and altitude**: showers start at 2.5 km in a standard atmosphere; Xmax 2-3 planes too early.

Smaller items in this group (audit's estimates): thinning weights (EM energy up to 5x low near 1e7 GeV), 25k-point
cap, species generated independently (variance up to 2x low), sqrt(2) ring correction, incidence angle (up to 5x on
slopes), photons counted at 2 % (4.5x fewer at 300 m), energy cuts (1.3-2x in counts), 12.5 km depth limit (7.4 % of
pairs), blob guard (may cut real EM showers near 1e7 GeV; not checked). With the 2x area these lower our counts by a
factor of a few against TamboSim, while item 1 raises them 2x, so absolute trigger numbers are uncertain at that level.

Item 1 without new experiments: not a pure 2x (brightness moves the low-energy end across the trigger threshold, the
visible fraction spreads, the muonic 17 % should be undetectable). Layout rankings are likely robust (all layouts see
the same showers); absolute efficiencies are optimistic, per tau decay at most 0.83x. Paper 2 (**added** 24 Nov): one paragraph in
`chapters_2/02_methodology.tex` after the pipeline summary (line 14), optionally one caveat line in `04_conclusion.tex`:

> Each shower is generated with the full tau energy and a shower type (electromagnetic or hadronic) drawn with equal
> probability. Real tau decays pass on average about half of the energy to neutrinos, are about 65 % hadronic and
> 18 % electromagnetic, and in 17 % of the cases decay to a muon, which produces no detectable air shower. Our showers
> are therefore on average about twice as bright as those from the same tau energy, and efficiencies per tau decay are
> at most 0.83 times those per shower given here. Since every layout is evaluated on the same showers, these
> simplifications shift all layouts alike; the conclusions concern the relative performance of layouts, not absolute
> rates. Conditioning the generator on the visible decay products is left for future work.

Items 2 and 3, same place, right after it (**added** 24 Nov; field distortion depends on direction, so "alike" holds only because every
layout sees the same corpus; not tested whether the ranking changes; item 3 partly offsets item 1):

> The CORSIKA 8 training showers use a geomagnetic field that does not match the TAMBO site in direction and strength,
> and a standard atmosphere with showers starting at 2.5 km instead of at the decay altitude. The field mainly affects
> the lateral shape and azimuth of showers travelling across it; the denser air makes showers about 10-20 % shorter and
> narrower than at the site, so footprints and trigger efficiencies are, if anything, underestimated. Both act on all
> layouts alike and change absolute numbers rather than the comparison between layouts; regenerating the training
> showers with the site's field and atmosphere is left for future work.

These change absolute numbers; whether they change which layout wins is unknown. As the audit says, they should
be fixed before further optimizer work, because they invalidate the corpus.

## Background rate (checked 23 Nov)

- **Background rate**: **No change needed** (keep as conservative). The 1,826 /m^2/s is SWGO's all-particle rate at
  ~4.8 km for water Cherenkov tanks (`swgolo148.cpp:329-331`, HAP-2024-004), applied here as muons. The audit's 4-6x
  compares it with muons only (300-450 /m^2/s at 3-4 km). Scintillators see every charged particle as ~1 MIP, and at
  3-4 km e+- are comparable to muons: all charged ~700-1,000 /m^2/s (rough estimate), so the current rate is ~2x high,
  less on tilted panels. Paper 2: keep 1,826 as a conservative upper bound (above TAMBO's altitude; **added** 24 Nov to the Background paragraph, cited as DORIGO2025116934, check vs HAP-2024-004); quote the muon-rate
  rescoring as the lower bound. 1 us results barely depend on the rate; 10 us results do strongly. Muon-only
  rescoring: `docs/background_rate_muon_rescore.md`.

- **How area, shower energy and background rate combine** (vs TamboSim):

  | | Area (2 vs ~4 m^2) | Shower energy (tau, not visible) | Background rate (all charged) | Net |
  |---|---|---|---|---|
  | Signal | x0.5 | x2 (mean; wide spread) | - | ~x1 on average, not per shower |
  | Background | x0.5 | - | ~x2 | ~x1 |

## Minor, not addressed: **Open**

- **Best-layout selection set**: still the end of the Stage-1 pool (`04_optimize_lbfgs_kernel.py:453-459`), which
  overlaps the reconstruction training. Only the choice of the best step is biased; paper scores are held out.
- **Noise handling, 03c vs Stage 4** (audit, not re-checked): 03c never applies the kNN gate; Stage 4 defaults to the
  clean reconstruction with background on; `t_fallback` fixed at 27.7 us in Stage 4 (`:834`), computed in 03c.
- **Stale docs**: `kernel.py:35` says nothing backprops through the kernel; `kernel.py:40` and `dual.py:16` cite
  THEORY §3.6; merge-conflict markers in four TAMBO-opt `conf/allshowers` yaml files.

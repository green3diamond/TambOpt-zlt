# TambOpt audit against TamboSim: what was tried and what could help

Status of the items in `TambOpt Audit Against TamboSim.pdf` (17 Nov 2026), as of 23 Nov 2026. The audit reviewed
`master` (69c04f6), not `cleanup` or `timing-model`, so parts of it are out of date. "Tried" is from the work done
so far; the items were not re-checked one by one in the code.

## Big findings after the 23 Nov checks

- **Still open and large**: physics items 1-3 (shower energy and decay class, magnetic field, atmosphere; need a new
  corpus) and the detector count model (area 2x, photons, energy cuts, incidence angle). Malata numbers are not rate
  weighted.
- **New**: on Malata the swgo_trigger gradient is mostly batch noise even at 5,120 showers per step (batch-to-batch
  cosine 0.17-0.25). Those optimizations move little from their start, consistent with no layout beating the grid.
- **Background rate**: conservative (~2x high for all charged particles). At the muon rate 1 us results barely change
  (rank correlation 0.96-1.00); 10 us results change strongly on Malata (~3 % -> 15-19 % within 1 deg, rank
  correlation 0.46-0.57). `docs/background_rate_muon_rescore.md`.
- **Checked, no problem**: trigger-gradient distance part (2-3 %), axis reference (< 1 % beyond 10 m), snap,
  surrogate paths, counting table for hadronic decays.

## Already tried or largely covered

| Audit item | What was done | Result |
|---|---|---|
| No timing gradient (`delta_l` detached) | `timing-model` branch: carry along the axis attached, mean-delay time model (`TAMBOPT_TIME_MODEL=mean`) | **Done**: all main-text optimizations of papers 1 and 2 rerun with it (`_m5k`: mean time without noise, 5,120 showers per step); batch gradients agree (cosine 0.66-0.76, flow matching). **Open**: the swgo_trigger gradient on Malata is still mostly batch noise at 5,120 showers (batch-to-batch cosine 0.17-0.25, `trig_grad_r_check.py`) |
| Signal unit (kinetic energy, not counts) | Counting model: expected counts mu per species, Poisson draws, mu >= 1 hits, trigger on mu (merged 23 Oct) | **Partly done**: trigger and scoring; the DeepSets reconstruction still reads E |
| Thresholds far below one particle | Hits and trigger use mu >= 1 | **Partly done**: scoring; the reconstruction inputs and the kNN gate still use the E thresholds |
| Optimized and scored with the same reconstruction (layouts can exploit quirks of the DeepSets network) | CORSIKA 8 columns scored with a separately fine-tuned DeepSets and with the analytical fit (no training) | **Partly done**: the CORSIKA 8 columns are independent checks; the flow-matching columns still use the network the layouts were optimized against |
| No validation of generated showers | Paper 1 Fig. 8 (CORSIKA 8 vs flow matching vs surrogate at the detectors); CORSIKA 8 scoring in paper 2; `scripts/paper2/time_residual.py` | **Done**: generator fidelity checked against CORSIKA 8. Whether the CORSIKA 8 setup itself matches TamboSim is the physics-realism items below (shower energy, magnetic field, atmosphere) |
| Utility credit for undetected showers | SWGO trigger utility is the main one in paper 2 | **No change needed**: the original utility's 0.27 credit is a property of that utility. **Open**: two-tier has no detection term; three-tier's "detected" still uses E > 1 in kernel units (both in the paper 2 main text) |
| Detector area (2 m^2 vs TamboSim's 2x2 m box: 4 m^2 top face plus sides) | Papers use 2 m^2 only (the 20 m^2 runs were dropped from paper 2) | **Open**: counts and background 2x low against TamboSim; 2 m^2 is stated in paper 2. The 20 m^2 runs are **removed**; box shape and incidence angle not modelled |
| Run 19 phase space | 12 km plane framed as a geometry study; Malata uses real tau decays | **No change needed**: framed; no physical rate weighting. **Open** for Malata: headline numbers are unweighted means over the injection spectrum (no `oneweight`) |

## Could help the optimizer: cheap, not tried

- **Trigger gradient, distance part** (`axis_distance` under `no_grad`, `background.py:137`): **No problem** (checked
  23 Nov, `trig_grad_r_check.py`, Malata, 4 x 5,120 showers, grid and the _m5k swgotrig layout). With r attached the
  gradient changes by 2-3 % (cosine 0.9996), U is identical; batch-to-batch cosine is 0.17-0.25. Not fixed.
- **Non-differentiable snap to the mesh** (`placement.py:42-65`): **No problem** (checked 23 Nov). The snap moves
  only detectors more than `max_gap` (2x the mean centroid spacing) off the mesh, and runs after each Adam step and
  after each L-BFGS chunk. Every chunk builds a new `LBFGS` (`04_optimize_lbfgs_kernel.py:695`), so **a snapped
  position never enters a curvature pair**; for Adam it acts as a projection onto the mesh.
- **Two surrogate-response paths disagree** (`04_optimize_lbfgs_ensemble.py` via `dual.py` vs the kernel script's
  `SurrogateResponse`): **No problem** (checked 23 Nov). All surrogate runs (`_sur`, `sur5*`) come from
  `04_optimize_lbfgs_kernel.py --response surrogate` (`kernel_opt_run19.sh`, `run20_quick_tests.sh`); the ensemble
  script is used only by two diagnostics.

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
the same showers); absolute efficiencies are optimistic, per tau decay at most 0.83x. Paper 2: one paragraph in
`chapters_2/02_methodology.tex` after the pipeline summary (line 14), optionally one caveat line in `04_conclusion.tex`:

> Each shower is generated with the full tau energy and a shower type (electromagnetic or hadronic) drawn with equal
> probability. Real tau decays pass on average about half of the energy to neutrinos, are about 65 % hadronic and
> 18 % electromagnetic, and in 17 % of the cases decay to a muon, which produces no detectable air shower. Our showers
> are therefore on average about twice as bright as those from the same tau energy, and efficiencies per tau decay are
> at most 0.83 times those per shower given here. Since every layout is evaluated on the same showers, these
> simplifications shift all layouts alike; the conclusions concern the relative performance of layouts, not absolute
> rates. Conditioning the generator on the visible decay products is left for future work.

Items 2 and 3, same place, right after it (field distortion depends on direction, so "alike" holds only because every
layout sees the same corpus; not tested whether the ranking changes; item 3 partly offsets item 1):

> The CORSIKA 8 training showers use a geomagnetic field that does not match the TAMBO site in direction and strength,
> and a standard atmosphere with showers starting at 2.5 km instead of at the decay altitude. The field mainly affects
> the lateral shape and azimuth of showers travelling across it; the denser air makes showers about 10-20 % shorter and
> narrower than at the site, so footprints and trigger efficiencies are, if anything, underestimated. Both act on all
> layouts alike and change absolute numbers rather than the comparison between layouts; regenerating the training
> showers with the site's field and atmosphere is left for future work.

These change absolute numbers; whether they change which layout wins is unknown. As the audit says, they should
be fixed before further optimizer work, because they invalidate the corpus.

## Worth checking for paper 2

- **Background rate**: **No change needed** (keep as conservative). The 1,826 /m^2/s is SWGO's all-particle rate at
  ~4.8 km for water Cherenkov tanks (`swgolo148.cpp:329-331`, HAP-2024-004), applied here as muons. The audit's 4-6x
  compares it with muons only (300-450 /m^2/s at 3-4 km). Scintillators see every charged particle as ~1 MIP, and at
  3-4 km e+- are comparable to muons: all charged ~700-1,000 /m^2/s (rough estimate), so the current rate is ~2x high,
  less on tilted panels. Paper 2: keep 1,826 as a conservative upper bound (above TAMBO's altitude); quote the muon-rate
  rescoring as the lower bound. 1 us results barely depend on the rate; 10 us results do strongly. Muon-only
  rescoring: `docs/background_rate_muon_rescore.md`.

- **How area, shower energy and background rate combine** (vs TamboSim):

  | | Area (2 vs ~4 m^2) | Shower energy (tau, not visible) | Background rate (all charged) | Net |
  |---|---|---|---|---|
  | Signal | x0.5 | x2 (mean; wide spread) | - | ~x1 on average, not per shower |
  | Background | x0.5 | - | ~x2 | ~x1 |

## Minor, not yet addressed: **Open**

- **Best-layout selection set**: still the end of the Stage-1 pool (`04_optimize_lbfgs_kernel.py:453-459`), which
  overlaps the reconstruction training. Only the choice of the best step is biased; paper scores are held out.
- **Noise handling, 03c vs Stage 4** (audit, not re-checked): 03c never applies the kNN gate; Stage 4 defaults to the
  clean reconstruction with background on; `t_fallback` fixed at 27.7 us in Stage 4 (`:834`), computed in 03c.
- **Stale docs**: `kernel.py:35` says nothing backprops through the kernel; `kernel.py:40` and `dual.py:16` cite
  THEORY §3.6; merge-conflict markers in four TAMBO-opt `conf/allshowers` yaml files.

## Suggested order

1. Decide on physics items 1-3 and the detector count model (area, photons, cuts), which decide whether the corpus
   has to be regenerated. State in paper 2 that the background rate is a conservative upper bound.

# Run 18 — South-facing 6 km plane, shower-frame kernel, all stages retrained

Status (2026-10-04): code done on branch `run18-showerframe` (tests: `tests/test_shower_frame.py`,
`tests/smoke_run18_kernel.py`); inputs written; pipeline submitted as a dependency chain
(`slurm/run18_pipeline.sh` gen → {build → recon, `kernel_bw_run18.sh`} → 6 × `kernel_opt_run18.sh` → score + core test).
Smoke finding: with circular showers most widths sit at the 50 m cap (points spread in 2-D, not on a line).

## Context
Run 17's diagnostics (2026-10-04) found five problems with the detector-response kernel and its inputs:
- **Layer depth:** placement puts each C8 layer one layer short (layer l is really at (l+1) × 500 m).
- **Global frame:** the kernel matches detectors to shower points in global North–Up with East slabs. That is only correct for horizontal showers heading West.
- **Quantized times:** arrival times jump in steps of one C8 layer (about 1.6 μs).
- **Vertical spread:** C8 never recorded the vertical sideways offset of each particle.
- **B-field shift:** C8 showers within ~30° of the simulation's magnetic field (+x, our East/West) are shifted sideways.

Run 18 (never started; this replaces the draft in `docs/run18-plan.md`) applies the user's decisions:
- **Layer depth:** fixed.
- **Kernel:** shower frame, with times interpolated between planes.
- **Vertical spread:** circular showers.
- **Widths:** capped at 50 m.
- **Geometry:** the run 17 plane generator, but the plane faces South like Malata, so showers head North, away from the field.
- **Showers:** zenith 60–100°, none within 30° of the field.
- **Sizes:** the run 18 sizes (6 × 6 km plane, 100 detectors, 210,600 events, decay box scaled with the plane).
- **Stages:** all retrained except Stage 2, which is skipped.
- **PDF first:** before starting, a new PDF chapter with the remaining run 17 diagnostics.

Branch: `run18-showerframe`, from `run17-shower-geometry`, in the main checkout. Commit after each step.

## Steps
0. **PDF chapter** (`figures/make_findings_pdf.py`, plus the repo copy in `plots/analysis/findings_pdf/`). New chapter "Run 17 diagnostics: fixed kernel", guarded on file existence:
   - `kernel_frame/kernel_schematic.png`
   - The array-size test redone: used vs fixed vs interpolated times (`array_size_fixed_kernel/*_cap50.png`)
   - Recon ablation table (`array_size_ablation_*.json`) and the time-quantization finding
   - A "Decisions for run 18" text page
   - Rebuild with `figures/build_findings_pdf.sh`; check text widths as before.
1. **Kernel** (`modules/showers/kernel.py`):
   - `GetCounts_planeaware` accepts detector coordinates as `(n_det,)` (unchanged) or `(B, n_det)`, the latter broadcast as `(B, 1, n_det)`.
   - New `t_interp=False` argument: `t_eff = t_p − Δl × LAYER_DX / c`, clamped ≥ 1e-12. Port it from `counts_interp` in `plots/analysis/run17/array_size_single_shower/array_size_shower.py`.
   - `adaptive_bandwidths` is unchanged (it groups by round(col2), which becomes the C8 layer + 1).
2. **Placement** (`modules/data/dataset_builder.py`):
   - New `place_clouds_shower(clouds, positions, dirs, seed_ids)` → points `[t1, t2, l + 1, e, t]` plus frames `(B, 12) = (p, d, e1, e2)`, with:
     - `e1 = (−d_N, d_E, 0)` normalised, `e2 = d × e1`
     - `t1 = x·e1_E + y·e1_N`
     - `t2` = the t1 of another point from the same (event, species, layer), paired at random. The randomness comes from a counter-based hash of (event id, species, point index), so it's vectorised and any slice of events reproduces exactly the same t2 in Stage 1, the Stage 4 pool and scoring. No cache file is needed.
     - Same 5-column layout as today, so `GetCounts_planeaware` and `adaptive_bandwidths` keep working on it. Padding rows (e = 0) stay zero.
   - Normalise `dirs`; `_CorpusMeta.dirs` isn't normalised today.
   - `compute_labels_batch(..., frames=None)`: with frames, detectors become `r = (E, N, surface(N, E)) − p`, then `s = r·d`, `t1 = r·e1`, `t2 = r·e2` per event, so `(B, n_det)`. These stay differentiable in (E, N).
   - Constant `KERNEL_FRAME = "shower"`, plus `KERNEL_T_INTERP = True` in `modules/constants.py`. `compute_labels_batch` raises if `KERNEL_FRAME == "shower"` and no frames are given, so old global-frame paths can't mix in silently.
   - Fix `place_clouds_enu` to (l + 1) as well, for the plots and older scripts.
3. **Thread the frames through:**
   - Stage 1: `_load_clouds` / `_label_chunk` (use `meta.positions` and `meta.dirs` for each chunk).
   - `true_utility.load_events`: also return the frames, with the same non-finite filter.
   - `KernelDualLabels` and both `CachedKernel` classes (`04_optimize_lbfgs_kernel.py`, `diagnostics/gate_scan.py`).
   - `MainPool.clouds` / `load` and `precompute_bandwidths` (the width caches `kernel_bw_k8_<species>.npy` are rebuilt in the new frame).
   - Event chunking stays as it is; frames are sliced alongside the clouds.
4. **No-surrogate paths:**
   - `03_train_recon_deepsets.py --recon-input kernel` skips `load_dual_surrogate` / `compute_fnn_predictions` and plots with `plot_recon_on_kernel`.
   - `diagnostics/score_layouts.py` loads the recon only (`build_recon_from_ckpt`).
5. **Inputs** (`scripts/17_make_plane_inputs.py`):
   - New options: `--side` (default 3000), `--facing {east,south,north}` (default east, so run 17 stays reproducible), `--group`, `--n-grid` (default about side / 136 + 1).
   - `--zenith 60 100`.
   - `--min-angle-b 30`: reject if arccos|d_x| < 30°.
   - Top-up loop so exactly `--n-events` are written.
   - Generalised geometry with facing f and strike g; decay box on the f side along [500, 500 + side]; asserts written as `t·f + U = 0` and `dec·f + U > 0`.
   - Run: `--side 6000 --facing south --n-events 210600 --group plane45s --out-dir <run18>/inputs`.
6. **Constants for run 18:**
   - `RUN_LOCATION = detector_optimization_v6/18_plane45S_6km_showerframe_200k`
   - `GEOMETRY_PATH` and `TAU_WHOLESKY_PATH` point to its `inputs/`; `GEOMETRY_GROUP = plane45s`
   - `NUM_SHOWERS = 210_600`, `KERNEL_INTERP_MAX` default 50
7. **Pipeline** `slurm/run18_pipeline.sh` (from `run17_pipeline.sh`):
   - Guard on run 18, the shower frame and cap 50; `--mail-type=FAIL`.
   - Modes:
     - `gen`: Stage 0
     - `build`: Stage 1
     - `recon`: Stage 3, kernel, `--adam-patience 50`, plus plots
     - `score`
   - `kernel_bw_run18.sh` (array 0–2, after `gen`).
   - `kernel_opt_run18.sh`: 6 runs (current, two-tier λ 0 and λ 0.3, each from grid and center starts); lr 10, 20k steps, 512 events per step, 1,024 scoring events. Uses a kernel two-tier JSON: 5° / 0.2, widths 1° / 0.05, refs 20° / 0.5.
   - Core test copied to `<run18>/diagnostics/core_test/`, with the hard-coded run 17 paths and the Up = −East core formula updated.
   - Chain: gen → {build → recon, widths} → 6 × opt → {score, core test}. END mail on the last job.

## Runtime (run 17 timings × 2 events)
Stage 0 ~10 h → Stage 1 ~3 h → Stage 3 ~3 h (in parallel with widths ~6 h) → kernel-in-the-loop ~13–25 h. The (B, n_det) frames and the 2-D width search may slow things somewhat. Total ~30–40 h plus queue waits. Disk ~105 GB for the shower cache plus ~30 GB of width caches.

## Verification
- `python tests/test_interp_kernel.py`, with new tests:
  - **(B, n_det) broadcasting:** identical to the `(n_det,)` path when every row is the same.
  - **Equivalence:** for an exactly horizontal, West-going shower with t2 = 0 and no time interpolation, shower frame = global frame + layer fix.
  - **Time interpolation:** a detector halfway between planes gets t_p + 250 m / c.
  - **Circular t2:** deterministic per event id, and the same per-layer distribution as t1.
- Inputs: asserts; `plane_primaries.png` (plane faces South, showers head North); azimuth and zenith histograms; exactly 210,600 events; no shower with arccos|d_x| < 30°.
- Smoke before submitting:
  - Stage 1 on ~2 chunks; widths on 512 events (max 50 m).
  - `03 --epochs 1 --lbfgs-iters 2` with FNN_FOLDER empty; `score_layouts` on 2 layouts.
  - `04 --epochs 5` from the pool.
  - Compare one event's footprint with `kernel_frame_issue.py` (shower frame).
- After the run: Stage 3 val loss and conditional plot; kernel-in-the-loop logs reach 20k steps; scores JSON; core-test figures; PDF chapter for run 18.

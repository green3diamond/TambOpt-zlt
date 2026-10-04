# Run 18 plan — 6 km plane, 2× showers, kernel widths capped at 50 m, kernel in the loop

Status: planned, not started (2026-10-04). Predates the shower-geometry checks of the same day
(`plots/analysis/run17/{array_size_*,zenith_cap100,kernel_frame}`): the layer depth in `place_clouds_enu`
is one layer short, the kernel matches in the global frame rather than the shower's, arrival times are
quantized by layer, and C8 showers within ~30° of its magnetic field (+x, our East/West) are shifted
sideways. Update this plan for those before running it.

## Context
Run 17's width-cap test (`17_plane45_interp_hurdle_100k/diagnostics/core_test_bwcap50/`) showed
that ~25% of every layout's utility, and all of the optimized layouts' lead over the grid, came
from weak signals spread by kernel widths of hundreds of metres. That test used a recon trained
on uncapped signals. Run 18 does it cleanly: labels (Stage 1) and a kernel-only recon (Stage 3)
built with widths capped at 50 m — no Stage 2 surrogate — then the 6 kernel-in-the-loop
optimizations (current, two-tier λ 0, λ 0.3 × grid / center starts), scored with the grid and the
8 hand-made layouts. Also twice the showers and a larger plane: 6 × 6 km (45°, facing East),
100 detectors, decay box scaled with the plane.

Kernel geometry, for reference: after `place_clouds_enu` the kernel's transverse coordinates are
(North, Up) and its "planes" are 500 m East slabs (`z_cont`). Each point's width is the distance
to its 8th nearest point in the same shower, species and East slab (North–Up distance), floored at
10 m — independent of detector positions, so the width cache stays valid while detectors move.

## Branch
`run18-plane6km` from `run17-bwcap50` (has the width cap and `plots/analysis`), checked out in
the main folder. Commit after each step.

## Steps
1. **Inputs** — `scripts/17_make_plane_inputs.py`: replace the fixed `SIDE` / `CUBE` with
   `--side` (default 3000, so run 17 is reproducible) and derive the box from it
   (East [500, 500 + side], North ±side/2, Up [0, side]); `--n-grid` default scales to keep
   ~136 m mesh spacing (45 for 6 km). Run with `--side 6000 --n-events 210600 --out-dir
   <run18>/inputs` (200k train after the 5% holdout). Same cuts: distance ≥ 1 km, zenith 60–120°.
   The plane's East extent (4.24 km) sits inside the 24 × 500 m layers from EAST_ENTRY = 1500.
2. **Constants** (`modules/constants.py`): RUN_LOCATION →
   `detector_optimization_v6/18_plane45_6km_bwcap50_200k`, GEOMETRY_PATH / TAU_WHOLESKY_PATH →
   its `inputs/`, `NUM_SHOWERS = 210_600`, `KERNEL_INTERP_MAX` default 50 (env override kept), so
   every kernel caller (Stage 1 `compute_labels_batch`, width precompute, `CachedKernel`
   scoring, `true_utility`) uses the cap without per-job environment variables.
3. **No surrogate anywhere** (three small code changes, no-ops when a surrogate exists). Training
   is unchanged: run 17's kernel recon already trained on kernel labels only
   (`build_recon_inputs` returns E/T.pt for `kernel`); the surrogate was loaded just for a
   diagnostic print and the surrogate-input plot. Without Stage 2 those loads would crash.
   - `scripts/03_train_recon_deepsets.py`: with `--recon-input kernel`, skip
     `load_dual_surrogate` / `compute_fnn_predictions` and train on `load_kernel_responses`
     directly; the end-of-run plot calls `plot_recon_on_kernel` instead of
     `plot_recon_only(fnn=dual, ...)`.
   - `scripts/diagnostics/score_layouts.py`: load the recon only (`build_recon_from_ckpt`, as
     `04_optimize_lbfgs_kernel.load_recon` does) instead of `load_models`, which loads the FNNs.
   - Plots: `plots/training/02_nn_target_vs_pred.py --kernel-only`.
   `04_optimize_lbfgs_kernel.py` already never loads the surrogate.
4. **Pipeline** `slurm/run18_pipeline.sh` (from `run17_pipeline.sh`; constants guard on run 18
   and cap = 50; per-mode status file; `--mail-type=FAIL`):
   - `gen`: Stage 0 (`00_generate_data_dual_species.py`)
   - `build`: Stage 1 (`01_build_dataset_northeast.py`)
   - `recon`: Stage 3 `03_train_recon_deepsets.py --recon-input kernel --adam-patience 50` as the
     run's production recon (default `_deepsets` folder), kernel conditional plots, and the
     kernel two-tier JSON (5°, 0.2; widths 1° / 0.05, refs 20° / 0.5 — fixed, no calibration)
   - `score`: `scripts/diagnostics/score_layouts.py` on the 6 kernel layouts + starts
5. **Kernel jobs** — `slurm/kernel_bw_run18.sh` (width cache, array 0-2; needs only the Stage 0
   cache, so it runs after `gen` in parallel with `build` → `recon`) and
   `slurm/kernel_opt_run18.sh` (from `kernel_opt_run17.sh`: lr 10, 20k steps, 512 events per
   step, 1,024-event scoring set; default recon folder), 6 jobs after `recon` + widths.
6. **Core test** — copy `plots/analysis/run17/core_test/*` to `<run18>/diagnostics/core_test/`
   (drop `--recon-dir`: the kernel-only recon is the default), score after the 6 runs; split
   figures (cores / hybrids / kernel-optimized, grid first). Copies also in `plots/analysis/run18/`.
7. **Chain** (gpu_requeue; mail on FAIL, END on the last job):
   `gen` → {`build` → `recon`, widths} → 6 × kernel opt → {`score`, core test}.

## Runtime (from run 17, doubled where it scales)
Stage 0 ~9 h, then Stage 1 ~3 h + Stage 3 ~3 h ∥ widths ~6 h, kernel opt ~12.5 h, scoring
~15 min → ~28 h of compute on the critical path, plus queue waits. Disk: shower cache ~105 GB
(the group had ~39 TB free on 2026-10-04).

## Verification
- Inputs: the script's asserts (targets on the plane, distance ≥ 1 km, decays in front);
  `plane_primaries.png`; 210,600 events; `load_tr_mountain` bbox ≈ 6 km (North) × 4.24 km (East).
- Cap: `scripts/diagnostics/bandwidth_histogram.py` on a few hundred events → max width 50 m;
  `tests/test_interp_kernel.py` still passes.
- Surrogate-free paths: before submitting, run `03_train_recon_deepsets.py --recon-input kernel
  --epochs 1 --lbfgs-iters 2 --out-suffix smoke` and `score_layouts.py` on 2 layouts against
  run 17 with FNN_FOLDER pointed at an empty folder — both must finish without the FNNs; delete
  the smoke folder after.
- Stage 1 log: detectors with ≥ 1 count vs run 17; Stage 3 val loss and kernel conditional plot;
  kernel-opt logs reach step 20,000; scores JSON has all 6 + starts.

# Run 19: 12 × 12 km plane, 421,200 showers, 100 detectors, relative-time recon

Status (2026-10-07): branch `run19-12km` (from `run18-reltime-recon`); inputs written; jobs next.

## Context
Run 18 (6 × 6 km South-facing 45° plane, 210,600 showers) showed that even coverage beats every concentrated
layout, and that the recon's resolution stops improving at ~6 lit detectors:
- zoo recon: median ~2.65°;
- relative-time recon: ~2.45°;
- within 1°: 12–16% at best.

Run 19 repeats the experiment on a plane twice as large in each direction.

## Decisions (user, 2026-10-07)
- **Plane:** 12 × 12 km, 45°, facing South. Same generator as run 18:
  `scripts/17_make_plane_inputs.py --side 12000 --box-depth 6000 --max-dist 11500 --facing south --zenith 60 100 --min-angle-b 30 --group plane45s12 --seed 19`.
  - **Decay box:** depth 0.5–6.5 km in front, as in run 18, but 12 km wide and 12 km high.
  - **Distance cap:** decays more than 11.5 km from their target are redrawn. The C8 showers cover only 12 km (24 planes × 500 m).
  - **Why both:** with a 12 km-deep box, half the decays were past 12 km; with the 6 km box alone, 24% still were. The targets stay uniform on the plane, and no event is dropped.
- **Showers:** 2× run 18, i.e. 421,200. That is half of run 18's shower density, a compromise between cost and signal.
- **Detectors:** 100, the same count. On 4× the area they sit ~1.2 km apart (run 18: ~0.6 km), so far fewer detectors are lit per shower.
- **Folder:** a new run folder, `detector_optimization_v6/19_plane45S_12km_showerframe_400k/`. Run 18 is untouched.
- **Recon:** the relative-time input (`--input-mode reltime`, 9 features). It beat the zoo recon on 96% of run 18's test layouts at 5°.
- **Scope:** recon plus layout tests. No Stage 2, no kernel-in-the-loop optimization.
- **Jobs:** run from the main checkout, no worktree. The run 18 relative-time training was stopped at its L-BFGS best (step 25, val 0.0782) before switching branches.
- **Unchanged from run 18:**
  - the shower-frame kernel, ring showers and interpolated times;
  - widths k = 8, capped at 50 m;
  - the 9-type layout zoo;
  - L-BFGS patience 50 (user change).

## Code changes (branch `run19-12km`)
- **`modules/constants.py`:**
  - `RUN_LOCATION`, `GEOMETRY_PATH` (`plane45s12.h5`) and `GEOMETRY_GROUP` (`plane45s12`);
  - `TAU_WHOLESKY_PATH` and `NUM_SHOWERS = 421_200`.
- **`modules/layouts/strategies.py`:** zoo sizes scaled to the plane.
  - `disk_a` r_max = 8 km (run 18: 4 km);
  - `grid_patch_a` side_max = 12 km (run 18: 6 km);
  - clusters stay 30–600 m, a scale set by the shower footprint.
- **`slurm/run19_guard.sh` and `slurm/run19_pipeline.sh`:** modes `gen`, `build` and `recon` (reltime).

## Steps
1. **Inputs:** done. 421,200 events.
   - Decay–target distance p5/p50/p95: 4.5 / 9.0 / 11.3 km (run 18 median: 6.5 km).
   - Zenith p5/p50/p95: 65 / 86 / 99°.
   - Every shower at least 30° from the x axis.
2. **Layout check:** 3 draws of each layout type on the 12 km plane, all on the plane (`<run19>/diagnostics/zoo/zoo_layouts.png`). Done. A Stage 1 smoke test needs Stage 0 first, so the `build` job is the first test of it.
3. **Jobs**, chained with `afterok` on gpu_requeue:
   - `gen`: ~20 h
   - `build`: ~6 h
   - `recon`: ~1–1.5 days
4. **Layout tests:** run 18's diagnostics, copied to `<run19>/diagnostics` with radii scaled.
   - grid and detector count (N = 4–100);
   - disks of N = 10/20/50/100 detectors, radius 0.1–24 km;
   - cores and hybrids;
   - clusters;
   - lit-detector resolution.
5. **PDF:** `figures/run19_12km.pdf`, compared against run 18's relative-time recon.

## Expected
About 2.5 days of compute plus queue waits, and ~210 GB of shower cache. With 100 detectors at ~1.2 km spacing, most showers light 0–1 detectors, so the shares are expected well below run 18's.

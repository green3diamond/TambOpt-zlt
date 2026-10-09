# Run 19: 12 × 12 km plane, 421,200 showers, 100 detectors, relative-time recon with weight decay

Status (2026-10-08): paused, nothing running. Branch `run19-12km` now includes `run18-reltime-recon`: weight decay,
the L-BFGS finish, `--n-det` and the test scripts. Inputs are final. To restart:
- clean the partial Stage 0 output;
- check out `run19-12km` in the main checkout, once the run 18 jobs are done;
- submit gen → build → recon → tests.

## Context
Run 18 (6 × 6 km South-facing 45° plane, 210,600 showers) found:
- **Even coverage wins** over every concentrated layout.
- **Resolution floor:** the recon's resolution stops improving at ~6 lit detectors. Median error: zoo recon ~2.65°, relative-time recon ~2.45°. At most 12–16% of showers land within 1°.
- **Density vs area:** up to ~50 detectors on 25 km², more area beats more density at 1°, 2.5° and 5°. Once the plane is full, extra detectors can only add density, and that still pays.
- **Weight decay:** AdamW with wd 0.03 gives ~1% lower val loss in ~2/3 of the Adam time.
- **reltime1** (run 18 test, 2026-10-09): same val as wd 0.03 + L-BFGS with Adam alone (0.0769); test-layout shares unchanged (±0.1 pts), so it saves training time, not accuracy.

Run 19 repeats the experiment on a plane twice as large in each direction, about 4× the area.
**Prediction:** what matters is detectors per area relative to the fixed shower footprint (~1 km across). By that measure:
- 100 detectors on the 12 km plane behave like ~25 detectors on the 6 km plane, so full coverage should still be best, but only just;
- 50 detectors behave like ~12 on the 6 km plane, so a smaller-than-full disk should win.

That is the first place where concentrating detectors should pay.

## Decisions (user, 2026-10-07 / 08)
- **Plane:** 12 × 12 km, 45°, facing South:
  `scripts/17_make_plane_inputs.py --side 12000 --box-depth 6000 --max-dist 11500 --facing south --zenith 60 100 --min-angle-b 30 --group plane45s12 --seed 19`.
  - **Decay box:** 0.5–6.5 km in front of the plane, as in run 18, but 12 km wide and 12 km high.
  - **Distance cap:** decays more than 11.5 km from their target are redrawn, because the C8 showers cover only 12 km (24 planes × 500 m). With a 12 km-deep box half the decays were past 12 km; with the 6 km box alone, 24% still were. The targets stay uniform on the plane, and no event is dropped.
- **Showers:** 421,200 (2× run 18, half its shower density).
- **Detectors:** 100, ~1.2 km apart on the full plane.
- **Folder:** `detector_optimization_v6/19_plane45S_12km_showerframe_400k/`. Run 18 is untouched.
- **Recon:** relative-time input with the dt reference at the earliest timed detector with ≥ 1 count (`--input-mode reltime1`), **AdamW weight decay 0.03**, Adam with patience 50, then L-BFGS (patience 50).
- **Scope:** recon plus layout tests. No Stage 2. Kernel-in-the-loop optimizations only if the run 18 1° runs show they are worth it.
- **Jobs:** run from the main checkout (no worktree for runs). The worktree `.claude/worktrees/run19-edit` was used for editing only, while run 18 jobs were running from the main checkout.
- **Unchanged from run 18:**
  - the shower-frame kernel, ring showers and interpolated times;
  - widths k = 8, capped at 50 m;
  - the 9-type layout zoo, with sizes scaled to the plane.

## Code (branch `run19-12km`)
- **`modules/constants.py`:**
  - run 19 `RUN_LOCATION`, geometry (`plane45s12`), `TAU_WHOLESKY_PATH`, `NUM_SHOWERS = 421_200`;
  - overrides `TAMBOPT_RUN_LOCATION`, `TAMBOPT_GEOMETRY_PATH` and `TAMBOPT_GEOMETRY_GROUP`, to run another run's tests from this branch;
  - `TAMBOPT_DATASET_TAG`.
- **`modules/layouts/strategies.py`:** `disk_a` r_max 8 km and `grid_patch_a` side_max 12 km (run 18: 4 / 6 km). Clusters stay 30–600 m.
- **`scripts/03_train_recon_deepsets.py`:** `--input-mode reltime1`, `--weight-decay`, L-BFGS patience 50.
- **`slurm/run19_guard.sh` and `slurm/run19_pipeline.sh`:** modes `gen`, `build`, `recon` (reltime1, wd 0.03) and `tests`.
- **Test suite**, `plots/analysis/run19/tests/`. Outputs go to `<run19>/diagnostics/tests/`.
  - `make_layouts.py`:
    - grids, N = 4–100;
    - centred disks, N = 10/20/50/100, radius 0.1–12 km;
    - cores 100–500 m and hybrids, N = 20/50/100;
    - clusters, K × M with r 50–1000 m.
  - `score_all.py`: one pass with the run's recon on the 1,536 held-out showers.
    - per layout: within 1/2.5/5°, good at 5°, median error;
    - per shower × layout: lit detectors, their mean distance, nearest-neighbour spacing, baseline, and the angle error.

    Each layout's own N detectors go to the recon, with no padding.
  - `analyze.py` (CPU), which writes:
    - disks vs radius at 1/2.5/5°;
    - the density-vs-area grid (`density_area.json`);
    - resolution vs lit detectors;
    - accuracy vs lit-detector spacing;
    - best per family;
    - the layout catalogue (grouped by N, ranked by share within 1°).
  - `run_tests.sh`: runs those three scripts plus `detector_count_grid.py`, skipping finished steps.
  - Smoke-tested on run 18 data (job 51257524).

## Steps
1. **Inputs:** done.
   - Decay–target distance p5/p50/p95: 4.5 / 9.0 / 11.3 km (run 18 median: 6.5 km).
   - Zenith p5/p50/p95: 65 / 86 / 99°.
2. **Layout-zoo check** on the 12 km plane: done (`<run19>/diagnostics/zoo/zoo_layouts.png`).
3. **Restart, once the run 18 jobs from the main checkout are done:**
   - check out `run19-12km`;
   - remove `<run19>/v6_run_00` (partial Stage 0) and `pipeline_status_run19_*.json`;
   - submit gen (~20 h) → build (~6 h) → recon (~1–1.5 days) → tests (~1 h), chained with `afterok` on gpu_requeue.
4. **PDF:** `figures/run19_12km.pdf`. Same structure as the run 18 one, compared against run 18's relative-time recon.

## Expected
About 2.5–3 days of compute plus queue waits, and ~210 GB of shower cache. At ~1.2 km spacing, most showers light 0–1 detectors, so the shares should be well below run 18's. The density-vs-area grid tests the prediction above.

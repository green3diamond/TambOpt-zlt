# `landscape_analysis/` — what the utility surface looks like around a layout

Run by hand, not wired into any pipeline stage. Three questions, plus a tail of
checks that ended up here without belonging.

`common.py` holds the setup every script needs: the frozen surrogate pair, the
mountain, the primary batches, and layout scoring. `layouts.py` resolves which
layout is analysed and where results go. Neither is a script.

## 1. Criticality and redundancy — which detectors matter

| file | role |
|---|---|
| `detector_removal_analysis.py` | the core leave-one-out sweep; **writes `detector_removal_results.json`** |
| `detector_removal_batch_robustness.py` | is the critical core stable across primary batches |
| `detector_removal_cross_layout_check.py` | does the core transfer to a differently-initialised optimum |
| `detector_removal_threshold_rescaled.py` | is the collapse an artefact of the fixed firing threshold |
| `critical_detector_geometry.py` | what is physically special about the critical detectors |

## 2. Local geometry around an optimum

| file | role |
|---|---|
| `detector_grid_scan.py` | move ONE detector over a grid, map U |
| `detector_grid_scan_broad.py` | the same for many detectors (much slower) |
| `detector_grid_scan_random_control.py` | control: is the flatness special to an optimum |
| `full_space_2d_slice.py` | random 2D plane through the full 200-dim space |

`full_space_2d_slice.py` takes `--step-range`. The default 400m sweep pushes
boundary detectors past the mesh snap tolerance (~160m), so part of what it
measures is snapping; re-run with `--step-range 100 --out-prefix
full_space_2d_slice_fine` for a cleaner reading.

## 3. Mode connectivity — one basin or many

`mode_connectivity.py`, the linear path between two optima.

## 4. Coverage shape

`ring_vs_grid.py` — does a boundary-only ring beat a uniform grid.

## Not landscape analysis, parked here

`fired_val_check.py`, `finetune_regression_check.py`,
`regenerate_3d_surface_plots.py`, `verify_seed_layout_extension.py`.

## Coordinates

`x` is East and `y` is North, everywhere, matching the ENU convention of the h5
files and what stage 4 writes into `layout_best.pt`. `project_to_mountain_ne`
takes `(mountain, E, N)` in that order. Several bugs here came from pairing an
East array with a North quantity, so `common.bbox_center` returns `(East, North)`
and the path variables are named `e_*` and `n_*` for the axis they hold.

## Where results go

**Not next to the code.** `layouts.results_dir()` resolves the output directory,
default `<runs>/landscape_analysis/`, override with `$TAMBO_LANDSCAPE_OUT`.
Writing results into the source tree meant every run left the repository dirty
and the outputs were versioned alongside the analysis that produced them.

## Which layout is analysed

`layouts.py` resolves it, so no script names a path. `layouts.primary()` is the
single-layout default and `layouts.secondary()` is the second, independently
initialised optimum for cross-layout checks. Point `$TAMBO_LAYOUT_BASE` at a
different stage-4 output directory to re-run everything against another world.

## Order matters

`detector_removal_results.json` has ONE writer, `detector_removal_analysis.py`,
and four readers: `detector_removal_batch_robustness`,
`detector_removal_cross_layout_check`, `detector_removal_threshold_rescaled`,
`critical_detector_geometry`.

**Do not run these in parallel.** Run the writer first, then the readers. Running
them concurrently races the file, and the failure is silent: a reader picks up
whatever version happens to be on disk and reports it as current.

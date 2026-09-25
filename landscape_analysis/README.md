# `landscape_analysis/`: what U looks like around an optimized layout

All scripts take `--run-dir` and `--out` and write `<out>/<script>/{run.log, results.json, figures/}`. Scoring goes through `analysis_core`.

| script | question |
|---|---|
| `single_detector_scan.py` | What is one detector worth at each point of the mountain? `--mode add` adds a detector to the fixed layout (as in the SWGO layout study, arXiv 2310.01857, Fig. 14); `--mode move` relocates an existing one. Maps of ΔU, ΔU over its error, per utility term, profiles, and kernel spot checks |
| `removal_curves.py` | How does U fall as detectors are removed? Greedy most important first, greedy least important first, and random orders. Greedy picks on one half of the events and reports on the other. `--rescale-threshold` scales the reconstruction threshold with detector count |
| `detector_importance.py` | Which detectors and areas matter? Leave-one-out, knockout of square cells, correlation with geometry, consistency across chain optima. `--scan` overlays the add-mode scan |
| `loss_slices.py` | U along the line from the starting layout to the optimum, on random 2D slices where every detector moves α metres, and on the plane of the optimizer path's first two PCA directions (after arXiv 1712.09913) |

## Order

`detector_importance.py --scan` reads the add-mode `single_detector_scan.py` results, so run the scan first. Everything else is independent.

## Notes

- `layout_best.pt` is the best-scoring iterate of the optimizer, not the last row of `trajectory.pt`; `loss_slices.py` centres on `layout_best.pt`.
- Points and paths that leave the mountain are masked or reported, never snapped.

Self-tests: `python tests/analysis/test_landscape.py`.

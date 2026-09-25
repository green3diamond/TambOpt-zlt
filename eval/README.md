# `eval/`: is what the pipeline reports true?

All scripts take `--run-dir` and `--out` and write `<out>/<script>/{run.log, results.json, figures/}`. Scoring goes through `analysis_core`.

| script | question |
|---|---|
| `scoring_check.py` | Does the shared scorer reproduce the run's own true utility, and how long does one evaluation take? Run this first |
| `recon_resolution.py` | Reconstruction error in physical units (angle in degrees, energy in dex) with kernel and surrogate inputs, per layout, against a predict-the-mean baseline, with bootstrap intervals and energy bins |
| `detection_stats.py` | Which detectors light up, kernel against surrogate, and detection efficiency against energy and decay distance with Wilson intervals |
| `event_display.py` | Chosen held-out showers drawn with the layout: deposits from above and side-on, kernel and surrogate counts per detector, grid and opt_grid in one figure. Each shower is the median member of a selection rule |

The upstream evaluators in `plots/layouts/` (`true_utility.py`, `utility_vs_energy.py`, `activation_counts.py`) are used as they are.

Self-tests: `python tests/analysis/test_eval.py` and `python tests/analysis/test_scoring.py`.

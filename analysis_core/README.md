# `analysis_core/`: shared machinery for the analysis scripts

Every script in `landscape_analysis/` and `eval/` scores layouts through this package, so they all use one definition of U, one event set convention, and one output format.

| file | role |
|---|---|
| `run.py` | `Run`: frozen surrogate and recon, mountain, surface, named layouts (`grid`, `center`, `opt_grid`, `opt_center`, and the chain optima), and the on-mountain test. `--run-dir` must equal `constants.RUN_LOCATION`, because the event loader reads its paths from there |
| `events.py` | Held-out events from `plots/layouts/true_utility.load_events`, shuffled with a seed and trimmed to a multiple of the chunk size |
| `scoring.py` | `Scorer`: detector labels computed once per layout (surrogate mean or kernel), U from the unmodified `utility_of_xy` on equal chunks, paired standard errors from chunk differences |
| `stats.py` | Bootstrap and Wilson intervals at one standard deviation, and binning |
| `logs.py` | `<out>/<script>/run.log`, `results.json` with provenance (args, git commit, dirty flag, host, job id, time) |
| `plots.py` | Shared figure style, mountain footprint underlay, figure saving to `<out>/<script>/figures/` |

## Scoring rules

- The label source is always explicit: `surrogate` or `kernel`.
- The surrogate is always scored on its mean. `utility_of_xy` samples by default, which scores the recon on inputs it was not trained on when the recon was trained on means.
- U is a mean over events, so equal chunks reproduce the full-batch U exactly. Differences between layouts are paired over identical chunks.

## Checks

- `tests/analysis/` holds CPU self-tests with known answers, including broken inputs that must fail.
- `eval/scoring_check.py` must reproduce the run's own `true_utility_eval.txt` before any new analysis is trusted.

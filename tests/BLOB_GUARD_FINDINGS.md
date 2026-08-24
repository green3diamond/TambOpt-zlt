# The re-roll does not recover degenerate showers

> **Resolved.** The re-roll was removed. The Step-1 flag (`flag_blob_showers` /
> `BLOB_MEDIAN_E`) now carries the whole defense and picked up the non-finite test
> the ratio check used to own. The open policy question below still stands.

`937ad26` added a Step-0 guard that re-generates a shower whose deposited/primary
energy ratio is degenerate, reasoning that AllShowers draws fresh noise per call so
a re-run redraws. The redraw is real. **The recovery is not.**

Measured on GPU, 486 pairs per run, guard branch at `937ad26`:

| run | band (GeV) | muon flagged | recovered by 3 retries | worst ratio |
|---|---|---|---|---|
| A | 1e5 – 1e8 | 5 / 486 (1.0%) | **0** | 7.53e11 |
| B | 1e7 – 1e8 | 51 / 486 (10.5%) | **0** | inf |

Nothing recovered, either run, either band — measured residual 100%, against the
original commit message's claimed ~1e-6. Not an artifact: the worst ratio moves
between passes (7.533e11 → 7.383e11 → 7.602e11), so a genuinely different shower is
drawn each time and every one is degenerate.

## What this means

**Degeneracy is a property of the primary, not of the latent draw.** Given the same
conditioning the model reliably produces a degenerate shower, so the question is not
"how many retries" but "what do we do with a primary this model cannot render".
`MAX_BLOB_RETRIES = 3` bought 4x the GPU cost on failing rows and nothing else.

Re-rolling was chosen because corpus rows are paired events with row-indexed species
and position sidecars, so a row cannot simply be dropped — a constraint that still
holds. **Still open:** the coherent alternative is dropping the whole *event* across
every species block, keeping the sidecars aligned. Today those rows stay in the
corpus, flagged and excluded from the fit. A policy decision, not a code fix.

## Why the rate rises with energy

All three checkpoints were trained on the same 130,000 primary energies
(element-identical across `combined_{muons,electrons,photons}.h5`, which differ only
in shower content):

| decade (GeV) | showers | share |
|---|---|---|
| 1e5 – 1e6 | 123,559 | 95.0% |
| 1e6 – 1e7 | 6,419 | 4.9% |
| 1e7 – 1e8 | **22** | 0.017% |
| 1e8 + | **0** | — |

Support ends at 4.9e7 (log10 7.690), cross-checked against the `cond_trafo`
StandardScaler in the checkpoints (log10 mean 5.437 / std 0.310) — two independent
paths, same answer. `LOG_E_MAX = 7` sits just inside that support but rests on ~22
showers, which predicts run B: 10.5% degenerate over 1e7–1e8 against 0.39% over the
production band.

**Do not raise `LOG_E_MAX` to 8 without retraining the generators.** The point caps
break first — after their full 10 retries the anti-clip guard still left 53/486
electron, 27/486 muon and 52/486 photon showers clipping, one muon predicting 56,926
points against a 25,088 cap. The caps (4096 / 25088 / 8064) are sized for the
trained range.

## Corpus-wide context

From a full scan of the 07 corpus (1,144,328 rows): **no gap** between the good and
bad populations under either candidate discriminator. Muon `total/E_prim` runs
continuously from the bulk out to `inf` — 1,933 rows above 30, 487 above 1e3, 157
above 1e6 — and `median_e` behaves the same way. Any cut clips some legitimate tail:
a policy choice about what the surrogate is trained on, not a boundary in the data.

## Reproduce

```bash
# per-row statistics over a whole corpus (CPU, ~6 min for 1.1M rows)
sbatch -p test -c 4 --mem 48G -t 4:00:00 \
  --wrap "python tests/scan_corpus_blobs.py --chunk 256 --out blob_scan.npz"
```

Both GPU runs above OOMed on the holdout pass *after* completing all three species,
so the guard measurements are complete; no corpus was written.

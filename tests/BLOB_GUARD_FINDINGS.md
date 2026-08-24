# The re-roll does not recover degenerate showers

`937ad26` added a Step-0 guard that re-generates a shower whose deposited/primary
energy ratio is degenerate, reasoning that AllShowers draws fresh noise per call
so a re-run redraws. The redraw is real. **The recovery is not.**

Measured on GPU, 486 pairs per run, guard branch at `937ad26`:

| run | band (GeV) | muon flagged | recovered by 3 retries | worst ratio |
|---|---|---|---|---|
| A | 1e5 – 1e8 | 5 / 486 (1.0%) | **0** | 7.53e11 |
| B | 1e7 – 1e8 | 51 / 486 (10.5%) | **0** | inf |

Not one shower was recovered, in either run, at either energy band. The commit
message's "3 retries leaves ~1e-6 residual" is wrong by six orders: the measured
residual is 100%.

The retry log shows why this is not a measurement artifact — the worst ratio
moves between passes, so a genuinely different shower is drawn each time, and
every one of them is degenerate:

```
[blob 1/3] re-generating 5/486 muon shower(s) ... (worst 7.533e+11)
[blob 2/3] re-generating 5/486 muon shower(s) ... (worst 7.383e+11)
[blob 3/3] re-generating 5/486 muon shower(s) ... (worst 7.602e+11)
[blob]     5/486 muon shower(s) still degenerate after 3 retries — kept last draw
```

## What this means

**Degeneracy is a property of the primary, not of the latent draw.** Given the
same conditioning, the model reliably produces a degenerate shower. So the
question the guard should answer is not "how many retries" but "what do we do
with a primary this model cannot render".

As it stands `MAX_BLOB_RETRIES = 3` buys 4x the GPU cost on failing rows and
nothing else. It should be set to 0 (the constant already documents 0 as
"disable") until the policy is decided.

Re-rolling was chosen because corpus rows are paired events with row-indexed
species and position sidecars, so a row cannot simply be dropped. That
constraint still holds — but the coherent alternative is dropping the whole
*event* across every species block, which keeps the sidecars aligned. That is a
policy decision, not a code fix, and is not made here.

## Why the rate rises with energy

All three AllShowers checkpoints were trained on the same 130,000 primary
energies (verified element-identical across `combined_{muons,electrons,photons}.h5`,
which differ only in shower content):

| decade (GeV) | showers | share |
|---|---|---|
| 1e5 – 1e6 | 123,559 | 95.0% |
| 1e6 – 1e7 | 6,419 | 4.9% |
| 1e7 – 1e8 | **22** | 0.017% |
| 1e8 + | **0** | — |

Support ends at 4.9e7 (log10 7.690). Cross-checked against the `cond_trafo`
StandardScaler inside the checkpoints (log10 mean 5.437 / std 0.310) — two
independent paths, same answer.

The current `LOG_E_MAX = 7` therefore sits just inside the support but rests on
~22 training showers. This is the most likely explanation for the energy
dependence of the blob rate, and it predicts what run B measured: 10.5% of muon
showers degenerate over 1e7–1e8, against 0.39% over the production band.

**Do not raise `LOG_E_MAX` to 8 without retraining the generators.** Beyond the
blob rate, the point caps break first — after their full 10 retries the
anti-clip guard still left 53/486 electron, 27/486 muon and 52/486 photon
showers clipping, with one muon shower predicting 56,926 points against a
25,088 cap. The caps (4096 / 25088 / 8064) are sized for the trained range.

## Reproduce

```bash
# per-row statistics over a whole corpus (CPU, ~6 min for 1.1M rows)
sbatch -p test -c 4 --mem 48G -t 4:00:00 \
  --wrap "python tests/scan_corpus_blobs.py --chunk 256 --out blob_scan.npz"

# guard behaviour at a given band: set LOG_E_MIN/LOG_E_MAX and RUN_LOCATION,
# then generate a small corpus and read the [blob] / [anti-clip] lines
python scripts/00_generate_data_dual_species.py --n-pairs 512
```

Both GPU runs above OOMed on the holdout pass *after* completing all three
species, so the guard measurements are complete; no corpus was written.

## Corpus-wide context

From a full scan of the 07 corpus (1,144,328 rows), for the record: there is no
gap between the good and bad populations under either candidate discriminator.
Muon `total/E_prim` runs continuously from the bulk out to `inf` — 1,933 rows
above 30, 487 above 1e3, 157 above 1e6 — and `median_e` behaves the same way.
Any cut clips some legitimate tail; it is a policy choice about what the
surrogate is trained on, not a boundary the data draws.

# AllShowers muon model: rare degenerate showers ("blobs")

The muon AllShowers checkpoint occasionally emits a shower wrong in both geometry
and energy scale. Rare, finite, and they passed every guard in the pipeline, so
they reached the training targets: a total deposit of 2e14 becomes a Step-2
`log1p(counts)` target of ~33 where normal is single digits.

Rates and the guard that now excludes them: `BLOB_GUARD_FINDINGS.md`.
Ours vs upstream, and the suspected mechanism: `UPSTREAM_COMPARISON.md`.

## Reproduce

```bash
python repro_blob_showers.py                      # diagnostic table
python repro_blob_showers.py --plot blobs.png     # + 3D comparison
```

numpy + matplotlib only — no TambOpt, showerdata, GPU or checkpoints. The three
showers live in `blob_showers.npz` (1.1 MB), kept with the run outputs rather than
git because it is data; the script defaults to

```
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/
    detector_optimization_v6/tests/blob_showers.npz
```

Off-cluster, copy it next to the script and pass `--npz blob_showers.npz`.

## What you should see

| | good_rod | blob_at_cap | blob_below_cap |
|---|---|---|---|
| points | 22,393 | 25,088 | 23,240 |
| primary energy | 2.94e5 | 9.95e6 | 4.09e6 |
| **median point energy** | **3.30** | **1.13e8** | **4.38e9** |
| total deposited | 6.24e5 | 5.73e12 | 2.06e14 |
| total / primary | 2.1 | 5.8e5 | 5.0e7 |
| top-100 points' share | 40% | 7.6% | 7.2% |

In 3D the good shower is a **rod** with the 24 observation planes visible as
discrete slabs; both blobs are structureless balls, spread over tens of km and far
off-axis.

## Why it is not the obvious things

- **Not one corrupt value.** A normal shower is spiky (top 100 points ≈ 40% of the
  energy); the blobs hold ~7%, *more uniform* than normal, top-1 share 0.002. All
  ~23,000 points are inflated by the same factor.
- **Not non-finite.** `np.isfinite` is True everywhere, so sanitization that zeroes
  non-finite points cannot see them.
- **Not only truncation.** `blob_at_cap` sits exactly at the 25,088 cap (the known
  rod→blob mode), but `blob_below_cap` has 23,240 points and was never truncated.
- **Not a code difference.** All three rows: one corpus, one Step-0 invocation, one
  checkpoint — muon rows 836, 996, 1,299. Same weights, different draw.

## Cheapest discriminator found

Median point energy: **3.3 normal vs 1e8–4e9 corrupt**. One `np.median` per shower,
better separated than any per-plane or total-energy ratio tried (those leave the two
ranges nearly touching). This is what `BLOB_MEDIAN_E` uses.

## Provenance

```
corpus  detector_optimization_v6/07_750k_primaires_meanvar/v6_run_00/
          cashed_showers_tau_dual.pt        (written 2026-07-31)
rows    573000 (good), 573160, 573463       muon block starts at 572,164
model   checkpoints/20260520_160031_Muons-Allshower
        checkpoints/20260521_043912_Muon-PointCountFM/compiled.pt
        trained on h5_files_v3/combined_muons.h5
```

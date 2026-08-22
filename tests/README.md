# AllShowers muon model: rare degenerate showers ("blobs")

The muon AllShowers checkpoint occasionally emits a shower that is wrong in both
geometry and energy scale. They are rare, finite, and pass every guard currently
in the pipeline, so they reach the training targets.

## Reproduce

```bash
python repro_blob_showers.py                      # diagnostic table
python repro_blob_showers.py --plot blobs.png     # + 3D comparison
```

numpy + matplotlib only — no TambOpt, no showerdata, no GPU, no checkpoints.

The three showers live in `blob_showers.npz` (1.1 MB), kept with the run outputs
rather than in git because it is data:

```
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/
    detector_optimization_v6/tests/blob_showers.npz
```

That is the script's default. Off-cluster, copy the file next to the script and
pass `--npz blob_showers.npz`.

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
discrete slabs; both blobs are structureless balls with no plane structure,
spread over tens of km and sitting far off-axis.

## Why it is not the obvious things

- **Not a single corrupt value.** A normal shower is spiky — its top 100 points
  hold ~40% of the energy. The blobs hold ~7%, i.e. *more uniform* than normal.
  A single bad number would give a top-1 share near 1.0; it is 0.002. Every one
  of the ~23,000 points is inflated by the same factor.
- **Not non-finite.** `np.isfinite` is True everywhere, so the existing
  sanitization (which zeroes non-finite points) does not see them.
- **Not only truncation.** `blob_at_cap` sits exactly at the 25,088 point cap, so
  it is the known "rod → blob" truncation mode. But `blob_below_cap` has 23,240
  points — under the cap, never truncated. The anti-clip re-roll could not have
  prevented that one, so truncation is at most part of the story.
- **Not a code difference.** All three rows come from the same corpus, the same
  Step-0 invocation, the same checkpoint — rows 836, 996 and 1,299 of the muon
  block. Same code, same weights, different random draw.

The suspected mechanism is the inverse energy transform (an exp of a latent)
landing far off for that sample. The generator already guards the *opposite*
extreme — its own comment notes the transform "can emit EXACTLY 0.0 for extreme
negative latents (float32 underflow)". This looks like the positive-latent
counterpart, and nothing catches it.

## Provenance

```
corpus  detector_optimization_v6/07_750k_primaires_meanvar/v6_run_00/
          cashed_showers_tau_dual.pt        (written 2026-07-31)
rows    573000 (good), 573160, 573463       muon block starts at 572,164
model   checkpoints/20260520_160031_Muons-Allshower
        checkpoints/20260521_043912_Muon-PointCountFM/compiled.pt
        trained on h5_files_v3/combined_muons.h5
```

## Rate

Not yet established. In 4,000 scanned rows per species, 9 muon showers had
total/primary > 100. A full-block scan of the electron half (572,164 rows) found
409 rows above 10× on the max-plane statistic and 9 above 1000×, so **electron is
affected too**, roughly 40× more rarely — rare enough that a 6,000-row sample
contained none. The muon full-block scan was not completed.

## Why it matters downstream

Step 1 turns these into surrogate training targets as `log1p(counts)`. A total of
2e14 becomes a target of ~33 where normal is single digits, so the per-species
muon surrogate was fit against contaminated targets.

## Cheapest discriminator found

Median point energy: **3.3 normal vs 1e8–4e9 corrupt** — eight orders of
magnitude of clean air, no threshold tuning needed, one `np.median` per shower.
Better separated than any per-plane or total-energy ratio we tried (those leave
normal and corrupt ranges nearly touching).

# Why the kernel-direct optimizer was attempted (and parked)

## Problem it was trying to solve

`eval_true_utility.py` showed a large "artifact gap" between the layout
optimizer's own objective and ground truth: it swaps only the label source
(FNN surrogate vs. the plane-aware kernel run on real point clouds) into the
otherwise-identical `opt_core.utility_of_xy`. A representative run:

```
                  surrogate-U     true-U
  baseline grid       13.5307        8.8140
  optimized          176.7188       34.4832
```

The optimizer's reported ΔU (surrogate, +163) wildly overstates the real gain
(true, +26) — the surrogate is being gamed, not just imprecise.

## Approach: bypass the surrogate entirely

If the surrogate is the problem, evaluate the kernel directly during
optimization instead of approximating it with a trained net. The blocker is
cost: the plane-aware kernel run point-by-point over the full ~100k-shower
corpus is too slow/memory-heavy to put inside an Adam/L-BFGS inner loop.

`modules_v6/voxel_corpus.py` makes this tractable by compacting each placed
shower's point cloud into (North, Up, layer) voxel bins (energy-weighted
centroid + summed energy/time per bin), shrinking the full corpus to a
GPU-resident ~7 GiB tensor. `KernelVoxelLabels` then matches the FNN's exact
`(primary_batch, xy_batch) -> (B, n_det, 2)` call signature, so
`04_optimize_kernel_direct.py` can reuse `opt_core.utility_of_xy` completely
unmodified — only the label source changes, same as the diagnostic script.
`fnn_surrogate_ne.compute_labels_batch` gained a `differentiable` flag (default
off, preserving the original behavior) so the kernel itself can be
backpropped through.

## Why it was set aside rather than finished

A rigorous, code-and-data-grounded validation pass
(`plots/validate_limitations.py`) was run against five suspected causes of the
gap. Four did not survive:

- Surrogate val R² is moderate (0.51-0.60), not ~0 as originally assumed.
- Training layouts are diverse (3000+ distinct layouts per strategy), not a
  handful.
- The current DeepSets recon is empirically confirmed permutation-invariant
  (a known problem in an older MLP-based recon does not apply here).
- The voxel-corpus NaN corruption that crashed the optimizer never reaches
  the real training labels (flagged events score exactly 0.0 in production
  E.pt/T.pt) — it was a bug introduced by the new compaction code, not a
  property of the original pipeline.

Only the measured artifact gap itself held up. Root-causing it further showed
it is not a noise/capacity floor at all: the gap is already present at the
un-optimized grid baseline (53% overstatement before a single gradient step),
and grid-like layouts are the one layout family with **zero** coverage in the
training strategies (`center_gaussian`, `uniform_random`, `latin_hypercube` —
Latin Hypercube stratifies each axis independently but explicitly
randomizes the North/East pairing, so it has no more joint/grid-like
structure than plain uniform sampling). Gradient ascent through the surrogate
then walks further into that same out-of-distribution region, compounding a
pre-existing bias rather than exploiting generic noise.

That reframes the problem as a training-distribution coverage gap, not a
surrogate-capacity problem — which two much cheaper interventions can
plausibly fix without ever needing full-corpus kernel-direct optimization:
adding a grid-like layout strategy to training, and/or restructuring the
surrogate to a mean+variance (heteroscedastic) output so its uncertainty is
explicit. Both are being tried on `master` first. This branch preserves the
kernel-direct path (code fixes applied, but the end-to-end rebuild+smoke test
was never confirmed — the job was cancelled mid-run) in case those cheaper
fixes don't close the gap and full ground-truth optimization is needed after
all.

# The blob bug is upstream's, not ours

The AllShowers generator lives in a sibling repo, `/n/home05/zdimitrov/tambo/
TAMBO-opt` — a fork of [hamzahanif2210/TAMBO-opt](https://github.com/hamzahanif2210/TAMBO-opt)
with local commits on top. This settles whether the degenerate showers described
in `README.md` come from that local divergence.

**They do not.** From identical inputs, the local generator and pristine upstream
produce byte-identical showers.

## Result

Pristine upstream cloned to `/n/home05/zdimitrov/tambo/TAMBO-opt-Hamza`
(`67bfdce restructure`, unchanged since the fork point).

| species | showers | elements compared | differing | blobs |
|---|---|---|---|---|
| electron | 512 | 10,485,760 | **0** | 0 |
| photon | 512 | 20,643,840 | **0** | 0 |
| muon | 384 | 48,168,960 | **0** | 0 |

`torch.equal` — not "close", exactly equal, in every column including the time
column that only the local version can produce at all.

So any degenerate shower one implementation emits, the other emits too, at the
same rate, by construction. **Our code is not the cause and cannot be the fix.**

## Reproduce

```bash
python tests/compare_upstream_generator.py mask-check     # CPU, seconds
sbatch -J upcmp_muon slurm/run_upstream_compare.sh muon 384 1
```

## What actually differs, and why none of it moves the numbers

Only `allshowers/generator.py` and `allshowers/transformer.py` differ on the
generation path. `preprocessing.py` (the `exp` inverse energy transform — the
prime suspect for the blobs), `flow_matching.py`, `ode_solvers.py` and
`util/allshowers_related/generate_showers.py` are **byte-identical**.

**`generator.py`** — adds `with_time`: detect `dim_inputs[0] == 4`, compose the
`samples_time_trafo`, sample 4 features, return 5 columns. Purely additive; the
non-time branch is byte-for-byte upstream. Plus `torch.no_grad()` in `generate()`,
which cannot change forward values.

All three checkpoints are `dim_inputs: [4, 6, 4]` time models, so **upstream
cannot run them at all** — it would sample 3 features into a 4-feature-trained
flow. Upstream's *own* `util/allshowers_related/generate_showers.py` (identical
to ours) already reads `generator.with_time` at line 231, an attribute upstream's
`generator.py` never sets. Upstream is internally inconsistent here; our change
is the repair, not a divergence.

**`transformer.py`** — two changes, both inert in production:

- A `pre_ln` flag defaulting to post-LN. But `stage_run_dir`
  (`scripts/00_generate_data_dual_species.py:167`) injects `pre_ln = True` into
  every staged conf, so we run pre-LN — upstream's hardcoded behaviour.
- `compute_mask` restricts the attention self-loop to padded queries. Verified
  inert by `mask-check`, which extracts both repos' real `mask_fn` closures and
  evaluates them densely:

```
  num_layer_cond    equal    differing elements   note
               4     True                     0   electron/muon
               8     True                     0   photon
              -1    False                   128   unused by these checkpoints
```

  Upstream's extra `| (q_idx == kv_idx)` is redundant for real queries — a point
  always satisfies `lower & upper & not_padding` against itself. The branch where
  they differ is not reachable for any of these checkpoints.

## Method

`tests/compare_upstream_generator.py`. Both repos ship a package named
`allshowers`, so they cannot share an interpreter — hence one process per
implementation, comparing saved tensors.

Controls, so that only the generator/transformer code varies:

- **`prep` runs once per species.** Primaries, PointCountFM and the anti-clip
  re-roll are computed once and saved, so both sides generate from byte-identical
  inputs and PointCountFM's RNG is not a confound.
- **Both sides use the same `forward`** (`time_forward`) and the same batching
  driver under `no_grad`, seeded per window from the window start index. The
  `no_grad` difference is therefore not a variable either.
- **Three implementations, not two.** `local` (native) vs `local-shim` (local
  `Generator`, ported forward) validates that the shim adds nothing beyond time
  support; `local-shim` vs `hamza` is then a clean read on the code difference.
  Both comparisons came back bit-identical for all three species.
- **`torch.set_float32_matmul_precision("high")`** is set, matching Step 0 — it
  changes float32 matmul numerics, so omitting it would have been a confound.
- The upstream run-dir gets a conf **without** `pre_ln`: upstream's `Transformer`
  has no such parameter and `Transformer(**params)` would raise `TypeError`. It is
  hardcoded pre-LN, which is what `pre_ln: true` asks the local one for.

Incidentally, bit-identity across separate processes shows the sampling pipeline
is deterministic process-to-process, which was not guaranteed — `flex_attention`
under `torch.compile` need not be.

## What this does NOT establish

**No blob occurred in 1,408 generated showers.** At the muon rate implied by
earlier scans (~0.2%) about one was expected in 384, so this is consistent, but
it means the run bounds the rate rather than measuring it. The heaviest tail seen
was muon row 290 at `median_e = 197.6` against a typical 3.7 — leaning the right
way, but six orders short of the 1e8 signature.

Bit-identity makes the *comparative* rate question moot: the two implementations
cannot differ in rate. The absolute rate is now a single-implementation question,
and the cheapest place to answer it is the existing 07 corpus already on disk
(572,164 muon rows) rather than fresh generation — a CPU scan on `-p test`, not
GPU time.

## Where to look next

`allshowers/preprocessing.py:79-85`, identical in both repos:

```python
self.log_base = math.log(base)
def forward(self, x):  return torch.log(x + self.alpha) / self.log_base
def inverse(self, x):  return torch.exp(self.log_base * x) - self.alpha
```

The energy inverse is an unguarded `exp` of a scaled latent (muon energy trafo is
`Log(alpha=0.0)` + `StandardScaler`). A latent landing a few sigma high scales
every point in the shower by one common factor — exactly the observed signature:
uniform inflation, normal point count, all values finite. The generator already
guards the *opposite* extreme; its own comment notes the transform "can emit
EXACTLY 0.0 for extreme negative latents (float32 underflow)". This is the
positive-latent counterpart, and nothing catches it.

That is upstream code, so it is a fix to propose upstream — or to guard on our
side at the corpus level, where the median-point-energy discriminator separates
the two populations by eight orders of magnitude.

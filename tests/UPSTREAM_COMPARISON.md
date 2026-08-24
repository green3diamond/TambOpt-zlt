# The blob bug is upstream's, not ours

The AllShowers generator lives in a sibling repo, `/n/home05/zdimitrov/tambo/
TAMBO-opt` — a fork of [hamzahanif2210/TAMBO-opt](https://github.com/hamzahanif2210/TAMBO-opt)
with local commits on top. This settles whether the degenerate showers in
`README.md` come from that divergence. **They do not:** from identical inputs the
local generator and pristine upstream produce byte-identical showers.

## Result

Pristine upstream at `/n/home05/zdimitrov/tambo/TAMBO-opt-Hamza` (`67bfdce
restructure`, unchanged since the fork point).

| species | showers | elements compared | differing |
|---|---|---|---|
| electron | 512 | 10,485,760 | **0** |
| photon | 512 | 20,643,840 | **0** |
| muon | 384 | 48,168,960 | **0** |

`torch.equal` — exactly equal, every column, including the time column only the
local version can produce. So either implementation emits the same degenerate
showers at the same rate, by construction: **our code is not the cause and cannot
be the fix.** (No blob occurred in these 1,408 showers, ~1 expected, so this bounds
rather than measures the rate — see `BLOB_GUARD_FINDINGS.md`.)

```bash
python tests/compare_upstream_generator.py mask-check     # CPU, seconds
sbatch -J upcmp_muon slurm/run_upstream_compare.sh muon 384 1
```

## What differs, and why none of it moves the numbers

Only `generator.py` and `transformer.py` differ on the generation path.
`preprocessing.py` (the prime suspect), `flow_matching.py`, `ode_solvers.py` and
`generate_showers.py` are byte-identical.

**`generator.py`** adds `with_time` (detect `dim_inputs[0] == 4`, sample 4 features,
return 5 columns) — purely additive, the non-time branch is byte-for-byte upstream —
plus a `torch.no_grad()` that cannot change forward values. All three checkpoints are
`dim_inputs: [4, 6, 4]` time models, so **upstream cannot run them at all**; its own
`generate_showers.py` (identical to ours) already reads `generator.with_time` at line
231, an attribute upstream never sets. Our change is the repair, not a divergence.

**`transformer.py`**, both changes inert in production:

- A `pre_ln` flag defaulting to post-LN — but `stage_run_dir` injects `pre_ln = True`
  into every staged conf, so we run pre-LN, upstream's hardcoded behaviour.
- `compute_mask` restricts the attention self-loop to padded queries. Upstream's extra
  `| (q_idx == kv_idx)` is redundant for real queries (a point always satisfies
  `lower & upper & not_padding` against itself). `mask-check` evaluates both repos'
  real `mask_fn` closures densely: 0 differing elements at `num_layer_cond` 4
  (electron/muon) and 8 (photon), differing only at `-1`, which these checkpoints
  never use.

## Method

`tests/compare_upstream_generator.py`. Both repos ship a package named `allshowers`,
so they cannot share an interpreter — one process each, comparing saved tensors.
Controls: `prep` runs once per species (primaries, PointCountFM and the anti-clip
re-roll saved, so PCFM's RNG is not a confound); both sides share the same `forward`
and batching driver under `no_grad`, seeded per window;
`set_float32_matmul_precision("high")` matches Step 0, which changes float32 matmul
numerics; the upstream run-dir gets a conf without `pre_ln`, whose `Transformer` has
no such parameter. A third implementation (`local-shim`) isolates the shim from the
code difference — both comparisons bit-identical. Incidentally this shows the
sampling pipeline is deterministic process-to-process, which `flex_attention` under
`torch.compile` did not guarantee.

## Where to look next

`allshowers/preprocessing.py:79-85`, identical in both repos:

```python
def inverse(self, x):  return torch.exp(self.log_base * x) - self.alpha
```

An unguarded `exp` of a scaled latent (muon energy trafo is `Log(alpha=0.0)` +
`StandardScaler`). A latent a few sigma high scales every point by one common factor
— exactly the signature: uniform inflation, normal point count, all finite. The
generator already guards the *opposite* extreme; its own comment notes the transform
"can emit EXACTLY 0.0 for extreme negative latents (float32 underflow)". This is the
positive-latent counterpart, and nothing catches it.

Upstream code, so it is a fix to propose upstream — or to guard at the corpus level,
which is what `BLOB_MEDIAN_E` does.

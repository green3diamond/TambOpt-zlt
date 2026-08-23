"""Generate a paired multi-species shower corpus from per-species AllShowers checkpoints.

One physical shower is split by secondary species into the components listed in
`constants.SPECIES_NAMES`; each has its own model, and a complete event is their
sum. Corpus layout is species-major: component s occupies rows
[s*N, (s+1)*N), so rows i, N+i, 2N+i … are one event's components and share a
primary (energy, direction, EM/hadronic label).

The corpus `pdg` field stores the EM/hadronic class (0 or 1) — the generator's
conditioning input, NOT the species. Species is written to a separate sidecar
`<corpus>_species.pt` (id = index into SPECIES_NAMES) for Step-1/2 routing.

Key design decisions:
- Per-species models have different point caps (electron 4096, photon 8064,
  muon 25088); every block is zero-padded to the largest so the file is uniform.
- Generation streams in chunks; the file is preallocated once (`create_empty_file`)
  and each chunk appended at its row offset (`save_batch`) — peak RAM is one chunk.
- `Generator`/`generate` are called directly (not the `GenerateShowers` wrapper)
  because the wrapper hardcodes max_points, staging, and full-corpus RAM return.
- Each per-species staged run-dir has `pre_ln: true` injected into conf.yaml —
  the May checkpoints use pre-LN transformer blocks; loading them into a post-LN
  model silently generates blobs (shared state_dict keys, no error).
- PointCountFM runs on CPU (TorchScript device constants baked at trace time).
- Anti-clip re-roll: if PCFM predicts > cap points, re-roll (up to MAX_PCFM_RETRIES)
  before the expensive GPU generate step to reduce blob artefacts from truncation.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch
import torch._utils  # noqa: F401 — torch 2.x lazy submodule needed by torch.save on Py3.13

torch.set_float32_matmul_precision("high")

import showerdata
import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import (
    LOG_E_MIN, LOG_E_MAX,
    ZENITH_MIN, ZENITH_MAX, AZIMUTH_MIN, AZIMUTH_MAX,
    SHOWER_CACHE, RUN_LOCATION, NUM_SHOWERS, BATCH_SIZE,
    USE_TAU_PRIMARIES, TAU_WHOLESKY_PATH, DUAL_SHOWER_CACHE_PATH,
    HOLDOUT_FRAC, HOLDOUT_SEED, HELDOUT_SHOWER_CACHE_PATH,
    SPECIES_NAMES,
)
from modules.showers import load_tau_primaries

# Low-level generator pieces (importing modules.generate_showers injects TAMBO-opt path).
from modules.showers.generate import GenerateShowers  # noqa: F401 — triggers TAMBO-opt sys.path injection in modules.showers.generate
from allshowers.generate_showers import (
    sample_primary_particles, run_point_count_fm,
)
from allshowers.generator import Generator, generate

# ── Config ───────────────────────────────────────────────────────────────────
BEST = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/checkpoints"

# Per-species model paths + point-cloud caps, keyed by species name. The species
# id is the BLOCK INDEX — defined by constants.SPECIES_NAMES, which this dict is
# checked against below — and is written to the Step-0 species sidecar. It is not
# in the corpus `pdg` field, which carries the EM/hadronic class fed to the
# generator.
SPECIES = {
    "electron": dict(
        allshower_run=os.path.join(BEST, "20260519_185649_Electron-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_040716_Electron-PointCountFM", "compiled.pt"),
        max_points=4096,
    ),
    "muon": dict(
        allshower_run=os.path.join(BEST, "20260520_160031_Muons-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_043912_Muon-PointCountFM", "compiled.pt"),
        max_points=25088,
    ),
    "photon": dict(
        allshower_run=os.path.join(BEST, "20260724_074020_Photon-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260727_041023_Photon-PointCountFM", "compiled.pt"),
        max_points=8064,
    )
}

# SPECIES_NAMES (modules/constants.py) is the id order the corpus sidecar and
# every downstream router use; this dict adds the model paths for each. Keeping
# them as two objects that must agree is the point of the check -- the paths are
# a Step-0 concern, the ORDER is a corpus-format concern.
if tuple(SPECIES) != tuple(SPECIES_NAMES):
    raise SystemExit(
        f"SPECIES {tuple(SPECIES)} does not match constants.SPECIES_NAMES "
        f"{tuple(SPECIES_NAMES)}.\nThe tuple's order IS the species id written to "
        f"the corpus sidecar, so the two must agree exactly (same names, same order).")

NUM_TIMESTEPS = 16
SOLVER        = "midpoint"
CHUNK_SIZE    = 2000               # showers per streamed write-batch (bounds peak RAM)
STAGE_ROOT    = os.path.join(RUN_LOCATION, "allshowers_staged")
DEVICE        = torch.device("cuda")

# Anti-clip resampling (mainly muons). PointCountFM predicts a per-shower total
# point count; if it exceeds the species cap (`max_points`, 25088 for muons) the
# generator TRUNCATES the tail — losing points turns a rod into a diffuse blob.
# Because the clip is decided from num_points BEFORE the expensive AllShowers
# generate, we cheaply re-roll the (stochastic) PointCountFM for over-cap showers.
# Each re-roll replaces the previous draw, so a shower that comes back under the
# threshold keeps that draw and stops; one that stays over cap for the whole
# budget simply keeps its LAST re-roll and is truncated as before. MAX_CLIP_FRAC
# = tolerable fraction of predicted points lost to the cap (0.0 = re-roll on any
# clipping). Set MAX_PCFM_RETRIES = 0 to disable.
MAX_CLIP_FRAC    = 0.10
MAX_PCFM_RETRIES = 10

# Degenerate-shower ("blob") re-generation. A rare draw comes back with EVERY
# point's energy inflated by one common factor: the inverse energy transform is
# an unguarded exp of a latent (allshowers/preprocessing.py Log.inverse), so a
# latent landing high rescales the whole cloud. Such a shower is FINITE and its
# point count is normal, so neither the underflow guard in _gen_chunk nor the
# anti-clip re-roll above can see it — and left in, one shower totalling 2.06e14
# reaches Step 2 as a log1p target of ~33 where normal is single digits.
#
# The discriminator is deposited energy / primary energy. There is NO gap
# between the good and bad populations: measured over all 1,144,328 rows of the
# 07 corpus the muon tail is continuously filled from ~10 up to 1e27. So each
# cut is placed at its own species' KNEE, where the quantile curve stops being
# smooth and starts escalating:
#   muon      p99=6.0   p99.5=10.1  p99.7=41.9  p99.8=112  p99.9=665
#   electron  p99.9=14.6  p99.95=19.2  p99.97=24.2  p99.99=58
# The species need different numbers because their bulks differ (median ratio
# 0.21 for muons, 3.15 for electrons). At these cuts 0.39% of muon and 0.022%
# of electron showers are re-generated. Because there is no gap the cut does
# clip some legitimate tail, but it is insensitive: moving the muon cut from 20
# to 100 changes the re-generated fraction by only 0.18% of rows.
#
# photon was NOT in the corpus these were measured on — its value is inherited
# from the electron knee and should be re-measured with tests/scan_corpus_blobs.py.
#
# MAX_BLOB_RETRIES = 3 because failures concentrate at high primary energy (the
# muon top E_prim decile fails at 3.1%, the bottom at 0.03%), so the per-draw
# probability for the showers that actually fail is ~3%, not the 0.39% average;
# 3 retries leaves ~1e-6 residual there. Set to 0 to disable.
MAX_TOTAL_OVER_PRIM = {"electron": 30.0, "muon": 20.0, "photon": 30.0}
MAX_BLOB_RETRIES    = 3

# State-dict wrapper keys to probe when checkpoints/best_epoch_*.pt is not already
# a flat tensor dict (the Generator wants the raw flow state_dict).
_WRAP_KEYS = ("model", "model_state_dict", "state_dict", "ema", "ema_model",
              "flow", "net", "network", "weights")


def _extract_state_dict(obj):
    """Return a flat {name: tensor} state_dict from a loaded checkpoint, whether
    it is already raw or wrapped under a common key."""
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj                                   # already a raw state_dict
    if isinstance(obj, dict):
        for k in _WRAP_KEYS:
            v = obj.get(k)
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                print(f"  [stage] extracted state_dict from wrapper key '{k}'")
                return v
        # Last resort: first dict-of-tensors value at the top level.
        for k, v in obj.items():
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                print(f"  [stage] extracted state_dict from key '{k}'")
                return v
        raise RuntimeError(
            f"could not find a state_dict inside checkpoint; top-level keys = {list(obj.keys())}")
    raise RuntimeError(f"unexpected checkpoint object type: {type(obj)}")


def stage_run_dir(name, cfg):
    """Build a Generator-loadable run-dir in fast local storage. Idempotent.

    Copies conf.yaml + preprocessing/trafos.pt, extracts the flow state_dict from
    checkpoints/best_epoch_*.pt → weights/best.pt, and copies the PointCountFM
    compiled.pt. Returns (staged_run_dir, staged_pcfm_path)."""
    src = cfg["allshower_run"]
    dst = os.path.join(STAGE_ROOT, name)
    weights_pt = os.path.join(dst, "weights", "best.pt")
    pcfm_dst   = os.path.join(dst, "pcfm_compiled.pt")

    os.makedirs(os.path.join(dst, "weights"), exist_ok=True)
    os.makedirs(os.path.join(dst, "preprocessing"), exist_ok=True)

    # conf.yaml is ALWAYS rewritten (even when already staged) so older staged
    # dirs pick up the injection: these May checkpoints were trained with
    # pre-LN transformer blocks (verified 2026-06-10 — post-LN loads silently
    # but generates blobs), so the staged conf requests pre_ln explicitly.
    with open(os.path.join(src, "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    conf["model"]["pre_ln"] = True
    with open(os.path.join(dst, "conf.yaml"), "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)

    if os.path.exists(weights_pt) and os.path.exists(pcfm_dst) \
            and os.path.exists(os.path.join(dst, "preprocessing", "trafos.pt")):
        print(f"[stage] {name}: already staged at {dst} (conf.yaml re-patched)")
        return dst, pcfm_dst

    print(f"[stage] {name}: staging {src} -> {dst}")
    shutil.copy2(os.path.join(src, "preprocessing", "trafos.pt"),
                 os.path.join(dst, "preprocessing", "trafos.pt"))

    ckpts = sorted(glob.glob(os.path.join(src, "checkpoints", "best_epoch_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints/best_epoch_*.pt in {src}")
    print(f"[stage] {name}: loading {os.path.basename(ckpts[-1])}")
    raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    sd = _extract_state_dict(raw)
    torch.save(sd, weights_pt)
    print(f"[stage] {name}: wrote weights/best.pt ({len(sd)} tensors)")

    shutil.copy2(cfg["pcfm_compiled"], pcfm_dst)
    print(f"[stage] {name}: copied PointCountFM compiled.pt")
    return dst, pcfm_dst


def _pad_points(samples, target_P):
    """Zero-pad a (N, P, 5) tensor up to target_P points (padding rows = 0, which
    the kernel ignores via its energy mask)."""
    N, P, C = samples.shape
    if P == target_P:
        return samples
    out = torch.zeros((N, target_P, C), dtype=samples.dtype)
    out[:, :P, :] = samples
    return out


def resample_overclip(pcfm, energies, directions, labels, num_points, cap,
                      max_clip_frac=MAX_CLIP_FRAC, max_retries=MAX_PCFM_RETRIES):
    """Re-roll PointCountFM for showers whose predicted total point count would be
    clipped by more than `max_clip_frac` at `cap` — truncation turns a rod into a
    diffuse blob. PointCountFM samples noise per call, so re-running it for the
    over-cap subset yields fresh counts; each re-roll REPLACES the previous draw
    (several retries → keep the last). Only the failed subset is re-rolled, and
    only the cheap CPU stage is touched (the GPU generate runs once afterward).
    Mutates and returns `num_points`. No-op when max_retries <= 0.

    Shared by the generation pipeline (`_gen_chunk`) and the angle-grid plots so
    both apply the identical anti-clip policy."""
    if max_retries <= 0:
        return num_points

    cap = int(cap)
    n = int(num_points.shape[0])

    def _clip_frac(npts):
        totals = npts.sum(1).to(torch.float32)             # (m,) predicted total
        return (totals - cap).clamp(min=0.0) / totals.clamp(min=1.0)

    clip_frac = _clip_frac(num_points)
    for attempt in range(1, max_retries + 1):
        bad = clip_frac > max_clip_frac
        nbad = int(bad.sum())
        if nbad == 0:
            break
        idx = torch.nonzero(bad, as_tuple=False).flatten()
        print(f"  [anti-clip {attempt}/{max_retries}] re-rolling PointCountFM "
              f"for {nbad}/{n} shower(s) clipping >{max_clip_frac:.0%} (cap {cap})")
        new_np = run_point_count_fm(
            model_path=pcfm, energies=energies[idx], directions=directions[idx],
            labels=labels[idx],
        )
        num_points[idx] = new_np                           # keep the latest draw
        clip_frac = _clip_frac(num_points)

    # Report only showers still ABOVE the re-roll threshold after the budget
    # (these were actually re-rolled and kept their last draw). Showers over the
    # cap but within max_clip_frac were intentionally never re-rolled — that
    # truncation is tolerated, so don't flag it as a retry failure.
    still = int((clip_frac > max_clip_frac).sum())
    if still:
        print(f"  [anti-clip] {still}/{n} shower(s) still clip >{max_clip_frac:.0%} "
              f"(cap {cap}) after {max_retries} retries — truncated (kept last draw)")
    return num_points


def _generate_batched(gen, energies, num_points, directions, labels, sp_batch):
    """AllShowers generate with OOM back-off, returning (samples, sp_batch).

    The reduced batch is handed back so a caller that generates more than once
    (the blob re-roll below) keeps the size this GPU was already shown to fit
    instead of rediscovering it from scratch."""
    while True:
        try:
            samples = generate(
                generator=gen, energies=energies, num_points=num_points,
                angles=directions, batch_size=sp_batch, device=str(DEVICE),
                labels=labels,
            ).float().cpu()                                # (n, sp_max_points, 5)
            return samples, sp_batch
        except torch.OutOfMemoryError:
            if sp_batch == 1:
                raise
            torch.cuda.empty_cache()
            sp_batch = max(1, sp_batch // 2)
            print(f"  [oom] retrying generate at batch {sp_batch}", flush=True)


def _deposited_over_primary(samples, energies):
    """Deposited energy / primary energy, one value per shower.

    Padding rows carry energy 0 and contribute nothing. Non-finite points
    propagate through the sum into the ratio, so the single `isfinite` test in
    the caller covers NaN/inf as well as the inflated-but-finite blobs (the 07
    corpus holds one inf muon row alongside its 487 finite ones)."""
    tot = samples[:, :, 3].clamp(min=0).sum(dim=1).to(torch.float64)
    return tot / energies.reshape(-1).to(torch.float64).clamp(min=1e-30)


def regenerate_degenerate(gen, name, samples, energies, num_points, directions,
                          labels, sp_batch, max_ratio, max_retries=MAX_BLOB_RETRIES):
    """Re-generate showers whose deposited/primary energy ratio is degenerate.

    AllShowers draws fresh noise per call and Step 0 never seeds it, so simply
    re-running the failed subset yields a different shower from the same
    conditioning. Each pass REPLACES the previous draw for those rows only; a
    shower that comes back clean stops being re-rolled, and one still bad after
    the budget keeps its last draw and is reported. Mutates `samples` in place
    and returns (samples, sp_batch). No-op when max_retries <= 0.

    Re-rolling rather than dropping is required, not stylistic: corpus rows are
    paired events with row-indexed species/position sidecars, so removing a row
    would desynchronise the pairing and both sidecars.

    See MAX_TOTAL_OVER_PRIM for where `max_ratio` comes from."""
    if max_retries <= 0 or not max_ratio:
        return samples, sp_batch

    n = int(samples.shape[0])
    ratio = _deposited_over_primary(samples, energies)
    for attempt in range(1, max_retries + 1):
        bad = ~torch.isfinite(ratio) | (ratio > max_ratio)
        nbad = int(bad.sum())
        if nbad == 0:
            break
        idx = torch.nonzero(bad, as_tuple=False).flatten()
        worst = ratio[bad].max().item()
        print(f"  [blob {attempt}/{max_retries}] re-generating {nbad}/{n} {name} "
              f"shower(s) with deposited/primary > {max_ratio:g} "
              f"(worst {worst:.4g})", flush=True)
        new, sp_batch = _generate_batched(
            gen, energies[idx], num_points[idx], directions[idx],
            None if labels is None else labels[idx], sp_batch)
        # gen.max_points is fixed per species, so this should always hold; check
        # rather than let a silent broadcast corrupt the block.
        if new.shape[1:] != samples.shape[1:]:
            raise RuntimeError(
                f"re-generated {name} showers have shape {tuple(new.shape)}, "
                f"incompatible with the chunk {tuple(samples.shape)}")
        samples[idx] = new                                 # keep the latest draw
        ratio = _deposited_over_primary(samples, energies)

    still = int((~torch.isfinite(ratio) | (ratio > max_ratio)).sum())
    if still:
        print(f"  [blob] {still}/{n} {name} shower(s) still degenerate after "
              f"{max_retries} retries — kept last draw", flush=True)
    return samples, sp_batch


def _gen_chunk(gen, pcfm, cfg, species_name, energies, directions, labels,
               shower_ids, target_P,
               max_clip_frac=MAX_CLIP_FRAC, max_retries=MAX_PCFM_RETRIES,
               max_blob_retries=MAX_BLOB_RETRIES, batch=BATCH_SIZE):
    """Generate one chunk of showers for a species from PRE-SAMPLED primaries
    → a showerdata.Showers, padded to target_P. Bounded memory (only this
    chunk is held). Primaries (energies, directions, labels, shower_ids) come in
    as slices of the corpus-wide arrays so both species blocks share them (paired
    events).

    `species_name` is the SPECIES key (electron/muon/photon) — it selects the
    per-species degenerate-shower threshold, so it must be the dict key rather
    than a display name.

    `labels` is the per-event EM/hadronic primary class (0/1) — the generator's
    conditioning input (both per-species models were trained on both classes)
    and the value stored as the corpus `pdg` field. `shower_ids` is the
    paired-event id, so every component row of one event shares it."""
    labels = labels.to(torch.int64)

    # Stage 1 — PointCountFM on CPU (TorchScript device-baked → CUDA mismatches).
    num_points = run_point_count_fm(
        model_path=pcfm, energies=energies, directions=directions, labels=labels,
    )

    # Anti-clip re-roll for over-cap showers (mainly muons): re-roll only the
    # failed subset on the cheap CPU stage so the single GPU generate below sees
    # counts that mostly fit the cap (truncation → blob). See resample_overclip.
    num_points = resample_overclip(
        pcfm, energies, directions, labels, num_points,
        cap=int(cfg["max_points"]), max_clip_frac=max_clip_frac, max_retries=max_retries,
    )

    # Stage 2 — AllShowers on GPU (max_points already set on gen).
    # Batch scaled by the species point cap: flex_attention's block mask is
    # O(batch x cap^2), and the caps differ 6x (electron 4096, photon 8064, muon
    # 25088). One flat batch either OOMs on muon or wastes the GPU on electron --
    # at BATCH_SIZE=60 the muon mask alone asked for 20.35 GiB on a 19.6 GB card.
    # `batch` is therefore the budget AT the electron cap. gpu_test also hands out
    # mixed cards, so halve and retry rather than assume which one we got.
    sp_batch = max(1, int(batch * 4096 / int(cfg["max_points"])))
    samples, sp_batch = _generate_batched(
        gen, energies, num_points, directions, labels, sp_batch)

    # Degenerate-shower re-roll: rare draws come back with every point's energy
    # inflated by a common factor. Finite and normally-sized, so this is the
    # only guard that sees them. Before _pad_points, so the ratio is computed on
    # the species-cap tensor the generator actually returned.
    samples, sp_batch = regenerate_degenerate(
        gen, species_name, samples, energies, num_points, directions, labels,
        sp_batch, max_ratio=MAX_TOTAL_OVER_PRIM.get(species_name),
        max_retries=max_blob_retries,
    )

    samples = _pad_points(samples, target_P)

    # Underflow guard: the inverse energy trafo (exp of a latent) can emit
    # EXACTLY 0.0 for extreme negative latents (float32 underflow, ~1-in-1e8
    # per point — guaranteed at production corpus sizes). showerdata requires
    # real points contiguous at the front: its ragged save slices
    # [:num_points] (an interior zero silently drops the last real point) and
    # a zero at slot 0 raises "Padding should be in the end". Stable-partition
    # each shower: e>0 rows first in original order, zero rows (including the
    # padding) at the end.
    key = (samples[:, :, 3] <= 0).to(torch.int8)           # 0 = real, 1 = zero/pad
    order = torch.argsort(key, dim=1, stable=True)         # (n, target_P)
    samples = torch.gather(
        samples, 1, order.unsqueeze(-1).expand(-1, -1, samples.shape[2]))
    
    # Saved pdg = the EM/hadronic primary class (the generator's conditioning
    # label). The e/µ species is recorded in the Step-0 species sidecar, not here.
    pdg = labels

    return showerdata.Showers(
        points=samples.numpy(), energies=energies.numpy(),
        pdg=pdg.numpy(), directions=directions.numpy(),
        shower_ids=shower_ids.numpy(),
    )


def _load_progress(progress_path):
    """{species_name: rows_done, ...}, or {} if no checkpoint yet."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {}


def _save_progress(progress_path, state):
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, progress_path)


def _generate_corpus(tag, out_path, energies_all, directions_all, labels_all,
                     event_ids_all, positions_all, target_P, chunk_size,
                     batch=BATCH_SIZE):
    """Generate one full paired electron+muon corpus at `out_path`, resuming
    automatically from `<out_path>.progress.json` if a prior attempt was
    preempted mid-run (gpu_requeue). `tag` is a log label ("train"/"holdout").

    Progress is tracked per-species (rows written so far); a chunk is only
    counted done AFTER `showerdata.save_batch` returns, so a kill mid-chunk
    just re-does that one chunk on restart — no partial rows on disk get
    silently treated as complete."""
    n_pairs   = int(energies_all.shape[0])
    n_species = len(SPECIES_NAMES)
    total     = n_species * n_pairs
    progress_path = out_path + ".progress.json"

    if os.path.exists(out_path):
        state = _load_progress(progress_path)
        print(f"[{tag}] resuming into existing {out_path}  progress={state}")
    else:
        state = {}
        showerdata.create_empty_file(out_path, shape=(total, target_P, 5), overwrite=True)
        print(f"[{tag}] preallocated {out_path} ({total}x{target_P}x5, "
              f"≈{total * target_P * 5 * 4 / 1e9:.1f} GB)")

    # Species + real-position sidecars: cheap, derived purely from n_pairs/
    # positions_all, so always safe to (re)write regardless of resume state.
    # Species-major, matching the block layout below: id s fills rows
    # [s*n_pairs, (s+1)*n_pairs).
    species_ids = torch.repeat_interleave(
        torch.arange(n_species, dtype=torch.int64), n_pairs)
    species_path = os.path.splitext(out_path)[0] + "_species.pt"
    torch.save(species_ids, species_path)
    if positions_all is not None:
        # Every component of an event decays at the same vertex, so the block is
        # repeated once per species rather than stored per row.
        positions_dual = torch.cat([positions_all] * n_species, dim=0)
        positions_path = os.path.splitext(out_path)[0] + "_positions.pt"
        torch.save(positions_dual, positions_path)
        print(f"[{tag}/positions] wrote sidecar {positions_path} {tuple(positions_dual.shape)}")
    print(f"[{tag}/species] wrote sidecar {species_path} "
          f"({n_pairs} rows x {' + '.join(SPECIES_NAMES)})")

    t0 = time.time()
    for i, (name, cfg) in enumerate(SPECIES.items()):
        block_start = i * n_pairs
        done = int(state.get(name, 0))
        if done >= n_pairs:
            print(f"[{tag}/{name}] block already complete ({done}/{n_pairs}) — skipping")
            continue
        print("=" * 72)
        print(f"[{tag}/{name}] {n_pairs - done} of {n_pairs} showers  "
              f"(max_points={cfg['max_points']}"
              f"{f', resuming at row {done}' if done else ''})")
        print("=" * 72)
        staged_dir, pcfm = stage_run_dir(name, cfg)
        gen = Generator(run_dir=staged_dir, num_timesteps=NUM_TIMESTEPS,
                        compile=True, solver=SOLVER)
        gen.max_points = int(cfg["max_points"])

        while done < n_pairs:
            c = min(chunk_size, n_pairs - done)
            sh = _gen_chunk(
                gen, pcfm, cfg, name,
                energies_all[done:done + c], directions_all[done:done + c],
                labels_all[done:done + c], event_ids_all[done:done + c], target_P,
                batch=batch,
            )
            showerdata.save_batch(sh, out_path, start=block_start + done)
            done += c
            state[name] = done
            _save_progress(progress_path, state)
            del sh
            torch.cuda.empty_cache()
            print(f"[{tag}/{name}] wrote {done}/{n_pairs}  "
                  f"(file offset {block_start + done}/{total})  {time.time()-t0:.0f}s")
        del gen
        torch.cuda.empty_cache()

    blocks = ", ".join(f"{nm} {i*n_pairs}..{(i+1)*n_pairs-1}"
                       for i, nm in enumerate(SPECIES_NAMES))
    print(f"[{tag}/done] {total} rows = {n_pairs} paired events ({blocks}) "
          f"in {time.time()-t0:.0f}s -> {out_path}")
    if os.path.exists(progress_path):
        os.remove(progress_path)


def main():
    ap = argparse.ArgumentParser()
    # Streamed in chunks → peak RAM is one chunk, not the whole corpus, so the
    # pair count can scale freely (disk is the only limit). Muons are capped at
    # 25088 points; the file is preallocated at that P and electrons are padded up.
    ap.add_argument("--n-pairs", type=int, default=0,
                    help="number of paired events BEFORE the holdout split (see "
                         "HOLDOUT_FRAC); 0 = all available (all in-band tau "
                         "events with USE_TAU_PRIMARIES, else NUM_SHOWERS "
                         "synthetic).")
    ap.add_argument("--seed", type=int, default=0,
                    help="primary-sampling seed (deterministic corpus)")
    ap.add_argument("--chunk", type=int, default=CHUNK_SIZE,
                    help="showers per streamed write-batch (bounds peak RAM)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="GPU generation batch AT the electron cap (4096); scaled "
                         "down per species by its own max_points, and halved on "
                         "OOM. Lower it on a small card -- the default needed "
                         "20.35 GiB for the muon block on a 19.6 GB GPU")
    ap.add_argument("--out", type=str, default=None,
                    help="MAIN (train-pool) output .pt path (default: "
                         "DUAL_SHOWER_CACHE_PATH from constants). The holdout "
                         "corpus always goes to HELDOUT_SHOWER_CACHE_PATH — it "
                         "isn't overridable, so eval_true_utility.py can always "
                         "find it.")
    args = ap.parse_args()

    os.makedirs(SHOWER_CACHE, exist_ok=True)
    os.makedirs(STAGE_ROOT, exist_ok=True)

    target_P = max(cfg["max_points"] for cfg in SPECIES.values())

    # Sample the primaries ONCE — both species blocks reuse them, so row i and
    # row n_pairs+i are the two components of one physical event.
    if USE_TAU_PRIMARIES:
        # Real tau primaries (energy + direction + physical ENU decay position),
        # filtered to the generator's trained energy band. n_pairs follows the
        # number of in-band events (capped by --n-pairs if given).
        prim = load_tau_primaries(
            TAU_WHOLESKY_PATH, e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX,
            n=int(args.n_pairs), seed=args.seed,
        )
        positions_all = prim["positions"]                 # (M, 3) ENU (East, North, Up)
    else:
        n_req = int(args.n_pairs) or NUM_SHOWERS
        prim = sample_primary_particles(
            e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX,
            zenith_min=ZENITH_MIN, zenith_max=ZENITH_MAX,
            azimuth_min=AZIMUTH_MIN, azimuth_max=AZIMUTH_MAX,
            n=n_req, seed=args.seed,
        )
        positions_all = None                              # synthetic: no real position (Step 1 re-centers)

    energies_all, directions_all = prim["energies"], prim["directions"]
    # Per-event EM/hadronic primary class (0/1); both species blocks reuse it so paired rows share the class.
    labels_all = prim["labels"]
    n_total = int(energies_all.shape[0])
    event_ids_all = torch.arange(n_total, dtype=torch.int64)   # global event id (informational)

    # ── Holdout split — PHYSICAL EVENT level, before any generation ──────────
    # Deterministic seeded permutation of the n_total events (independent of
    # --seed, which only drives the EM/hadronic label draw): HOLDOUT_FRAC go to
    # a separate corpus that Steps 1-4 never read, reserved for
    # eval_true_utility.py. Sorting each half keeps rows in file order (cheap,
    # not required for correctness, just tidier logs/debugging).
    g = torch.Generator().manual_seed(HOLDOUT_SEED)
    perm = torch.randperm(n_total, generator=g)
    n_holdout = max(1, int(round(HOLDOUT_FRAC * n_total)))
    holdout_idx = perm[:n_holdout].sort().values
    train_idx   = perm[n_holdout:].sort().values
    n_train     = n_total - n_holdout

    out_path     = args.out or DUAL_SHOWER_CACHE_PATH
    heldout_path = HELDOUT_SHOWER_CACHE_PATH

    print("=" * 72)
    print("v6/00_generate_data_dual_species.py — paired electron + muon corpus (streamed)")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"primaries   : {f'REAL tau ({os.path.basename(TAU_WHOLESKY_PATH)}, ENU position + direction)' if USE_TAU_PRIMARIES else 'synthetic sample_primary_particles'}")
    if USE_TAU_PRIMARIES:
        print(f"            : {n_total} in-band taus in [1e{LOG_E_MIN:g}, 1e{LOG_E_MAX:g}] GeV "
              f"(East {positions_all[:,0].min():.0f}..{positions_all[:,0].max():.0f}, "
              f"North {positions_all[:,1].min():.0f}..{positions_all[:,1].max():.0f}, "
              f"Up {positions_all[:,2].min():.0f}..{positions_all[:,2].max():.0f} m)")
    print(f"total events: {n_total}  (seed={args.seed})")
    print(f"holdout     : {n_holdout} events ({100*HOLDOUT_FRAC:.1f}%, seed={HOLDOUT_SEED}) "
          f"-> {heldout_path}  [eval_true_utility.py ONLY]")
    print(f"train pool  : {n_train} events -> {out_path}  [Steps 1-4]")
    for name in SPECIES:
        print(f"{name:12s} : max_points={SPECIES[name]['max_points']}")
    print(f"chunk       : {args.chunk}  -> peak RAM ≈ "
          f"{args.chunk * target_P * 5 * 4 / 1e9:.2f} GB/chunk")

    def _slice(idx):
        pos = positions_all[idx] if positions_all is not None else None
        return energies_all[idx], directions_all[idx], labels_all[idx], event_ids_all[idx], pos

    e_tr, d_tr, l_tr, id_tr, pos_tr = _slice(train_idx)
    e_ho, d_ho, l_ho, id_ho, pos_ho = _slice(holdout_idx)

    _generate_corpus("train",   out_path,     e_tr, d_tr, l_tr, id_tr, pos_tr, target_P,
                     args.chunk, batch=args.batch)
    _generate_corpus("holdout", heldout_path, e_ho, d_ho, l_ho, id_ho, pos_ho, target_P,
                     args.chunk, batch=args.batch)


if __name__ == "__main__":
    main()

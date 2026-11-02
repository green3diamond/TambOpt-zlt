"""Step-1 dataset builder: layouts, ring-kernel labels in the shower frame, training tensors.

Detectors are placed by horizontal map coordinates (East, North) — the ENU
convention of the h5 files — with `surface` giving Up = g(North, East), and read
in each shower's own frame (`detectors_in_frames`). `encode_primary` /
`compute_normalization` are re-exported from `surrogates.fnn` for convenience.
"""

import os
import time
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from ..showers import GetCounts_planeaware, adaptive_bandwidths

from ..constants import (BLOB_MEDIAN_E, N_DETECTORS,
                         PRIMARY_DIM, SPECIES_NAMES, SIGMA_SPATIAL,
                         KERNEL_INTERP_K, C8_PLANE_DX)
from ..layouts.strategies import (_STRATEGIES, _STRATEGY_FNS)
from ..surrogates import encode_primary, compute_normalization  # noqa: F401  (re-export)
from ..surrogates.fnn import _load_species_sidecar


def _batch_layout_rng(seed: int, s_idx: int, b_idx: int):
    """Generator for one (strategy, batch) detector-layout draw.

    Every species row of one event must land under the same layout, but the
    builder streams one species block after another through a single loop body.
    A shared running generator would have advanced by all the preceding blocks
    before a later species starts, so the rows of one event would draw different
    layouts. Keying the draw on (seed, strategy, batch index within the species
    block) makes it a pure function of "where in the species block are we",
    which is identical for every pass.
    """
    return np.random.default_rng((int(seed), int(s_idx), int(b_idx)))


def _positions_sidecar_path(shower_cache_path: str) -> str:
    """Path of the Step-0 ENU decay-position sidecar paired with a dual corpus
    .pt (`…_dual.pt` -> `…_dual_positions.pt`). Written by
    00_generate_data_dual_species.py for real-primary (tau) runs; row-aligned
    with the corpus, columns (East, North, Up) in metres."""
    base, ext = os.path.splitext(shower_cache_path)
    return base + "_positions" + ext


def _load_positions_sidecar(shower_cache_path: str, keep_idx):
    """Load the Step-0 ENU decay-position sidecar, indexed by `keep_idx`.

    No synthetic fallback: a missing sidecar is an error, not a silent change of
    geometry."""
    path = _positions_sidecar_path(shower_cache_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"decay-position sidecar not found: {path}\n"
            "Placement requires the real ENU decay vertices from tau_wholesky.jl. "
            "Re-run 00_generate_data_dual_species.py to write the sidecar.")
    return torch.load(path)[torch.as_tensor(keep_idx)].float()          # (N, 3) East,North,Up


# ── Shower-frame placement (run 18 on) ───────────────────────────────────────

def shower_frames(positions: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """(M, 12) per-shower frames [p, d, e1, e2]: decay vertex p, unit axis d,
    e1 = (-d_N, d_E, 0)/|.| horizontal and perpendicular to the heading (left of it),
    e2 = d x e1 perpendicular to the axis (straight up for a horizontal shower).

    e1 / e2 are C8's own ObservationPlane axes up to sign (e1 = -y_hat, e2 = -x_hat
    of c8_air_shower.cpp). e1 is undefined for a vertical shower; the run's zenith
    cut (60-100 deg) keeps |d_horizontal| >= sin(60 deg)."""
    p = positions.double()
    d = dirs.double()
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-12)
    h = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], 1)
    hn = h.norm(dim=1, keepdim=True)
    if bool((hn < 1e-6).any()):
        raise ValueError("shower_frames: vertical shower (no horizontal heading); e1 undefined")
    e1 = h / hn
    e2 = torch.cross(d, e1, dim=1)
    return torch.cat([p, d, e1, e2], 1).float()


def place_clouds_shower(clouds: torch.Tensor,
                        positions: torch.Tensor,
                        dirs: torch.Tensor,
                        event_ids: torch.Tensor,
                        species,
                        plane_dx: float = C8_PLANE_DX) -> torch.Tensor:
    """Place native AllShowers clouds in each shower's own frame.

    Native cols [x, y, layer, e, t]: x, y = horizontal (East, North) offsets from
    the shower start (the decay), in fixed global axes, which already include the
    travel along the axis; layer l = C8 plane at depth (l + 1) * plane_dx.

    t1 = x e1_E + y e1_N is the horizontal transverse offset from the axis. Exact:
    e1 has no vertical part and is perpendicular to the heading, so neither the
    lost z nor the travel along the axis enters. The other transverse offset t2
    (~vertical) was not recorded by C8, so showers are taken as circularly
    symmetric: each point is a ring of radius |t1| around the axis, and the kernel
    reads detectors by their distance from the axis.

    Output cols (MODIFIED IN PLACE, energy-carrying points only):
        0, 1: |t1|, 0             ring radius
        2: l + 1                  depth along the axis in units of plane_dx.
        3, 4: energy, time        unchanged.
    `GetCounts_planeaware` and `adaptive_bandwidths` (per-layer, columns 0/1) read it.

    Args:
        clouds    : (M, P, 5) native clouds of ONE species.
        positions : (M, 3) decay vertices (East, North, Up) [m].
        dirs      : (M, 3) tau travel directions (normalised here).
        event_ids : (M,) int, physical event id (unused since the ring placement; kept so
                    callers stay unchanged).
        species   : int or (M,) int, species index (unused, as event_ids).
    Returns:
        frames : (M, 12) float32 [p, d, e1, e2] for `compute_labels_batch(frames=...)`.
    """
    frames = shower_frames(positions, dirs).to(clouds.device)
    e1 = frames[:, 6:9]
    mask = clouds[:, :, 3] > 0
    t1 = clouds[:, :, 0] * e1[:, 0:1] + clouds[:, :, 1] * e1[:, 1:2]
    t1 = torch.where(mask, t1, torch.zeros_like(t1))
    layer = clouds[:, :, 2]
    clouds[..., 0] = torch.where(mask, t1.abs(), clouds[..., 0])
    clouds[..., 1] = torch.where(mask, torch.zeros_like(t1), clouds[..., 1])
    clouds[..., 2] = torch.where(mask, layer + 1.0, clouds[..., 2])
    return frames


def detectors_in_frames(e_det: torch.Tensor, n_det: torch.Tensor, surface,
                        frames: torch.Tensor, plane_dx: float = C8_PLANE_DX):
    """Map one layout (East, North; Up from the surface) into every shower's frame.
    Returns (rho, z) each (M, n_det): rho = distance from the axis (a tiny epsilon
    keeps its gradient finite on the axis), z = depth along the axis / plane_dx.
    Differentiable in (East, North) (through the bilinear surface too)."""
    up = surface(n_det, e_det)
    det = torch.stack([e_det, n_det, up], -1).to(frames.dtype)          # (n_det, 3)
    r = det.unsqueeze(0) - frames[:, None, 0:3]                         # (M, n_det, 3)
    s = (r * frames[:, None, 3:6]).sum(-1)
    t1 = (r * frames[:, None, 6:9]).sum(-1)
    t2 = (r * frames[:, None, 9:12]).sum(-1)
    return torch.sqrt(t1 ** 2 + t2 ** 2 + 1e-6), s / plane_dx


def flag_blob_showers(clouds: torch.Tensor,
                      blob_median_e: float = BLOB_MEDIAN_E) -> torch.Tensor:
    """Per-shower degenerate-cloud flag: median point energy > `blob_median_e`.

    A property of the SHOWER, not of any layout, so it is computed once per
    loaded cloud and replicated across strategies. Padding rows carry energy 0
    and are excluded from the median; a cloud with no energy-carrying point at
    all is not a blob (it is empty, which the E channel already reports as 0).
    A shower carrying any non-finite energy is flagged too — see the comment on
    the return.

    Args:
        clouds : (M, P, 5) point clouds — placed or native, either works, since
                 placement rewrites only the position columns.
    Returns:
        (M,) bool.
    """
    e = clouds[..., 3]
    live = e > 0
    # nan-median over the live points: dead entries become NaN so they neither
    # drag the median down nor need a ragged gather.
    masked = torch.where(live, e, torch.full_like(e, float("nan")))
    med = masked.nanmedian(dim=1).values                  # NaN where no live point
    hot = torch.nan_to_num(med, nan=0.0) > blob_median_e
    # NaN energies would otherwise slip through: `live` rejects them (NaN > 0 is
    # False) so they never reach the median, and the Step-0 underflow guard sorts
    # on `e <= 0`, which is also False for NaN. Since the Step-0 ratio re-roll —
    # the only other place non-finite points were tested — was removed, this is
    # the last check that sees them. +inf needs no special case (inf > cut).
    return hot | ~torch.isfinite(e).all(dim=1)


# ── Label computation (batched over showers, one shared layout per batch) ────

def compute_labels_batch(clouds:   torch.Tensor,
                         e_det:    torch.Tensor,    # East  (detector col 0)
                         n_det:    torch.Tensor,    # North (detector col 1)
                         surface,                    # SurfaceUpMap: (North, East) → Up
                         frames:   torch.Tensor,
                         sigma_spatial: float = SIGMA_SPATIAL,
                         differentiable: bool = False,
                         bandwidth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the ring kernel on a batch of showers sharing one layout.

    Clouds come from `place_clouds_shower`; the layout is mapped into every shower's frame
    (`detectors_in_frames`: distance from the axis, depth along it), and
    `GetCounts_planeaware` reads it there.

    Args:
        clouds : (B, max_points, 5) placed clouds.
        e_det, n_det : (n_det,) detector East / North.
        frames : (B, 12) shower frames from `place_clouds_shower`.
        differentiable : False (default) runs under torch.no_grad() (label building, scoring);
            True keeps the graph for a kernel-in-the-loop Stage 4.
        bandwidth : precomputed (B, max_points) widths for these clouds; None searches them here
            (`adaptive_bandwidths`, k = constants.KERNEL_INTERP_K). Passing them lets a caller that
            labels the same clouds under many layouts search neighbours once.
    Returns:
        E, T : (B, n_det) local intensities and leading-edge arrival times
               (seconds; 0 where no point cleared the detection threshold).
    """
    import contextlib
    ctx = contextlib.nullcontext() if differentiable else torch.no_grad()
    with ctx:
        if bandwidth is None:
            bandwidth = adaptive_bandwidths(clouds, KERNEL_INTERP_K, sigma_spatial)
        rho, z = detectors_in_frames(e_det, n_det, surface, frames.to(clouds.device))
        return GetCounts_planeaware(clouds, rho.to(clouds.dtype), z.to(clouds.dtype), bandwidth,
                                    sigma=sigma_spatial, layer_dx=C8_PLANE_DX)


# ── Dataset builder stages ───────────────────────────────────────────────────

class _CorpusMeta(NamedTuple):
    """Per-shower metadata for the kept rows; clouds are streamed separately."""
    dirs:      torch.Tensor    # (n_showers, 3) unit tau travel direction E,N,U
    positions: torch.Tensor    # (n_showers, 3) ENU decay vertices [m]
    primaries: torch.Tensor    # (n_showers, PRIMARY_DIM) encoded primaries
    species:   torch.Tensor    # (n_showers,) index into constants.SPECIES_NAMES
    n_file:    int             # rows in the corpus file
    per_sp:    int             # file rows per species block
    k_sp:      int             # rows kept per species
    n_species: int             # blocks in the corpus
    n_showers: int             # n_species * k_sp


def _load_corpus_metadata(mountain, shower_cache_path: str,
                          max_showers: int = 0) -> _CorpusMeta:
    """Read the corpus metadata (no point clouds) and encode the primaries.

    The corpus is `len(SPECIES_NAMES)` equal blocks back to back, so
    `max_showers` keeps the first `k_sp` of EACH block, not the first
    `max_showers` rows.
    """
    import showerdata

    n_species = len(SPECIES_NAMES)
    meta   = showerdata.load_inc_particles(shower_cache_path)
    n_file = meta.pdg.shape[0]
    per_sp = n_file // n_species                          # rows per species block
    keep   = n_file if not max_showers else min(int(max_showers), n_file)
    k_sp   = keep // n_species                            # rows kept per species

    keep_idx = np.concatenate([np.arange(s * per_sp, s * per_sp + k_sp)
                               for s in range(n_species)])
    dirs   = torch.as_tensor(meta.directions[keep_idx], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep_idx],   dtype=torch.float32)
    pdg    = torch.as_tensor(meta.pdg[keep_idx],        dtype=torch.long)
    # Real decay vertices: drive both the placement and the primary's rel_E/N/U.
    positions_all = _load_positions_sidecar(shower_cache_path, keep_idx)  # (N,3) E,N,U
    # Array centre from the mesh, not a constant, so it tracks the geometry.
    array_center = torch.as_tensor(mountain.centroids_ENU,
                                   dtype=torch.float32).mean(dim=0)       # (3,) E,N,U
    primaries_all = encode_primary(dirs, energs, pdg,
                                   positions_all, array_center)  # (n_showers, PRIMARY_DIM)

    # Species per kept shower from the Step-0 sidecar (same keep_idx as the
    # metadata). Corpus `pdg` is the EM/hadronic class, so the Step-2 split keys
    # on this sidecar, not on the pdg feature.
    species_all = _load_species_sidecar(shower_cache_path, keep_idx)   # (N,)

    return _CorpusMeta(dirs=dirs, positions=positions_all, primaries=primaries_all,
                       species=species_all, n_file=n_file, per_sp=per_sp,
                       k_sp=k_sp, n_species=n_species, n_showers=n_species * k_sp)


def _build_chunk_list(k_sp: int, per_sp: int, load_chunk: int, batch_size: int):
    """Flat streaming plan, plus the `load_chunk` actually used.

    Flat and precomputed so resume is just "skip the first N".

    `load_chunk` is rounded down to a whole number of `batch_size` sub-batches
    HERE, before the list is built, and the rounded value is returned so no caller
    can build the list off the raw one: a short final sub-batch changes the number
    of per-chunk layout rng draws, and so changes the dataset.

    Returns:
        load_chunk : rounded chunk size.
        chunk_list : list of (tag, file_start, ds_start, c_lo, c_hi).
    """
    load_chunk = max(int(batch_size), (int(load_chunk) // int(batch_size)) * int(batch_size))
    chunk_list = []
    # Species-major, matching the corpus layout. A species appended to
    # SPECIES_NAMES therefore appends its chunks AFTER the existing ones, so the
    # shared layout `rng` produces the same draws for the earlier species as it
    # did before -- their labels are unchanged by the addition.
    for s_i, tag in enumerate(SPECIES_NAMES):
        for c_lo in range(0, k_sp, load_chunk):
            chunk_list.append((tag, s_i * per_sp, s_i * k_sp,
                               c_lo, min(c_lo + load_chunk, k_sp)))
    return load_chunk, chunk_list


class _ResumeState:
    """The accumulating out_* tensors plus their resume checkpoint.

    The interval is WALL-CLOCK, not every-N-chunks: a checkpoint is the whole
    out_* set (~11.8 GB at 750k events), so a per-N policy would scale write
    volume with corpus size. See docs/THEORY.md §11.5.

    The per-chunk layout draws are keyed on (seed, strategy, batch) rather
    than taken from a running generator, so a resumed run reproduces the
    layouts an uninterrupted run would have used for the remaining chunks.
    """

    def __init__(self, n_pairs: int, n_det: int,
                 path: Optional[str] = None,
                 every_s: float = 1800.0,
                 verbose: bool = True):
        self.n_pairs = n_pairs
        self.n_det   = n_det
        self.path    = path
        self.every_s = every_s
        self.verbose = verbose
        self.chunks_done = 0
        self._last_t = time.time()

    def restore(self, n_chunks: int) -> None:
        """Reload a compatible checkpoint, else allocate the out_* tensors fresh."""
        if self.path and os.path.exists(self.path):
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            if ckpt.get("n_pairs") == self.n_pairs:
                self.primary, self.xy = ckpt["out_primary"], ckpt["out_xy"]
                self.E, self.T = ckpt["out_E"], ckpt["out_T"]
                self.strat, self.species = ckpt["out_strat"], ckpt["out_species"]
                self.blob = ckpt["out_blob"]
                self.chunks_done = int(ckpt["chunks_done"])
                if self.verbose:
                    print(f"[resume] {self.path}: {self.chunks_done}/{n_chunks} chunks "
                          "already done")
            elif self.verbose:
                print(f"[resume] {self.path} shape mismatch (n_pairs "
                      f"{ckpt.get('n_pairs')} != {self.n_pairs}) — ignoring, starting fresh")

        if self.chunks_done == 0:
            self.primary = torch.empty((self.n_pairs, PRIMARY_DIM), dtype=torch.float32)
            self.xy      = torch.empty((self.n_pairs, self.n_det, 2), dtype=torch.float32)
            self.E       = torch.empty((self.n_pairs, self.n_det),    dtype=torch.float32)
            self.T       = torch.empty((self.n_pairs, self.n_det),    dtype=torch.float32)
            self.strat   = torch.empty((self.n_pairs,), dtype=torch.int64)
            self.species = torch.empty((self.n_pairs,), dtype=torch.int64)
            self.blob    = torch.zeros((self.n_pairs,), dtype=torch.bool)

        self._last_t = time.time()

    def maybe_checkpoint(self, chunk_i: int, n_chunks: int) -> None:
        """Write a checkpoint if the interval has elapsed.

        Skips the final chunk — it is deleted right after the loop anyway.
        """
        if not self.path or (time.time() - self._last_t) < self.every_s:
            return
        if chunk_i + 1 >= n_chunks:
            return
        t_ck = time.time()
        tmp = self.path + ".tmp"
        torch.save({
            "n_pairs": self.n_pairs, "chunks_done": chunk_i + 1,
            "out_primary": self.primary, "out_xy": self.xy,
            "out_E": self.E, "out_T": self.T,
            "out_strat": self.strat, "out_species": self.species,
            "out_blob": self.blob,
        }, tmp)
        os.replace(tmp, self.path)
        self._last_t = time.time()
        if self.verbose:
            print(f"[resume] checkpointed {chunk_i + 1}/{n_chunks} chunks "
                  f"in {self._last_t - t_ck:.1f}s -> {self.path}")

    def cleanup(self) -> None:
        """Drop the checkpoint once the build has completed."""
        if self.path and os.path.exists(self.path):
            os.remove(self.path)

    def tensors(self):
        return (self.primary, self.xy, self.E, self.T, self.strat,
                self.species, self.blob)


def _label_chunk(clouds_chunk, meta: _CorpusMeta, state: _ResumeState,
                 mountain, surface, seed, frames_chunk=None, *,
                 ds_start: int, c_lo: int, csz: int,
                 batch_size: int, n_det: int, device,
                 was_corrupted: torch.Tensor, strategies=None) -> None:
    """Label one loaded chunk under every strategy, writing into `state`.

    One layout is drawn per (strategy, sub-batch) and shared by the whole
    sub-batch. The draw is keyed on (seed, strategy, batch index) rather than
    taken from a running generator, so every species row of one event receives
    the same layout and a resumed run reproduces an uninterrupted one.

    `strategies` (global ids into `_STRATEGIES`, default all) labels only those:
    the global id keys the layout draw and is written to `state.strat`, so a
    subset build reproduces those blocks of a full build exactly; the position
    in the list sets the output block.
    """
    n_showers = meta.n_showers
    # Degeneracy is a property of the cloud, so it is evaluated once here and
    # written under every strategy — the same shower is a blob in all of them.
    # `was_corrupted` is OR'd in because a fully-inf shower gets zeroed by
    # _load_clouds before this call, which makes flag_blob_showers alone see an
    # indistinguishable "legitimate empty shower" — see _load_clouds's docstring.
    blob = flag_blob_showers(clouds_chunk) | was_corrupted
    # The interpolated kernel's widths depend only on the clouds, so they are
    # searched once per chunk here rather than once per (strategy, sub-batch).
    bw_chunk = torch.cat([
            adaptive_bandwidths(clouds_chunk[lo:lo + batch_size].to(device),
                                KERNEL_INTERP_K, SIGMA_SPATIAL).cpu()
            for lo in range(0, csz, batch_size)])
    if strategies is None:
        strategies = range(len(_STRATEGIES))
    for s_out, s_idx in enumerate(strategies):
        s_name, fn_name, kwargs = _STRATEGIES[s_idx]
        fn = _STRATEGY_FNS[fn_name]
        for sb_lo in range(0, csz, batch_size):
            sb_hi = min(sb_lo + batch_size, csz)
            B = sb_hi - sb_lo

            # Batch index within the species block: every species block walks an
            # identical (c_lo, sb_lo) grid because _build_chunk_list ranges c_lo
            # over k_sp per species and carries the block offset in ds_start, and
            # load_chunk is a whole number of sub-batches. So this lines every
            # species pass onto the same draw.
            b_idx = (c_lo + sb_lo) // batch_size
            e_det, n_det_xy = fn(mountain, n_det=n_det,
                                 rng=_batch_layout_rng(seed, s_idx, b_idx),
                                 **kwargs)  # (East, North)
            e_det     = e_det.float().to(device)
            n_det_xy  = n_det_xy.float().to(device)

            clouds = clouds_chunk[sb_lo:sb_hi].to(device)
            bw = bw_chunk[sb_lo:sb_hi].to(device)
            fr = frames_chunk[sb_lo:sb_hi].to(device)
            E, T = compute_labels_batch(clouds, e_det, n_det_xy, surface, fr, bandwidth=bw)
            E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
            T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

            ds_lo = ds_start + c_lo + sb_lo
            ds_hi = ds_start + c_lo + sb_hi
            dst = slice(s_out * n_showers + ds_lo, s_out * n_showers + ds_hi)
            state.primary[dst]  = meta.primaries[ds_lo:ds_hi]
            state.xy[dst, :, 0] = e_det.cpu().unsqueeze(0).expand(B, -1)     # East  (col 0)
            state.xy[dst, :, 1] = n_det_xy.cpu().unsqueeze(0).expand(B, -1)  # North (col 1)
            state.E[dst] = E.cpu()
            state.T[dst] = T.cpu()
            state.strat[dst] = s_idx
            state.species[dst] = meta.species[ds_lo:ds_hi]
            state.blob[dst] = blob[sb_lo:sb_hi]


def _load_clouds(shower_cache_path: str, meta: _CorpusMeta, chunk):
    """Stream one chunk of point clouds and place it at its real decay vertices.

    Returns (clouds, n_bad, was_corrupted, frames):
        clouds        : placed in the shower frame, with non-finite points zeroed (see below)
        frames        : (csz, 12) shower frames
        n_bad         : count of non-finite POINTS zeroed (float32 energy
                        overflow in the corpus)
        was_corrupted : (csz,) bool, one per SHOWER, true if any of its points
                        needed zeroing

    The AllShowers generator can diverge to a fully-inf point cloud on rare
    high-energy primaries (seen on muon showers past its trained energy range —
    ~0.5% of the muon block once LOG_E_MAX opened 1e7-1e8 GeV). Zeroing those
    points makes the shower's energy column all-zero, i.e. indistinguishable
    from a genuine "detector never triggered" empty shower to `flag_blob_showers`
    downstream, which explicitly treats an all-zero shower as legitimate — so a
    fully-corrupted shower would silently enter training as a normal zero-
    deposit event instead of being excluded. `was_corrupted` is computed BEFORE
    zeroing so the caller can fold it into the blob flag directly; it is not
    recoverable from the zeroed cloud after this function returns.
    """
    import showerdata

    _tag, file_start, ds_start, c_lo, c_hi = chunk
    sub = showerdata.load(shower_cache_path,
                          start=file_start + c_lo, stop=file_start + c_hi)
    clouds_chunk = torch.as_tensor(sub.points, dtype=torch.float32)
    del sub

    bad = ~torch.isfinite(clouds_chunk).all(dim=-1)
    n_bad = int(bad.sum())
    was_corrupted = bad.any(dim=1)
    if n_bad:
        clouds_chunk[bad] = 0.0

    # Rows are species-major (keep_idx = species blocks of k_sp), so the physical
    # event id is row % k_sp and the species row // k_sp —
    # the same ids MainPool (Stage 4) uses for these events.
    rows = torch.arange(ds_start + c_lo, ds_start + c_hi)
    frames = place_clouds_shower(
        clouds_chunk,
        meta.positions[ds_start + c_lo: ds_start + c_hi],   # (csz,3) decay E,N,U
        meta.dirs[ds_start + c_lo: ds_start + c_hi],        # (csz,3) dir (normalised in the frame)
        event_ids=rows % meta.k_sp, species=rows // meta.k_sp)
    return clouds_chunk, n_bad, was_corrupted, frames


def build_training_pairs(mountain, surface,
                         shower_cache_path: str,
                         batch_size:        int = 20,
                         max_showers:       int = 0,
                         seed:              int = 0,
                         device:            torch.device = torch.device("cpu"),
                         verbose:           bool = True,
                         load_chunk:        int = 4096,
                         resume_path:       Optional[str] = None,
                         resume_every_s:    float = 1800.0,
                         strategies:        Optional[list] = None):
    """Build (primary, xy, E, T) training tensors from the cached shower corpus.

    Runs as four stages — `_load_corpus_metadata`, `_build_chunk_list`, then per
    chunk `_load_clouds` (stream + place at the real ENU decay vertex) and
    `_label_chunk` (strategies × sub-batches) — accumulating into a
    `_ResumeState`, which also owns the gpu_requeue resume checkpoint. Point
    clouds are never loaded whole: peak RAM is one chunk, not the corpus.

    Layouts and labels use the (North, East) convention (`detector_strategies`
    plus `compute_labels_batch` above, with `surface` a SurfaceUpMap).

    Returns:
        primaries : (N_pairs, PRIMARY_DIM) float32
        xy        : (N_pairs, 100, 2) float32   columns = (East, North)
        E         : (N_pairs, 100) float32
        T         : (N_pairs, 100) float32
        strategy_ids : (N_pairs,)  int64 — index into `_STRATEGIES` (with
                       `strategies`, only those ids, in that order)
        species_ids  : (N_pairs,)  int64 — index into `constants.SPECIES_NAMES`
        blob_ids     : (N_pairs,)  bool  — degenerate shower, excluded by Step 2
                       (see `flag_blob_showers` / `constants.BLOB_MEDIAN_E`)
    """
    meta    = _load_corpus_metadata(mountain, shower_cache_path, max_showers)
    strategies = list(range(len(_STRATEGIES))) if strategies is None else [int(s) for s in strategies]
    n_strat = len(strategies)
    n_det   = N_DETECTORS

    load_chunk, chunk_list = _build_chunk_list(meta.k_sp, meta.per_sp,
                                               load_chunk, batch_size)
    if verbose:
        print(f"[load] streaming {meta.n_showers} rows ({meta.k_sp}/species) of "
              f"{meta.n_file} in chunks of {load_chunk}; peak RAM ≈ one chunk, "
              "not the corpus")

    state = _ResumeState(meta.n_showers * n_strat, n_det,
                         path=resume_path, every_s=resume_every_s, verbose=verbose)
    state.restore(len(chunk_list))

    n_sanitized = 0

    for chunk_i, chunk in enumerate(chunk_list):
        if chunk_i < state.chunks_done:
            continue
        tag, _file_start, ds_start, c_lo, c_hi = chunk

        clouds_chunk, n_bad, was_corrupted, frames_chunk = _load_clouds(
            shower_cache_path, meta, chunk)
        n_sanitized += n_bad

        _label_chunk(clouds_chunk, meta, state, mountain, surface, seed, frames_chunk,
                     ds_start=ds_start, c_lo=c_lo, csz=c_hi - c_lo,
                     batch_size=batch_size, n_det=n_det, device=device,
                     was_corrupted=was_corrupted, strategies=strategies)

        del clouds_chunk
        if verbose:
            print(f"[build] {tag} rows {c_lo}-{c_hi}/{meta.k_sp} done "
                  f"(×{n_strat} strategies)")
        state.maybe_checkpoint(chunk_i, len(chunk_list))

    state.cleanup()

    if verbose:
        print(f"[place] shower frame at the real decay vertices (N={meta.positions.shape[0]}; "
              f"rings of radius |t1|, depth (l+1)x{C8_PLANE_DX:g} m, times interpolated)")
    if verbose and n_sanitized:
        print(f"[sanitize] zeroed {n_sanitized} non-finite points (float32 energy overflow)")
    if verbose:
        n_blob_rows = int(state.blob.sum())
        print(f"[blob] flagged {n_blob_rows}/{state.n_pairs} pairs "
              f"({100 * n_blob_rows / max(state.n_pairs, 1):.2f}%) as degenerate "
              f"(median point energy > {BLOB_MEDIAN_E:g}); Step 2 excludes them")

    return state.tensors()

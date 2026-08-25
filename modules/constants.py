# ── Paths / constants (match v4's active script) ─────────────────────────────

import os


# Mountain mesh. `load_tr_mountain` rotates ECEF vertices into site-local ENU
# anchored at the mesh's own `location`, so detector centroids share the origin
# used by the tau corpus. `malata` = 266-face detector region, ENU bbox
# North [-956, 716], East [-499, 777], Up [2748, 3712] m.
GEOMETRY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/data/malata.h5"
GEOMETRY_GROUP = "malata"
DET_KEY        = "detector1"

# Prefer a copy of the mesh next to the repo, else the absolute path. Centralized
# so callers don't recompute it with a stale `colca_valley.h5` fallback.
_GEOM_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", os.path.basename(GEOMETRY_PATH))
GEOMETRY_PATH_RESOLVED = _GEOM_LOCAL if os.path.exists(_GEOM_LOCAL) else GEOMETRY_PATH

# AllShowers longitudinal geometry, from decay_locations/c8_air_shower.cpp:
# 24 ObservationPlanes 500 m apart along the shower axis. LAYER_EAST_DX MUST equal
# that real spacing — the old 150 compressed showers to ~30% of their extent.
# EAST_ENTRY is the East gauge of layer 0; it cancels against the detector z_cont,
# so its absolute value is free.
N_PLANES       = 24
EAST_ENTRY     = 1500.0
LAYER_EAST_DX  = 500.0

# Detector spatial-response Gaussian width [m]. 200 → 50 for malata: 100 detectors
# at ~120 m spacing over ~1.4 km, so 200 m over-smoothed neighbours together.
# Set at dataset-build time; the trained surrogate inherits this resolution.
SIGMA_SPATIAL  = 50.0

# A shower point counts as DETECTED at a detector when its deposit there —
# energy * kernel acceptance — clears this. It defines the T label: T is the
# time of the earliest detected point (see showers/kernel.py), so the threshold
# sets where the leading edge is read, exactly like a trigger discriminator.
# Swept over 1e-1..1e-6 on the malata 3-species testset (center_gauss400): the
# TIME barely moves — encoded median 7.078 -> 7.066, std 0.471 -> 0.503 — so the
# leading edge is robust and this only sets how many detectors report at all
# (T > 0 at 27% -> 44% of detectors, per species). 1e-3 gives 35%, which covers
# the 23.6% of detectors carrying E > 1 with margin; looser values only add
# detectors whose whole deposit is marginal.
#
# E > 0 is a WEAKER condition than T > 0 and always will be: E sums many
# sub-threshold deposits, so ~40% of detectors have E > 0 with no single
# detected point and therefore T == 0. Do not treat T == 0 as "E == 0".
HIT_DEPOSIT_MIN = 1e-3

# Degenerate ("blob") showers: the muon AllShowers checkpoint occasionally emits
# a shower wrong in both geometry and energy scale — a diffuse cloud spanning
# ~20 km with median per-point energy orders of magnitude too high, instead of a
# rod. They are RARE and FINITE, so `isfinite`/`nan_to_num` never catch them, and
# a total deposit of 2.06e14 becomes a Step-2 target of ~33 where normal is
# single digits.
#
# A shower is degenerate when the median energy of its energy-carrying points
# exceeds this. Median (not max/total) because the failure is the whole cloud
# being hot, not one outlier point. Measured with tests/compare_upstream_
# generator.py on the eda/degenerate-muon-showers branch: 0.53% of muon showers,
# 0.00% of electron, and bit-identical between our generator fork and upstream —
# so this is intrinsic to the checkpoint, not something the pipeline introduced.
BLOB_MEDIAN_E = 1e3

# Fixed architecture constants
N_DETECTORS = 100
# [dir_x, dir_y, dir_z, log_e_norm, pdg, rel_E, rel_N, rel_U]
#   pdg       = EM/hadronic primary class 0/1 (NOT the e/µ species)
#   rel_E/N/U = decay vertex relative to the array centre [m]
# The vertex is needed because tau_wholesky.jl aims every tau at the array, so
# direction barely discriminates (aleatoric floor R^2 >= 0.49 without it, >= 0.56
# with). Cols 0-3 keep their meaning, so Step 3's `primary[:, :4]` target is
# unchanged. Bumping this invalidates every checkpoint — rebuild from Step 1.
PRIMARY_DIM = 8

# Primary energy bounds (log10 GeV) for min-max normalization, and the band
# load_tau_primaries filters the whole-sky throw down to.
#
# Raised 7.0 -> 8.0 to open the top decade (the old value's comment claimed 1e8
# but 10**7.0 is 1e7, so the decade was never in the corpus). Note the top decade
# is where the AllShowers generator extrapolates: its training set ran out at
# 4.9e7, and a 1e7-1e8 stress run still had the anti-clip retry failing on
# 53/486 electron, 27/486 muon and 52/486 photon showers after 10 attempts.
# Anything trained across it should be checked against those guards.
LOG_E_MIN = 5.0   # log10(1e5 GeV)
LOG_E_MAX = 8.0   # log10(1e8 GeV)

# Direction bounds for sampling primaries
ZENITH_MIN   = 60.0  # degrees
ZENITH_MAX   = 100.0 # degrees
AZIMUTH_MIN  = 0.0   # degrees
AZIMUTH_MAX  = 360.0 # degrees


RUN_LOCATION = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/09_logemax8_64k"
SHOWER_CACHE   = os.path.join(RUN_LOCATION, "v6_run_00")

# Generated figures (EDA PNGs, paper PDFs) live outside the repo — they are
# regenerable output, and 18 MB of them tracked in git bloated every clone.
# Sibling of RUN_LOCATION so the figures are not tied to one run's corpus.
FIGURES_DIR = os.path.join(os.path.dirname(RUN_LOCATION), "figures")
PAPER_FIGURES_DIR = os.path.join(FIGURES_DIR, "paper_figures")
EDA_OUTPUTS_DIR   = os.path.join(FIGURES_DIR, "eda_outputs")

TRAINING_DATASET_FOLDER = os.path.join(RUN_LOCATION, "test_v6_run_01_northeast")
FNN_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_02_recentered")
RECON_FOLDER            = os.path.join(RUN_LOCATION, "test_v6_run_03_recentered")
# 04_optimize.py appends "_{scheme}" (one folder per init scheme).
OPT_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_04_optimize")

# 02: fraction of training-set indices to keep (val set always full).
TRAIN_FRACTION = 1.00
# 01: fraction of the dual corpus to load, applied per species so both stay
# represented. 1.0 is ~501 GB dense and OOMs at --mem=100g.
DATASET_FRACTION = 1.00

# Descriptive only (also the synthetic-primaries fallback count in Step 0).
# With USE_TAU_PRIMARIES the real event count comes from tau_wholesky_n750k.h5
# (~751,931 surviving events) minus the HOLDOUT_FRAC reserve below; Step 1's
# MAX_SHOWERS no longer derives from this (see 01_build_dataset_northeast.py),
# so an approximate value here can't silently truncate the corpus.
NUM_SHOWERS = 750_000
BATCH_SIZE  = 60
BATCH_SIZE_TRAIN  = 20

# ── Held-out final-eval reserve ───────────────────────────────────────────────
# A fixed fraction of PHYSICAL EVENTS, split off in Step 0 before generation and
# written to its own corpus file/sidecars. Steps 1-4 (all FNN/recon train+val
# splits and the stage-4 layout optimizer) only ever read the main corpus below;
# this reserve is exclusively for eval_true_utility.py's final scoring, so a
# layout is never evaluated on the same events used to fit it or train the
# surrogate/recon. Independent seed — unrelated to FNN's (seed 0) or recon's
# (seed 1) own val splits, which stay internal to the main (non-holdout) pool.
HOLDOUT_FRAC = 0.05
HOLDOUT_SEED = 999

# ── Multi-species (paired) pipeline ───────────────────────────────────────────
# The secondary-species components a physical shower is split into. Each has its
# own AllShowers model; a complete event is the SUM of all of them (counts are
# extensive, times count-weighted -- see modules/surrogates/dual.py).
#
# INDEX INTO THIS TUPLE IS THE SPECIES ID written to the Step-0 `_species.pt`
# sidecar and read by Step 1/2 routing. Reordering it silently mislabels every
# existing corpus, exactly as reordering _STRATEGIES invalidates strategy_ids.
# APPEND ONLY. Corpus rows are species-major: block s occupies
# [s*n_pairs, (s+1)*n_pairs), and row s*n_pairs + i is event i's s-component.
SPECIES_NAMES = ("electron", "muon", "photon")

# Step 0 samples NUM_SHOWERS primaries ONCE and generates every component: rows
# i, N+i, 2N+i share one physical event's (energy, direction, EM/hadronic class).
#
# With USE_TAU_PRIMARIES, primaries come from tau_wholesky.h5 (energy, direction,
# and a real ENU decay position) instead of the synthetic sampler. That file uses
# the same site-local ENU frame as the mesh, so showers and mountain share the
# origin. Energies are filtered to [10**LOG_E_MIN, 10**LOG_E_MAX] in the loader.
USE_TAU_PRIMARIES = True
TAU_WHOLESKY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/decay_locations/tau_wholesky_balanced_1e5_1e8_max.h5"

# TAU_WHOLESKY_PATH = ".../decay_locations/tau_wholesky_n2M_83k.h5"  # built the current corpus
TAU_CORPUS_PATH   = os.path.join(SHOWER_CACHE, "cashed_showers_tau_dual.pt")

# Corpus Step 1 reads. Tau runs use a fixed name (pair count is only known after
# energy filtering); synthetic runs keep the count-based name.
DUAL_SHOWER_CACHE_PATH = (
    TAU_CORPUS_PATH if USE_TAU_PRIMARIES
    else os.path.join(SHOWER_CACHE, f"cashed_showers_dual_{2 * NUM_SHOWERS}.pt"))
# Step-0 sidecars, row-aligned with the corpus. Both derive from the corpus path
# by the `<corpus>_*.pt` rule the Step-1 builders use, so they track it.
#   _species.pt   : e/µ component id (0=electron block, 1=muon block) — the corpus
#                   `pdg` carries the EM/hadronic class, so Step 2 splits on this.
#   _positions.pt : (M, 3) ENU decay vertex (East, North, Up); drives both the
#                   placement and the primary encoding's rel_E/N/U.
DUAL_SPECIES_IDS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_species.pt"
DUAL_POSITIONS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_positions.pt"

# Held-out corpus (HOLDOUT_FRAC of events, see above) — same file layout as the
# main corpus/sidecars, written by Step 0, read only by eval_true_utility.py.
HELDOUT_SHOWER_CACHE_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_heldout.pt"
HELDOUT_SPECIES_IDS_PATH  = os.path.splitext(HELDOUT_SHOWER_CACHE_PATH)[0] + "_species.pt"
HELDOUT_POSITIONS_PATH    = os.path.splitext(HELDOUT_SHOWER_CACHE_PATH)[0] + "_positions.pt"

# 02 log-compresses T targets as log1p(T*T_LOG_SCALE); surrogates/dual.py must
# invert the same transform, so the scale lives here.
#
# Lowered 1e8 -> 1e6. Hit arrival times span only 127x (1.11e-6 s to 1.41e-4 s),
# so at 1e8 log1p mapped them to [4.72, 9.55]: half the target's dynamic range
# was a constant offset carrying no information, and the net had to emit ~6.9
# just to reach the mean. 1e6 gives [0.75, 4.95] -- the dead offset drops from
# 49% of the range to 15%.
#
# It also shrinks, but does not close, the gap that is the real difficulty here:
# 64% of detector-samples are exactly dark (T = 0) and every hit is above the
# light-travel floor, so the target is bimodal with nothing in between and an
# MSE fit must place predictions inside that empty gap. Separating the hit/dark
# decision from the time regression is the actual fix; this only narrows the gap
# from 4.72 to 0.75.
#
# NOTE: plots/training/02_nn_target_vs_pred.py and plots/training/aleatoric_floor.py
# keep their own literal copies of this value -- change all three together.
T_LOG_SCALE = 1.0e6

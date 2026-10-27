"""Pinned inputs of the data lane: CORSIKA showers, the generator code and weights, and run-12 held-out events."""
import hashlib
import importlib.util
import os
import subprocess
import sys
import types

import h5py
import numpy as np
import torch

from modules.constants import HELDOUT_POSITIONS_PATH, HELDOUT_SHOWER_CACHE_PATH, SPECIES_NAMES

LAB = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone"
CORSIKA_DIR = f"{LAB}/hhanif/tambo_simulations_for_training/h5_files_v3"
CORSIKA_NAME = {"electron": "electrons", "muon": "muons", "photon": "photons"}
CAP = {"electron": 4096, "muon": 25088, "photon": 8064}

SRC = f"{LAB}/jagarwal/data_lane_src"
# Two pinned sources of the generator code: the AllShowers package with its TAMBO-opt wrapper, and
# the TAMBO-opt fork the pipeline imports, which carries both.
CODE = {"upstream": {"repos": {"AllShowers": "2eea4dec7492b767b2824b1488d9c37d7e4a599b",
                            "TAMBO-opt": "67bfdce05a14900726aea9198bdd926900ca2ad3"},
                  "package": "AllShowers/allshowers",
                  "wrapper": "TAMBO-opt/util/allshowers_related/generate_showers.py"},
        "fork": {"repos": {"TAMBO-opt-fork": "1fafe7eeeabbe1d866b2c2ced22b8b06bce8d6e3"},
                 "package": "TAMBO-opt-fork/allshowers",
                 "wrapper": "TAMBO-opt-fork/util/allshowers_related/generate_showers.py"}}
_active = {}
CKPT = f"{LAB}/zdimitrov/detector_optimization_v6/checkpoints"
STAGED = f"{LAB}/zdimitrov/detector_optimization_v6/13_full_750k_no_sampling/allshowers_staged"
RUN_DIR = {"electron": "20260519_185649_Electron-Allshower",
           "muon": "20260520_160031_Muons-Allshower",
           "photon": "20260724_074020_Photon-Allshower"}
WEIGHTS_MD5 = {"electron": "6d48b593fe", "muon": "ed0c4c9f97", "photon": "a479c60e75"}
STAGE0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "00_generate_data_dual_species.py")

if tuple(CAP) != tuple(SPECIES_NAMES):
    raise RuntimeError(f"species order {tuple(CAP)} differs from constants {SPECIES_NAMES}")


def corsika_path(species, split):
    suffix = "_test" if split == "test" else ""
    return os.path.join(CORSIKA_DIR, f"combined_{CORSIKA_NAME[species]}{suffix}.h5")


def _same(meta, other, species):
    for k in ("ids", "energies", "directions", "labels"):
        if not np.array_equal(meta[k], other[k]):
            raise ValueError(f"{species}: {k} differ from {SPECIES_NAMES[0]}; species rows are misaligned")


def corsika(split, rows, path=corsika_path):
    """CORSIKA showers at `rows`: native (B, cap, 5) clouds per species and their shared primaries."""
    rows = np.sort(np.asarray(rows))
    clouds, meta = {}, None
    for s in SPECIES_NAMES:
        with h5py.File(path(s, split), "r") as f:
            n_feat = int(f["shape"][2])
            if n_feat != 5:
                raise ValueError(f"{s}: {n_feat} features per point, expected 5")
            m = {"ids": f["shower_ids"][rows], "energies": f["energies"][rows].reshape(-1, 1),
                 "directions": f["directions"][rows], "labels": f["pdg"][rows].astype(np.int64),
                 "actual_pdg": f["actual_pdg"][rows], "num_points": f["num_points"][rows]}
            c = np.zeros((len(rows), CAP[s], 5), np.float32)
            for i, r in enumerate(rows):
                p = np.asarray(f["showers"][r], np.float32).reshape(-1, 5)
                if len(p) != m["num_points"][i] or len(p) > CAP[s]:
                    raise ValueError(f"{s} row {r}: {len(p)} points, num_points {m['num_points'][i]}, cap {CAP[s]}")
                c[i, :len(p)] = p
        if meta is None:
            meta = m
        else:
            _same(meta, m, s)
        clouds[s] = torch.from_numpy(c)
    meta = {k: torch.as_tensor(v) for k, v in meta.items() if k != "num_points"}
    meta["energies"], meta["directions"] = meta["energies"].float(), meta["directions"].float()
    return clouds, meta


def heldout(n, start=0):
    """Run-12 held-out events start to start + n: native clouds per species, generator conditions and decay vertices."""
    import showerdata
    positions = torch.load(HELDOUT_POSITIONS_PATH)
    n_pairs = positions.shape[0] // len(SPECIES_NAMES)
    if start + n > n_pairs:
        raise ValueError(f"events {start} to {start + n} requested, {n_pairs} held out")
    clouds, meta = {}, None
    for i, s in enumerate(SPECIES_NAMES):
        sh = showerdata.load(HELDOUT_SHOWER_CACHE_PATH, start=i * n_pairs + start, stop=i * n_pairs + start + n)
        m = {"ids": np.asarray(sh.shower_ids), "energies": np.asarray(sh.energies, np.float32).reshape(-1, 1),
             "directions": np.asarray(sh.directions, np.float32), "labels": np.asarray(sh.pdg, np.int64)}
        if meta is None:
            meta = m
        else:
            _same(meta, m, s)
        clouds[s] = torch.as_tensor(sh.points, dtype=torch.float32)
    meta = {k: torch.as_tensor(v) for k, v in meta.items()}
    meta["positions"] = positions[start:start + n].float()
    return clouds, meta


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _git(repo, *cmd):
    return subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True).stdout.strip()


def generator_code(code="upstream"):
    """Import one pinned allshowers package and expose its generate_showers wrapper under both names the pipeline imports."""
    if "allshowers" in sys.modules:
        raise RuntimeError(f"allshowers already imported from {sys.modules['allshowers'].__path__}")
    cfg = CODE[code]
    for repo, commit in cfg["repos"].items():
        path = os.path.join(SRC, repo)
        if _git(path, "rev-parse", "HEAD") != commit or _git(path, "status", "--porcelain"):
            raise RuntimeError(f"{path} is not a clean checkout of {commit}")
    _active["code"] = code
    # allshowers has no __init__.py, so as a namespace package it would merge every copy on the path
    # (the env carries an editable install of another checkout). Register it with our folder only.
    allshowers = types.ModuleType("allshowers")
    allshowers.__path__ = [os.path.join(SRC, cfg["package"])]
    sys.modules["allshowers"] = allshowers
    _check_from_src(allshowers)
    spec = importlib.util.spec_from_file_location("allshowers.generate_showers", os.path.join(SRC, cfg["wrapper"]))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    for name in ("util", "util.allshowers_related"):
        if name in sys.modules:
            raise RuntimeError(f"a module named {name} is already imported")
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__path__ = []
    sys.modules["allshowers.generate_showers"] = wrapper
    sys.modules["util.allshowers_related.generate_showers"] = wrapper
    return cfg["repos"]


def _check_from_src(module):
    """The (namespace) package and its loaded file must come from the pinned checkout only."""
    paths = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", []))
    paths = [os.path.realpath(p) for p in paths if p]
    if not paths or not all(p.startswith(os.path.join(SRC, CODE[_active["code"]]["package"])) for p in paths):
        raise RuntimeError(f"{module.__name__} resolved to {paths}")


def stage0(code="upstream"):
    """The pipeline's own Step 0 module, imported against one pinned generator code set."""
    generator_code(code)
    spec = importlib.util.spec_from_file_location("stage0", STAGE0)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("allshowers", "allshowers.generator", "allshowers.transformer"):
        _check_from_src(sys.modules[name])
    return module


def weights(species):
    """Staged flow weights of one species, checked against the hash shared by runs 13, 17 and 18."""
    path = os.path.join(STAGED, species, "weights", "best.pt")
    if not _md5(path).startswith(WEIGHTS_MD5[species]):
        raise RuntimeError(f"{path} does not match md5 {WEIGHTS_MD5[species]}")
    return path


@torch.no_grad()
def generate(s0, energies, directions, labels, seed, species=SPECIES_NAMES):
    """Native clouds for the given primaries, through the pipeline's Step 0 chunk (PointCountFM, anti-clip re-roll, AllShowers)."""
    from allshowers.generator import Generator
    ids = torch.arange(len(energies))
    out = {}
    for k, s in enumerate(species):
        torch.manual_seed(seed * 1000 + k)
        if _active["code"] == "fork":   # as the pipeline: staged run dir (conf with pre_ln, weights/best.pt)
            weights(s)
            gen = Generator(run_dir=os.path.join(STAGED, s), num_timesteps=s0.NUM_TIMESTEPS,
                            compile=True, solver=s0.SOLVER)
        else:
            gen = Generator(run_dir=os.path.join(CKPT, RUN_DIR[s]), num_timesteps=s0.NUM_TIMESTEPS,
                            compile=True, solver=s0.SOLVER, checkpoint=weights(s))
        gen.max_points = CAP[s]
        sh = s0._gen_chunk(gen, os.path.join(STAGED, s, "pcfm_compiled.pt"), {"max_points": CAP[s]}, s,
                           energies, directions, labels, ids, CAP[s])
        out[s] = torch.as_tensor(sh.points, dtype=torch.float32)
        del gen
        torch.cuda.empty_cache()
    return out

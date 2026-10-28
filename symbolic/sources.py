"""Pinned inputs: the CORSIKA training files and the deployed neural chain (code checkout and staged weights)."""
import hashlib
import importlib.util
import os
import subprocess
import sys
import types

from modules.constants import SPECIES_NAMES

LAB = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone"
CORSIKA_DIR = f"{LAB}/hhanif/tambo_simulations_for_training/h5_files_v3"
CORSIKA_NAME = {"electron": "electrons", "muon": "muons", "photon": "photons"}
CAP = {"electron": 4096, "muon": 25088, "photon": 8064}       # cells kept per shower by the preprocessing

SRC = f"{LAB}/jagarwal/data_lane_src"
CODE = {"repo": "TAMBO-opt-fork", "commit": "1fafe7eeeabbe1d866b2c2ced22b8b06bce8d6e3",
        "package": "TAMBO-opt-fork/allshowers", "wrapper": "TAMBO-opt-fork/util/allshowers_related/generate_showers.py"}
STAGED = f"{LAB}/zdimitrov/detector_optimization_v6/13_full_750k_no_sampling/allshowers_staged"
WEIGHTS_MD5 = {"electron": "6d48b593fe", "muon": "ed0c4c9f97", "photon": "a479c60e75"}
STAGE0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "00_generate_data_dual_species.py")

if tuple(CAP) != tuple(SPECIES_NAMES):
    raise RuntimeError(f"species order {tuple(CAP)} differs from constants {SPECIES_NAMES}")


def corsika_path(species, split):
    suffix = "_test" if split == "test" else ""
    return os.path.join(CORSIKA_DIR, f"combined_{CORSIKA_NAME[species]}{suffix}.h5")


def _git(repo, *cmd):
    return subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True).stdout.strip()


def _check_from_src(module):
    paths = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", []))
    paths = [os.path.realpath(p) for p in paths if p]
    if not paths or not all(p.startswith(os.path.join(SRC, CODE["package"])) for p in paths):
        raise RuntimeError(f"{module.__name__} resolved to {paths}")


def _import_generator_code():
    if "allshowers" in sys.modules:
        raise RuntimeError(f"allshowers already imported from {sys.modules['allshowers'].__path__}")
    path = os.path.join(SRC, CODE["repo"])
    if _git(path, "rev-parse", "HEAD") != CODE["commit"] or _git(path, "status", "--porcelain"):
        raise RuntimeError(f"{path} is not a clean checkout of {CODE['commit']}")
    # allshowers has no __init__.py; register it with the pinned folder only, so no other copy on the path merges in
    allshowers = types.ModuleType("allshowers")
    allshowers.__path__ = [os.path.join(SRC, CODE["package"])]
    sys.modules["allshowers"] = allshowers
    _check_from_src(allshowers)
    spec = importlib.util.spec_from_file_location("allshowers.generate_showers", os.path.join(SRC, CODE["wrapper"]))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    for name in ("util", "util.allshowers_related"):
        if name in sys.modules:
            raise RuntimeError(f"a module named {name} is already imported")
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__path__ = []
    sys.modules["allshowers.generate_showers"] = wrapper
    sys.modules["util.allshowers_related.generate_showers"] = wrapper


def stage0():
    """The pipeline's own shower-generation module, imported against the pinned generator code."""
    _import_generator_code()
    spec = importlib.util.spec_from_file_location("stage0", STAGE0)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("allshowers", "allshowers.generator", "allshowers.transformer"):
        _check_from_src(sys.modules[name])
    return module


def weights(species):
    """Staged flow weights of one species, checked against their hash."""
    path = os.path.join(STAGED, species, "weights", "best.pt")
    if not hashlib.md5(open(path, "rb").read()).hexdigest().startswith(WEIGHTS_MD5[species]):
        raise RuntimeError(f"{path} does not match md5 {WEIGHTS_MD5[species]}")
    return path

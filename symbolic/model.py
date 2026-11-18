"""The model folder: part 1 (part1.npz, the sampler of each class) and the in-plane laws (rings, width, times, energy,
spread .npz, keys <species>.<class>[.<what>]), each written by fit.py next to a .json of what it was fitted on."""
import json
import os

import numpy as np

from modules.constants import SPECIES_NAMES
from symbolic import data, sampler

LAWS = ("rings", "width", "times", "energy", "spread")
FILES = ("part1",) + LAWS


def load(folder):
    """-> ({class: sampler}, {(species, class): {rings, width, lead, delay, energy, spread}})."""
    z = np.load(os.path.join(folder, "part1.npz"))
    part1 = {cn: sampler.from_arrays(z, cn) for cn in data.CLASS.values()}
    f = {w: np.load(os.path.join(folder, f"{w}.npz")) for w in LAWS}
    laws = {}
    for s in SPECIES_NAMES:
        for cn in data.CLASS.values():
            key = f"{s}.{cn}"
            laws[(s, cn)] = {"rings": f["rings"][key], "width": f["width"][key], "lead": f["times"][f"{key}.lead"],
                             "delay": f["times"][f"{key}.delay"], "energy": f["energy"][key], "spread": f["spread"][key]}
    return part1, laws


def showers(folder):
    """Every shower table any part of the model was fitted on."""
    seen = set()
    for w in FILES:
        with open(os.path.join(folder, f"{w}.json")) as h:
            seen |= set(json.load(h)["showers"])
    return seen


def save(folder, name, arrays, info):
    """<folder>/<name>.npz and <name>.json; refuses to overwrite."""
    path = os.path.join(folder, f"{name}.npz")
    if os.path.exists(path):
        raise ValueError(f"exists, refusing: {path}")
    os.makedirs(folder, exist_ok=True)
    np.savez(path, **arrays)
    with open(os.path.join(folder, f"{name}.json"), "w") as h:
        json.dump({"git": data.git_head(), **info}, h, indent=1)

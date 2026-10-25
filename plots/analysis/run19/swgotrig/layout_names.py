"""Validation layouts of a run, in table order: (key, label) from <D>/validate_count_layouts.txt (name=path lines),
so the table scripts work for any run (run 19's plane, run 20's Malata). Labels: a fixed name for the optimised
layouts, parsed from the name for the baselines (grid_n99, disk_r<R>_n99, tri33_r<r>_n99, tri33disk_R<R>_r<r>_n99).
"""
import os
import re

OPT = {"opt_a2_grid": "Opt. Poisson 2 m², grid start", "opt_a2_center": "Opt. Poisson 2 m², centre start",
       "opt_a20_grid": "Opt. Poisson 20 m², grid start",
       "opt_a2_step_grid_5000": "Opt. step 2 m², grid start (5,000 steps)",
       "opt_a2_step_grid_8000": "Opt. step 2 m², grid start (8,000 steps)",
       "opt_a2_step_grid": "Opt. step 2 m², grid start", "opt_a2_step_center": "Opt. step 2 m², centre start"}


def _metres(v):
    v = int(v)
    return f"{v / 1000:g} km" if v >= 1000 else f"{v} m"


def label(key, short=False):
    if key in OPT:
        return OPT[key].replace(" 2 m²", "").replace(", grid start", ", grid").replace(", centre start", ", centre") \
            .replace(" (5,000 steps)", " (5k)").replace(" (8,000 steps)", " (8k)") if short else OPT[key]
    if key.startswith("grid_n"):
        return "Grid"
    m = re.fullmatch(r"disk_r(\d+)_n\d+", key)
    if m:
        return f"{'Disk' if short else 'Uniform disk'} {_metres(m.group(1))}"
    m = re.fullmatch(r"tri\d+_r(\d+)_n\d+", key)
    if m:
        return f"Triplets {m.group(1)} m" if short else f"Triplets r = {m.group(1)} m on grid"
    m = re.fullmatch(r"tri\d+disk_R(\d+)_r(\d+)_n\d+", key)
    if m:
        return f"Triplets, {_metres(m.group(1))} disk" if short else f"Triplets r = {m.group(2)} m on {_metres(m.group(1))} disk"
    return key


def layouts(d, short=False):
    """[(key, label)]: the baselines, then the optimised layouts, each in the order of
    <d>/validate_count_layouts.txt."""
    keys = [l.split("=", 1)[0].strip() for l in open(os.path.join(d, "validate_count_layouts.txt")) if l.strip()]
    keys = [k for k in keys if not k.startswith("opt_")] + [k for k in keys if k.startswith("opt_")]
    return [(k, label(k, short)) for k in keys]

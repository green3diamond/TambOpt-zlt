"""Output folder, log file, results.json with provenance, and figure saving, shared by the symbolic scripts."""
import json
import logging
import os
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})


def start(name, args):
    """Create <args.out>/<name>/{run.log, figures/} and return (folder, logger)."""
    out = os.path.join(args.out, name)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out, "run.log"))):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("args %s", json.dumps(vars(args)))
    return out, log


def _git(*cmd):
    return subprocess.run(["git", "-C", REPO, *cmd], capture_output=True, text=True).stdout.strip()


def save(out, results, args, log):
    """Write results.json with the provenance of this run."""
    provenance = {"args": vars(args), "git": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
                  "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(out, "results.json")
    with open(path, "w") as f:
        json.dump({"provenance": provenance, **results}, f, indent=2)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))


def save_fig(fig, out, name, log):
    path = os.path.join(out, "figures", f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)

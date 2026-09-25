"""Output folder, log file and results.json shared by every analysis script."""
import json
import logging
import os
import socket
import subprocess
import sys
import time

from analysis_core.run import REPO


def start(name, args):
    """Create <out>/<name>/{run.log, figures/} and return (folder, logger)."""
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
    provenance = {"args": vars(args), "git": _git("rev-parse", "HEAD"),
                  "dirty": bool(_git("status", "--porcelain")), "host": socket.gethostname(),
                  "job": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(out, "results.json")
    with open(path, "w") as f:
        json.dump({"provenance": provenance, **results}, f, indent=2)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))

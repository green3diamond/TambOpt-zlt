"""One pipeline run: frozen models, mountain, surface and named layouts."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "plots", "layouts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from modules import constants as C
from modules.geometry import (SurfaceUpMap, load_tr_mountain,
                              project_to_mountain_ne, sample_initial_layout_ne)
from modules.geometry.placement import _ne_max_gap
from modules.optimize import load_models

ENSEMBLE_DIR = "test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_{}"
SCHEMES = ("grid", "center")
LAYOUTS = ("grid", "center", "opt_grid", "opt_center")


def add_run_args(ap):
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True, help="results root; each script writes a subfolder")
    ap.add_argument("--n-events", type=int, default=5120)
    ap.add_argument("--chunk", type=int, default=64, help="events per chunk for paired errors")
    ap.add_argument("--kernel-chunk", type=int, default=128, help="events per kernel call")
    ap.add_argument("--seed", type=int, default=0)


class Run:
    def __init__(self, run_dir, device):
        # The event loader reads corpus paths from constants, so the two must agree.
        if os.path.realpath(run_dir) != os.path.realpath(C.RUN_LOCATION):
            raise SystemExit(f"--run-dir {run_dir} differs from constants.RUN_LOCATION "
                             f"{C.RUN_LOCATION}")
        self.dir = run_dir
        self.device = device
        self.mountain = load_tr_mountain(
            C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
            layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
        self.surface = SurfaceUpMap.from_mountain(self.mountain).to(device)
        self.fnn, self.recon = load_models(device)
        self.max_gap = _ne_max_gap(self.mountain)
        self._centroids = torch.as_tensor(self.mountain.centroids_ENU[:, :2], dtype=torch.float32)

    def snap(self, e, n):
        e, n = project_to_mountain_ne(self.mountain, e.float().reshape(-1), n.float().reshape(-1))
        return e.float(), n.float()

    def layout(self, name):
        """(East, North) of grid, center, opt_grid or opt_center, snapped to the mountain."""
        if name in SCHEMES:
            e, n = sample_initial_layout_ne(self.mountain, n_units=C.N_DETECTORS, scheme=name)
            return self.snap(torch.as_tensor(np.asarray(e)), torch.as_tensor(np.asarray(n)))
        if name not in LAYOUTS:
            raise ValueError(f"unknown layout {name!r}")
        d = self._load(name.split("_", 1)[1], "layout_best.pt")
        return self.snap(d["x"], d["y"])

    def chains(self, scheme):
        """Every chain's final layout for one scheme, as a list of (East, North)."""
        return [self.snap(a[:, 0], a[:, 1]) for a in self._load(scheme, "layouts_all.pt")["aligned"]]

    def on_mountain(self, e, n):
        """True where a point lies within the snap tolerance of the mesh."""
        pts = torch.stack([e.float().reshape(-1), n.float().reshape(-1)], -1)
        d = torch.cat([torch.cdist(p, self._centroids).min(1).values for p in pts.split(512)])
        return d <= self.max_gap

    def _load(self, scheme, fname):
        path = os.path.join(self.dir, ENSEMBLE_DIR.format(scheme), fname)
        return torch.load(path, map_location="cpu", weights_only=False)

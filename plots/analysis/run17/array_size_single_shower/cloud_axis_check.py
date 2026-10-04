"""Does each placed shower cloud point along its labelled direction? For the first --n held-out events
(electron block): an energy-weighted principal axis through the decay point, fitted to the placed
points in ENU, compared with the tau direction; plus the points' North / Up offset from the true axis
in the layers the plane covers (z_cont 3-7.5). Zenith agrees by construction: place_clouds_enu builds
Up from the labelled d_Up, while East / North are the generator's own horizontal offsets.

    python cloud_axis_check.py [--n 200]
"""
import argparse
import sys

import numpy as np
import showerdata
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")   # branch run17-bwcap50 (run 17 constants)
import modules.constants as C  # noqa: E402
from modules.data import place_clouds_enu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
B = ap.parse_args().n
p = torch.load(C.HELDOUT_POSITIONS_PATH).float()[:B]
ip = showerdata.load_inc_particles(C.HELDOUT_SHOWER_CACHE_PATH, 0, B)
d = torch.as_tensor(ip.directions).float()
d = d / d.norm(dim=1, keepdim=True)
c = torch.as_tensor(showerdata.load(C.HELDOUT_SHOWER_CACHE_PATH, start=0, stop=B).points).float()
place_clouds_enu(c, p, d, east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX)

ang, daz, dze, offN, offU = [], [], [], [], []
for i in range(B):
    x = c[i]
    w = x[:, 3]
    k = w > 0
    x, w = x[k].numpy().astype(float), w[k].numpy().astype(float)
    if len(x) < 50:
        continue
    E = C.EAST_ENTRY - C.LAYER_EAST_DX * x[:, 2]
    R = np.stack([E, x[:, 0], x[:, 1]], 1) - p[i].numpy()
    v = np.linalg.eigh((R * w[:, None]).T @ R)[1][:, -1]
    t = d[i].numpy()
    v = v if v @ t > 0 else -v
    ang.append(np.degrees(np.arccos(np.clip(v @ t, -1, 1))))
    daz.append((np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(t[1], t[0])) + 180) % 360 - 180)
    dze.append(np.degrees(np.arccos(v[2]) - np.arccos(t[2])))
    s = (E - p[i, 0].item()) / t[0]                      # axis parameter at each point's East
    on = (x[:, 2] > 3) & (x[:, 2] < 7.5)
    if on.sum() >= 5:
        offN.append(np.average(x[on, 0] - (p[i, 1].item() + s[on] * t[1]), weights=w[on]))
        offU.append(np.average(x[on, 1] - (p[i, 2].item() + s[on] * t[2]), weights=w[on]))


def row(name, a, unit):
    a = np.asarray(a)
    print(f"{name:42s} median {np.median(a):+7.1f}{unit}   16-84%: {np.percentile(a, 16):+7.1f} .. "
          f"{np.percentile(a, 84):+7.1f}   ({len(a)} events)")


row("cloud axis vs labelled direction", ang, " deg")
row("azimuth, cloud - labelled", daz, " deg")
row("zenith, cloud - labelled", dze, " deg")
row("North offset from the axis on the plane", offN, " m")
row("Up offset from the axis on the plane", offU, " m")

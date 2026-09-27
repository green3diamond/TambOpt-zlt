#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import showerdata

'''
python pointcountfm/inference_cond_file_plot.py \
  /n/home04/hhanif/PointCountFM/results/20260301_125736_PointCountFM/compiled.pt \
  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_test.h5 \
  --start 0 --stop 20 --num-layers 24 \
  --pdg-codes 0 1 \
  --showers-dset showers \
  --plot-dir plots \
  --max-plots 100
'''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="TorchScript compiled model (.pt)")
    p.add_argument("input_file", type=str, help="Input H5 file")

    p.add_argument("--start", type=int, default=0, help="Start shower index (inclusive)")
    p.add_argument("--stop", type=int, default=-1, help="Stop shower index (exclusive). -1 => end")
    p.add_argument("--num-layers", type=int, default=24, help="Number of layers")

    p.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        help="PDG code list defining class ordering for one-hot.",
    )

    p.add_argument(
        "--showers-dset",
        type=str,
        default="showers",
        help="Name of variable-length showers dataset (flat [x,y,z,e,...])",
    )

    p.add_argument(
        "--plot-dir",
        type=str,
        default="plots_compare",
        help="Base directory to write plots into (a timestamped subdir is created).",
    )

    # Optional: to avoid saving hundreds of plots by accident
    p.add_argument(
        "--max-plots",
        type=int,
        default=-1,
        help="If >0, only save up to this many per-shower plots.",
    )

    return p.parse_args()


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def calc_num_points_per_layer_h5(h5_dataset, start: int, stop: int, num_layers: int) -> np.ndarray:
    """
    Count number of hits per layer for each shower using e > 0 mask.
    Reads directly from h5py dataset (variable-length flat arrays).

    layer index = (z + 0.1).astype(int32)
    """
    num_showers = stop - start
    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)

    for i, global_i in enumerate(range(start, stop)):
        shower = np.array(h5_dataset[global_i])
        if shower.size == 0:
            continue
        points = shower.reshape(-1, 4)  # (n_hits, 4): x,y,z,e
        layer_idx = np.clip((points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        mask = (points[:, 3] > 0).astype(np.int32)  # e > 0
        np.add.at(points_per_layer[i], layer_idx, mask)

    return points_per_layer


class InferenceModel(torch.nn.Module):
    """Wrap model to run per-shower (avoids any model batching assumptions)."""
    def __init__(self, model: torch.jit.ScriptModule):
        super().__init__()
        self.model = model

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(conditions.size(0)):
            results.append(self.model(conditions[[i]]))
        return torch.cat(results, dim=0)


def main():
    args = parse_args()
    torch.set_num_threads(1)

    # Resolve stop
    file_len = showerdata.get_file_length(args.input_file)
    start = max(0, args.start)
    stop = file_len if args.stop == -1 else min(args.stop, file_len)
    if stop <= start:
        raise ValueError(f"Invalid range: start={start}, stop={stop}, file_len={file_len}")

    n = stop - start
    num_layers = args.num_layers

    # ----- Load ML model -----
    print("Loading model...")
    t0 = time.time()
    model = torch.jit.load(args.model)
    model.eval()
    inference = InferenceModel(model).to(torch.float32)
    inference = torch.jit.script(inference)
    print(f"Model+wrapper ready in {(time.time() - t0) * 1000.0:.1f} ms")

    # ----- Load conditioning data for [start, stop) -----
    print(f"Loading conditioning data for showers [{start}, {stop}) (N={n}) ...")
    t0 = time.time()
    cond_data = showerdata.load_inc_particles(args.input_file, start=start, stop=stop)

    labels = to_labels(torch.from_numpy(cond_data.pdg), args.pdg_codes)
    if (labels < 0).any():
        bad = int((labels < 0).sum().item())
        raise ValueError(
            f"{bad} showers have PDG not in --pdg-codes. "
            f"Update --pdg-codes to include all PDGs present."
        )

    conditions = torch.concatenate(
        (
            torch.from_numpy(cond_data.energies).to(torch.float32),
            torch.nn.functional.one_hot(labels, num_classes=len(args.pdg_codes)).to(torch.float32),
            torch.from_numpy(cond_data.directions).to(torch.float32),
        ),
        dim=1,
    )
    print(f"Conditioning loaded in {(time.time() - t0) * 1000.0:.1f} ms")

    # ----- ML inference -----
    print("Running ML inference...")
    t0 = time.time()
    ml = inference(conditions)  # (N, num_layers)
    dt = time.time() - t0
    print(f"Inference done in {dt:.2f} s ({(dt / n) * 1000.0:.3f} ms / shower)")

    ml = (torch.clamp(ml, min=0.0) + 0.5).to(torch.int32).cpu().numpy()
    if ml.ndim != 2 or ml.shape[1] != num_layers:
        raise ValueError(f"ML output has shape {ml.shape}, expected (N, {num_layers})")

    # ----- Ground truth -----
    print("Computing ground truth from H5 showers dataset...")
    t0 = time.time()
    with h5py.File(args.input_file, "r") as hf:
        if args.showers_dset not in hf:
            raise KeyError(
                f"Dataset '{args.showers_dset}' not found in {args.input_file}. "
                f"Available keys: {list(hf.keys())}"
            )
        gt = calc_num_points_per_layer_h5(hf[args.showers_dset], start, stop, num_layers)
    print(f"Ground truth computed in {(time.time() - t0):.2f} s")

    # ----- Plot per shower -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.plot_dir) / f"compare_{ts}_start{start}_stop{stop}"
    per_dir = outdir / "per_shower"
    per_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(num_layers, dtype=np.int32)

    max_plots = args.max_plots
    to_plot = n if max_plots is None or max_plots < 0 else min(n, max_plots)

    print(f"Saving {to_plot} per-shower plots to: {per_dir}")

    for i in range(to_plot):
        global_idx = start + i

        ml_i = ml[i]
        gt_i = gt[i]

        plt.figure()

        plt.plot(
            x,
            ml_i,
            linestyle="None",
            marker="o",
            label="ML",
        )

        plt.plot(
            x,
            gt_i,
            linestyle="None",
            marker="s",
            label="Ground truth (CORSIKA)",
        )

        plt.xlabel("layer_idx")
        plt.ylabel("num_points")
        plt.title(f"Shower {global_idx}: num points per layer (ML vs GT)")

        # Show every layer tick explicitly
        plt.xticks(x)

        plt.legend()
        plt.tight_layout()

        plot_path = per_dir / f"shower_{global_idx:07d}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  saved {i+1}/{to_plot} plots...")

    print("\nDone.")
    print(f"Plots saved in: {per_dir}")
    print(f"Output root: {outdir}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
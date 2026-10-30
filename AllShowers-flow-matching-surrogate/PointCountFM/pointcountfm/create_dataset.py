#!/usr/bin/env python3

'''
python pointcountfm/create_dataset.py \
  --input  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons.h5  \
  --output /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_for_layers.h5  \
  --num-layers 24 

python pointcountfm/create_dataset.py \
  --input  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_muons.h5  \
  --output /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_muons_for_layers.h5  \
  --num-layers 24 


python pointcountfm/create_dataset.py \
  --input  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons.h5  \
  --output /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons_for_layers.h5  \
  --num-layers 24 
'''

import argparse
import os
import h5py
import numpy as np
import showerdata


# -------------------------------------------------
# Label utilities (same ordering as create_label_list)
# -------------------------------------------------
def create_label_list_numpy(pdg_1d: np.ndarray) -> list[int]:
    unique = np.unique(pdg_1d).tolist()
    unique.sort(key=lambda x: (abs(int(x)), -int(x)))
    return [int(x) for x in unique]


def pdg_to_label_numpy(pdg_1d: np.ndarray, label_list: list[int]) -> np.ndarray:
    label_map = {pdg_val: i for i, pdg_val in enumerate(label_list)}
    return np.fromiter(
        (label_map[int(x)] for x in pdg_1d),
        dtype=np.int32,
        count=pdg_1d.size,
    )


# -------------------------------------------------
# num_points per layer (e > 0 mask, direct from h5py)
# -------------------------------------------------
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
        points = shower.reshape(-1, 5)                          # (n_particles, 5): x,y,z,e,t
        layer_idx = np.clip(
            (points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1
        )
        mask = (points[:, 3] > 0).astype(np.int32)             # e > 0
        np.add.at(points_per_layer[i], layer_idx, mask)

    return points_per_layer


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create minimal HDF5 with directions, energies, labels, num_points"
    )
    parser.add_argument(
        "--input",
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    if os.path.exists(args.output):
        if not args.overwrite:
            raise FileExistsError("Output exists. Use --overwrite.")
        os.remove(args.output)

    # -------------------------------------------------
    # Read PDG once (input is read-only)
    # -------------------------------------------------
    with h5py.File(args.input, "r") as hin:
        N = hin["pdg"].shape[0]
        pdg_all = hin["pdg"][:].astype(np.int32)

    label_list = create_label_list_numpy(pdg_all)

    # -------------------------------------------------
    # Create new minimal file
    # -------------------------------------------------
    with h5py.File(args.output, "w") as hout, h5py.File(args.input, "r") as hin:

        # Create datasets
        d_dir = hout.create_dataset(
            "directions",
            shape=hin["directions"].shape,
            dtype=hin["directions"].dtype,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_en = hout.create_dataset(
            "energies",
            shape=hin["energies"].shape,
            dtype=hin["energies"].dtype,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_lab = hout.create_dataset(
            "labels",
            shape=(N,),
            dtype=np.int32,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_np = hout.create_dataset(
            "num_points",
            shape=(N, args.num_layers),
            dtype=np.int32,
            chunks=(min(args.chunk_size, N), args.num_layers),
            compression="gzip",
            shuffle=True,
        )

        # Save metadata
        hout.attrs["label_list"] = np.array(label_list, dtype=np.int32)
        hout.attrs["num_layers"] = np.int32(args.num_layers)

        # -------------------------------------------------
        # Copy + compute in chunks
        # -------------------------------------------------
        step = args.chunk_size

        for start in range(0, N, step):
            stop = min(N, start + step)

            # Copy directions & energies
            d_dir[start:stop] = hin["directions"][start:stop]
            d_en[start:stop] = hin["energies"][start:stop]

            # Labels
            d_lab[start:stop] = pdg_to_label_numpy(
                pdg_all[start:stop], label_list
            )

            # Compute num_points using e > 0 mask directly from h5py
            np_chunk = calc_num_points_per_layer_h5(
                hin["showers"], start=start, stop=stop, num_layers=args.num_layers
            )

            d_np[start:stop] = np_chunk

            if stop % (step * 10) == 0 or stop == N:
                print(f"Processed {stop}/{N}")

    print("\nDone.")
    print(f"Output file: {args.output}")
    with h5py.File(args.output, "r") as hout:
        print("Datasets inside:")
        for name in ["directions", "energies", "labels", "num_points"]:
            print(f"  - {name}: {hout[name].shape}")


if __name__ == "__main__":
    main()
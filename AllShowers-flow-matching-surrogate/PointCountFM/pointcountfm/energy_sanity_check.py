#!/usr/bin/env python3
import numpy as np
import h5py
import showerdata

INPUT_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5"
CHUNK_SIZE = 50000


def main():

    with h5py.File(INPUT_FILE, "r") as hf:
        total_showers = hf["pdg"].shape[0]

    negative_hits = 0
    zero_hits = 0
    total_hits = 0

    print("\nChecking energies across all showers...\n")

    for start in range(0, total_showers, CHUNK_SIZE):
        stop = min(total_showers, start + CHUNK_SIZE)

        showers = showerdata.load(INPUT_FILE, start=start, stop=stop)
        energies = showers.points[..., 3]

        negative_hits += np.sum(energies < 0)
        zero_hits += np.sum(energies == 0)
        total_hits += energies.size   # <-- total number of hits

        print(f"Processed {stop}/{total_showers}")

    print("\n===== RESULT =====")
    print("Total hits:          ", int(total_hits))
    print("Negative energy hits:", int(negative_hits))
    print("Zero energy hits:    ", int(zero_hits))

    if negative_hits == 0:
        print("\n✅ No negative energies found.")
    else:
        print("\n⚠️ Negative energies detected!")


if __name__ == "__main__":
    main()
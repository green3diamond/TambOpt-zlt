#!/usr/bin/env python3
import h5py
import numpy as np
import showerdata


INPUT_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5"
NUM_SHOWERS = 5
NUM_LAYERS = 24


# -------------------------------------------------
# Threshold mask (exact logic you provided)
# -------------------------------------------------
def compute_threshold_mask(showers, detector_config=None):
    if detector_config is None:
        return showers.points[..., 3] > 0

    ecal_mask = (
        (showers.points[..., 2] < detector_config.num_layers_ecal - 0.5)
        & (showers.points[..., 3] > detector_config.ecal_threshold / 1e3)
    )

    hcal_mask = (
        (showers.points[..., 2] >= detector_config.num_layers_ecal - 0.5)
        & (showers.points[..., 3] > detector_config.hcal_threshold / 1e3)
    )

    return ecal_mask | hcal_mask


# -------------------------------------------------
# Num points per layer (WITH mask)
# -------------------------------------------------
def calc_num_points_per_layer_masked(showers, num_layers=24, detector_config=None):

    num_showers = len(showers)

    layer_idx = (showers.points[..., 2] + 0.1).astype(np.int32)

    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)

    shower_indices = (
        np.arange(num_showers)
        .reshape(-1, 1)
        .repeat(showers.points.shape[1], axis=1)
    )

    mask = compute_threshold_mask(showers, detector_config).astype(np.int32)

    layer_idx = np.clip(layer_idx, 0, num_layers - 1)

    np.add.at(points_per_layer, (shower_indices, layer_idx), mask)

    return points_per_layer


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    print("\nLoading first 5 showers...\n")

    showers = showerdata.load(INPUT_FILE, stop=NUM_SHOWERS)

    computed = calc_num_points_per_layer_masked(
        showers, num_layers=NUM_LAYERS, detector_config=None
    )

    with h5py.File(INPUT_FILE, "r") as hf:
        existing = hf["observables/num_points_per_layer"][:NUM_SHOWERS]

    print("Computed:")
    print(computed)

    print("\nExisting:")
    print(existing)

    diff = computed - existing

    print("\nDifference (computed - existing):")
    print(diff)

    if np.array_equal(computed, existing):
        print("\n✅ PERFECT MATCH")
    else:
        print("\n❌ MISMATCH FOUND")
        print("Max absolute difference:", np.max(np.abs(diff)))

        # Extra debug
        for i in range(NUM_SHOWERS):
            if not np.array_equal(computed[i], existing[i]):
                print(f"\nShower {i} differs:")
                print("Computed:", computed[i])
                print("Existing:", existing[i])
                break


if __name__ == "__main__":
    main()
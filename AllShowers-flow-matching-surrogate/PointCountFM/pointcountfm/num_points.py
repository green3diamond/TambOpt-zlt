import h5py
import numpy as np

INPUT_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5"
NUM_SHOWERS = 5000
NUM_LAYERS = 24

# --- Compute from showers ---
computed = np.zeros((NUM_SHOWERS, NUM_LAYERS), dtype=np.int32)
with h5py.File(INPUT_FILE, 'r') as f:
    dataset = f['showers']
    for i in range(min(NUM_SHOWERS, len(dataset))):
        shower = np.array(dataset[i])
        points = shower.reshape(-1, 4)
        layer_idx = (points[:, 2] + 0.1).astype(np.int32)
        mask = (points[:, 3] > 0).astype(np.int32)
        np.add.at(computed[i], layer_idx, mask)

# --- Load stored observable ---
with h5py.File(INPUT_FILE, 'r') as f:
    stored = f['observables/num_points_per_layer'][:NUM_SHOWERS]

print(f"Computed shape: {computed.shape}")
print(f"Stored   shape: {stored.shape}")

# --- Compare ---
match = np.array_equal(computed, stored)
print(f"\nExact match: {match}")

if not match:
    diff = computed - stored
    n_mismatch = np.sum(diff != 0)
    print(f"Mismatched elements: {n_mismatch} / {diff.size}")
    print(f"Max absolute difference: {np.abs(diff).max()}")
    print(f"Mean absolute difference: {np.abs(diff).mean():.6f}")

    # Show first few mismatching showers
    mismatch_showers = np.where((diff != 0).any(axis=1))[0]
    print(f"\nFirst 5 mismatching shower indices: {mismatch_showers[:5]}")
    for idx in mismatch_showers[:3]:
        print(f"\nShower {idx}:")
        print(f"  Computed: {computed[idx]}")
        print(f"  Stored:   {stored[idx]}")
        print(f"  Diff:     {diff[idx]}")
else:
    print("\nComputed values perfectly match stored observables!")
    print(f"Sample (first 3 showers):\n{computed[1:2]}")
    print(f"Sample (first 3 showers):\n{stored[1:2]}")
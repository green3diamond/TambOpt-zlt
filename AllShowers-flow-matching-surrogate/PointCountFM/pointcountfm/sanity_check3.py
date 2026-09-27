import h5py
import numpy as np

INPUT_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5"

with h5py.File(INPUT_FILE, 'r') as f:
    showers = f['showers'][:20000]

print(f"Number of showers: {len(showers)}")
print(f"Example shower length: {len(showers[0])}, n_particles: {len(showers[0]) // 4}\n")

count = 0
for i, shower in enumerate(showers):
    shower = np.array(shower)
    n_particles = len(shower) // 4
    reshaped = shower.reshape(n_particles, 4)  # x, y, z, e
    bad_mask = reshaped[:, 3] <= 0
    if bad_mask.any():
        bad = reshaped[bad_mask]
        print(f"Shower {i}:")
        for p in bad:
            print(f"  x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f}, e={p[3]:.4f}")
        count += 1
        if count >= 20:
            break

if count == 0:
    print("No showers found with negative or zero energy.")
else:
    print(f"\nFound {count} showers with negative/zero energy (showing up to 20).")
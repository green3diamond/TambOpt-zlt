import pandas as pd
import awkward as ak
import yaml
import numpy as np
import os
import shlex
from tqdm import tqdm

base_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/tambo_optimization/hamza_test_set"

fields = [
    "pdg", "name", "total_energy", "kinetic_energy",
    "x", "y", "z", "nx", "ny", "nz", "time"
]

output_dir = "../ml/processed_events"
os.makedirs(output_dir, exist_ok=True)

# List all event directories
event_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# event_dirs=event_dirs[0:10]

for event_id in tqdm(event_dirs, desc="Processing events"):
    event_path = os.path.join(base_dir, event_id)

    # --- Safety checks at event level ---
    config_path = os.path.join(event_path, "config.yaml")
    primary_path = os.path.join(event_path, "primary/summary.yaml")
    particles_path = os.path.join(event_path, "particles/particles.parquet")

    if not (os.path.exists(config_path) and os.path.exists(primary_path) and os.path.exists(particles_path)):
        print(f"Skipping {event_id} (missing required files)")
        continue
    # -----------------------------------

    # Load event config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Parse args into a dictionary
    args = config.get("args", "")
    tokens = shlex.split(args)

    arg_dict = {}
    for i, token in enumerate(tokens):
        if token.startswith("--"):
            key = token.lstrip("-")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                arg_dict[key] = tokens[i + 1]
            else:
                arg_dict[key] = True

    injection_height = float(arg_dict.get("injection-height", "0"))
    zenith = float(arg_dict.get("zenith", "0"))
    azimuth = float(arg_dict.get("azimuth", "0"))

    # Load primary summary
    with open(primary_path) as f:
        primary = yaml.safe_load(f)

    if isinstance(primary, dict) and len(primary) == 1 and isinstance(next(iter(primary.values())), dict):
        primary = next(iter(primary.values()))

    # First pass: find min/max time
    tmin, tmax = np.inf, -np.inf
    for i in range(0, 24):
        folder_name = "particles" if i == 0 else f"particles{i}"
        particledir = os.path.join(event_path, folder_name)
        parquet_file = os.path.join(particledir, "particles.parquet")

        if not os.path.exists(parquet_file):
            continue

        akf = ak.from_parquet(parquet_file)
        if "time" not in ak.fields(akf) or len(akf["time"]) == 0:
            continue

        times = akf["time"]
        tmin = min(tmin, np.min(times))
        tmax = max(tmax, np.max(times))

    # Collect all planes into one event DataFrame
    all_planes = []
    for i in range(0, 24):
        folder_name = "particles" if i == 0 else f"particles{i}"
        particledir = os.path.join(event_path, folder_name)
        parquet_file = os.path.join(particledir, "particles.parquet")
        plane_config = os.path.join(particledir, "config.yaml")

        if not (os.path.exists(parquet_file) and os.path.exists(plane_config)):
            continue

        akf = ak.from_parquet(parquet_file)

        # Load plane info
        with open(plane_config) as f:
            pconfig = yaml.safe_load(f)

        center = np.array(pconfig["plane"]["center"])
        zhat = np.array(pconfig["plane"]["normal"])
        xhat = np.array(pconfig["x-axis"])
        yhat = np.array(pconfig["y-axis"])
        mat = np.array([xhat, yhat, zhat])

        x = ak.to_numpy(akf["x"])
        y = ak.to_numpy(akf["y"])
        zeros = np.zeros(len(x), dtype=float)

        coords = np.matmul(
            mat,
            np.vstack([x, y, zeros])
        ) + center[:, None]
        df = ak.to_dataframe(akf)

        # Add transformed columns
        df["time_transformed"] = (akf["time"] - tmin) / (tmax - tmin)
        df["X_transformed"] = coords[0, :]
        df["Y_transformed"] = coords[1, :]
        df["Z_transformed"] = coords[2, :]

        df["injection_height"] = injection_height
        df["zenith"] = zenith
        df["azimuth"] = azimuth

        # Add sin(theta) and cos(theta) where theta = azimuth
        df["sin_azimuth"] = np.sin(np.deg2rad(azimuth))
        df["cos_azimuth"] = np.cos(np.deg2rad(azimuth))

        df["sin_zenith"] = np.sin(np.deg2rad(zenith))
        df["cos_zenith"] = np.cos(np.deg2rad(zenith))
        # Add distance
        if i == 0:
            df["distance"] = 12000.0
        else:
            df["distance"] = 500.0 * i



        # Add primary info
        for key in fields:
            df[f"primary_{key}"] = primary.get(key, 0)

        df["plane"] = float(i)
        df["event_id"] = event_id

        all_planes.append(df)

    if all_planes:
        event_df = pd.concat(all_planes, ignore_index=True)

        # Filter only for the desired PDG codes

        # Make PDG codes positive
        event_df["pdg"] = event_df["pdg"].abs()

        # Filter only for the desired PDG codes (now all positive)
        keep_pdgs = [11, 13, 22]
        event_df = event_df[event_df["pdg"].isin(keep_pdgs)]


        # Convert all columns to float except "name"
        for col in event_df.columns:
            if col != "name":
                event_df[col] = event_df[col].astype(float, errors="ignore")


        # Randomly sample 10,000 rows if larger
        if len(event_df) > 10_000:
            event_df = event_df.sample(n=10_000, random_state=42)

        out_file = os.path.join(output_dir, f"{event_id}.parquet")
        event_df.to_parquet(out_file, index=False)

print("✅ All events processed and written to", output_dir)

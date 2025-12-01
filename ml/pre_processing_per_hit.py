import pandas as pd
import awkward as ak
import yaml
import numpy as np
import os
import shlex
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_events.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

base_dir = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/tambo_optimization/hamza_test_set/monoenergetic/"


fields = [
    "pdg", "name", "total_energy", "kinetic_energy",
    "x", "y", "z", "nx", "ny", "nz", "time"
]

output_dir = "../ml/processed_events"
os.makedirs(output_dir, exist_ok=True)

try:
    event_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
except Exception as e:
    logger.error(f"Error listing base directory {base_dir}: {e}")
    event_dirs = []

for event_id in tqdm(event_dirs, desc="Processing events"):
    try:
        event_path = os.path.join(base_dir, event_id)

        config_path = os.path.join(event_path, "config.yaml")
        primary_path = os.path.join(event_path, "primary/summary.yaml")
        particles_path = os.path.join(event_path, "particles/particles.parquet")

        if not (os.path.exists(config_path) and os.path.exists(primary_path) and os.path.exists(particles_path)):
            logger.warning(f"Skipping {event_id} (missing required files)")
            continue

        with open(config_path) as f:
            try:
                config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed parsing config.yaml in {event_id}: {e}")
                continue

        args = config.get("args", "")
        try:
            tokens = shlex.split(args)
        except Exception as e:
            logger.error(f"Failed parsing args string in {event_id}: {e}")
            tokens = []

        arg_dict = {}
        for i, token in enumerate(tokens):
            if token.startswith("--"):
                key = token.lstrip("-")
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    arg_dict[key] = tokens[i + 1]
                else:
                    arg_dict[key] = True

        try:
            injection_height = float(arg_dict.get("injection-height", "0"))
            zenith = float(arg_dict.get("zenith", "0"))
            azimuth = float(arg_dict.get("azimuth", "0"))
        except ValueError as e:
            logger.error(f"Error converting injection args to float in {event_id}: {e}")
            injection_height = 0.0
            zenith = 0.0
            azimuth = 0.0

        with open(primary_path) as f:
            try:
                primary = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed parsing primary summary.yaml in {event_id}: {e}")
                continue

        if isinstance(primary, dict) and len(primary) == 1 and isinstance(next(iter(primary.values())), dict):
            primary = next(iter(primary.values()))

        tmin, tmax = np.inf, -np.inf
        for i in range(0, 24):
            folder_name = "particles" if i == 0 else f"particles{i}"
            particledir = os.path.join(event_path, folder_name)
            parquet_file = os.path.join(particledir, "particles.parquet")

            if not os.path.exists(parquet_file):
                continue

            try:
                akf = ak.from_parquet(parquet_file)
            except Exception as e:
                logger.error(f"Failed to read parquet {parquet_file} in {event_id}: {e}")
                continue

            if "time" not in ak.fields(akf) or len(akf["time"]) == 0:
                continue

            times = akf["time"]
            tmin = min(tmin, np.min(times))
            tmax = max(tmax, np.max(times))

        all_planes = []
        for i in range(0, 24):
            folder_name = "particles" if i == 0 else f"particles{i}"
            particledir = os.path.join(event_path, folder_name)
            parquet_file = os.path.join(particledir, "particles.parquet")
            plane_config = os.path.join(particledir, "config.yaml")

            if not (os.path.exists(parquet_file) and os.path.exists(plane_config)):
                continue

            try:
                akf = ak.from_parquet(parquet_file)
            except Exception as e:
                logger.error(f"Failed to load parquet {parquet_file} in {event_id}: {e}")
                continue

            try:
                with open(plane_config) as f:
                    pconfig = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed loading plane config {plane_config} in {event_id}: {e}")
                continue

            try:
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

                df["time_transformed"] = (akf["time"] - tmin) / (tmax - tmin)
                df["X_transformed"] = coords[0, :]
                df["Y_transformed"] = coords[1, :]
                df["Z_transformed"] = coords[2, :]

                df["injection_height"] = injection_height
                df["zenith"] = zenith
                df["azimuth"] = azimuth

                df["sin_azimuth"] = np.sin(np.deg2rad(azimuth))
                df["cos_azimuth"] = np.cos(np.deg2rad(azimuth))

                df["sin_zenith"] = np.sin(np.deg2rad(zenith))
                df["cos_zenith"] = np.cos(np.deg2rad(zenith))

                df["distance"] = 12000.0 if i == 0 else 500.0 * i

                for key in fields:
                    df[f"primary_{key}"] = primary.get(key, 0)

                df["plane"] = float(i)
                df["event_id"] = event_id

                all_planes.append(df)
            except Exception as e:
                logger.error(f"Error processing plane {i} for event {event_id}: {e}")
                continue

        if all_planes:
            event_df = pd.concat(all_planes, ignore_index=True)

            event_df["pdg"] = event_df["pdg"].abs()

            keep_pdgs = [11, 13, 22]
            event_df = event_df[event_df["pdg"].isin(keep_pdgs)]

            for col in event_df.columns:
                if col != "name":
                    try:
                        event_df[col] = event_df[col].astype(float, errors="ignore")
                    except Exception as e:
                        logger.warning(f"Failed to convert column {col} in {event_id}: {e}")

            if len(event_df) > 10_000:
                event_df = event_df.sample(n=10_000, random_state=42)

            out_file = os.path.join(output_dir, f"{event_id}.parquet")
            try:
                event_df.to_parquet(out_file, index=False)
            except Exception as e:
                logger.error(f"Failed to write parquet for {event_id}: {e}")
                continue

        else:
            logger.warning(f"No planes processed for event {event_id}")

    except Exception as e:
        logger.error(f"Unexpected error processing event {event_id}: {e}")

logger.info("✅ All events processed and written to " + output_dir)

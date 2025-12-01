import pandas as pd
import awkward as ak
import yaml
import numpy as np
import os
import shlex
from tqdm import tqdm
import logging
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
import time

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

base_dir = "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulation"
output_dir = "../ml/processed_events_50k"
os.makedirs(output_dir, exist_ok=True)

fields = [
    "pdg", "name", "total_energy", "kinetic_energy",
    "x", "y", "z", "nx", "ny", "nz", "time"
]

def process_single_event(event_id, base_dir, output_dir, fields):
    """Process a single event - multiprocessing target function"""
    try:
        event_path = os.path.join(base_dir, event_id)
        config_path = os.path.join(event_path, "config.yaml")
        primary_path = os.path.join(event_path, "primary/summary.yaml")
        particles_path = os.path.join(event_path, "particles/particles.parquet")

        if not (os.path.exists(config_path) and os.path.exists(primary_path) and os.path.exists(particles_path)):
            return event_id, False, "missing required files"

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)
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

        # Load primary
        with open(primary_path) as f:
            primary = yaml.safe_load(f)
        if isinstance(primary, dict) and len(primary) == 1:
            primary = next(iter(primary.values()))

        # Find time bounds
        tmin, tmax = np.inf, -np.inf
        for i in range(24):
            folder_name = "particles" if i == 0 else f"particles{i}"
            parquet_file = os.path.join(event_path, folder_name, "particles.parquet")
            if not os.path.exists(parquet_file):
                continue
            try:
                akf = ak.from_parquet(parquet_file)
                if "time" in ak.fields(akf) and len(akf["time"]) > 0:
                    times = akf["time"]
                    tmin = min(tmin, np.min(times))
                    tmax = max(tmax, np.max(times))
                del akf
            except:
                continue
        if tmin == np.inf or tmax == -np.inf:
            tmin, tmax = 0.0, 1.0

        # Process planes
        all_planes = []
        for i in range(24):
            folder_name = "particles" if i == 0 else f"particles{i}"
            particledir = os.path.join(event_path, folder_name)
            parquet_file = os.path.join(particledir, "particles.parquet")
            plane_config = os.path.join(particledir, "config.yaml")

            if not (os.path.exists(parquet_file) and os.path.exists(plane_config)):
                continue

            try:
                akf = ak.from_parquet(parquet_file)
                with open(plane_config) as f:
                    pconfig = yaml.safe_load(f)

                center = np.array(pconfig["plane"]["center"])
                zhat = np.array(pconfig["plane"]["normal"])
                xhat = np.array(pconfig["x-axis"])
                yhat = np.array(pconfig["y-axis"])
                mat = np.array([xhat, yhat, zhat])

                # FILTER + DOWNSAMPLE AWKWARD ARRAY
                mask = (akf["pdg"] == 11) | (akf["pdg"] == 13) | (akf["pdg"] == 22)
                akf_filtered = akf[mask]
                if len(akf_filtered) == 0:
                    del akf, pconfig
                    continue

                max_per_plane = 1000
                if len(akf_filtered) > max_per_plane:
                    sample_idx = np.random.choice(len(akf_filtered), size=max_per_plane, replace=False)
                    akf = akf_filtered[ak.Array(sample_idx)]
                else:
                    akf = akf_filtered
    
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
                del akf, x, y, coords, pconfig
            except Exception as e:
                continue

        if all_planes:
            event_df = pd.concat(all_planes, ignore_index=True)
            for col in event_df.columns:
                if col != "name":
                    try:
                        event_df[col] = pd.to_numeric(event_df[col])
                    except:
                        pass
            if len(event_df) > 10_000:
                event_df = event_df.sample(n=10_000, random_state=42)

            out_file = os.path.join(output_dir, f"{event_id}.parquet")
            event_df.to_parquet(out_file, index=False)
            del event_df, all_planes
            gc.collect()
            return event_id, True, "success"
        else:
            return event_id, False, "no planes processed"

    except Exception as e:
        return event_id, False, str(e)

# Main execution
try:
    event_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
except Exception as e:
    logger.error(f"Error listing base directory {base_dir}: {e}")
    event_dirs = []

logger.info(f"Found {len(event_dirs)} events")

# Use 75% of available CPUs (leave some for system)
num_workers = max(1, cpu_count())
logger.info(f"Using {num_workers} parallel workers (total CPUs: {cpu_count()})")

# Process in chunks to avoid memory explosion
chunk_size = 100
total_processed = 0
results = []

with tqdm(total=len(event_dirs), desc="Processing events") as pbar:
    for i in range(0, len(event_dirs), chunk_size):
        chunk = event_dirs[i:i + chunk_size]
        
        # Partial function with fixed args
        process_func = partial(process_single_event, base_dir=base_dir, 
                             output_dir=output_dir, fields=fields)
        
        with Pool(num_workers) as pool:
            chunk_results = pool.starmap(process_func, [(event_id,) for event_id in chunk])
        
        # Count successes and log
        successes = [r[1] for r in chunk_results]
        total_processed += sum(successes)
        results.extend(chunk_results)
        
        logger.info(f"Chunk {i//chunk_size + 1}: {sum(successes)}/{len(chunk)} successful")
        pbar.update(len(chunk))
        
        # Clean up
        gc.collect()

# Summary
successful = sum(1 for r in results if r[1])
errors = [r for r in results if not r[1]]
logger.info(f"✅ COMPLETE: {successful}/{len(event_dirs)} successful ({successful/len(event_dirs)*100:.1f}%)")
logger.info(f"Output directory: {output_dir}")

if errors:
    logger.info(f"Failed events ({len(errors)}): {len(set([r[2] for r in errors if r[2] != 'no planes processed']))} unique errors")

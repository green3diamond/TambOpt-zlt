import pandas as pd
import awkward as ak
import yaml
import numpy as np
import os
import shlex
from tqdm import tqdm
import logging
import gc
import multiprocessing as mp
import time

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler("process_events.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

base_dir = "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulation"

fields = [
    "pdg", "name", "total_energy", "kinetic_energy",
    "x", "y", "z", "nx", "ny", "nz", "time"
]

output_dir = "../ml/processed_events_50k"
os.makedirs(output_dir, exist_ok=True)


def process_one_event(event_id: str):
    """Exactly your single-event logic, but wrapped in a function."""
    try:
        event_path = os.path.join(base_dir, event_id)
        config_path = os.path.join(event_path, "config.yaml")
        primary_path = os.path.join(event_path, "primary/summary.yaml")
        particles_path = os.path.join(event_path, "particles/particles.parquet")

        if not (os.path.exists(config_path) and os.path.exists(primary_path) and os.path.exists(particles_path)):
            return False, f"{event_id}: missing required files"

        # ---- load config ----
        with open(config_path) as f:
            config = yaml.safe_load(f)

        args = config.get("args", "")
        try:
            tokens = shlex.split(args)
        except Exception as e:
            logger.error(f"{event_id}: Failed parsing args string: {e}")
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
            logger.error(f"{event_id}: Error converting injection args to float: {e}")
            injection_height = 0.0
            zenith = 0.0
            azimuth = 0.0

        # ---- load primary ----
        with open(primary_path) as f:
            primary = yaml.safe_load(f)

        if isinstance(primary, dict) and len(primary) == 1 and isinstance(next(iter(primary.values())), dict):
            primary = next(iter(primary.values()))

        # ---- first pass: time bounds ----
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
                logger.error(f"{event_id}: Failed to read parquet {parquet_file}: {e}")
                continue

            if "time" not in ak.fields(akf) or len(akf["time"]) == 0:
                del akf
                continue

            times = akf["time"]
            tmin = min(tmin, np.min(times))
            tmax = max(tmax, np.max(times))
            del akf, times

        if tmin == np.inf or tmax == -np.inf:
            logger.warning(f"{event_id}: No time data, using defaults")
            tmin, tmax = 0.0, 1.0

        # ---- planes ----
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
                logger.error(f"{event_id}: Failed to load parquet {parquet_file}: {e}")
                continue

            try:
                with open(plane_config) as f:
                    pconfig = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"{event_id}: Failed loading plane config {plane_config}: {e}")
                del akf
                continue

            try:
                center = np.array(pconfig["plane"]["center"])
                zhat = np.array(pconfig["plane"]["normal"])
                xhat = np.array(pconfig["x-axis"])
                yhat = np.array(pconfig["y-axis"])
                mat = np.array([xhat, yhat, zhat])

                # filter + downsample in awkward
                mask = (akf["pdg"] == 11) | (akf["pdg"] == 13) | (akf["pdg"] == 22)
                akf_filtered = akf[mask]

                if len(akf_filtered) == 0:
                    del akf, pconfig
                    continue

                max_per_plane = 1000
                if len(akf_filtered) > max_per_plane:
                    idx = np.random.choice(len(akf_filtered), size=max_per_plane, replace=False)
                    akf_plane = akf_filtered[ak.Array(idx)]
                else:
                    akf_plane = akf_filtered

                x = ak.to_numpy(akf_plane["x"])
                y = ak.to_numpy(akf_plane["y"])
                zeros = np.zeros(len(x), dtype=float)

                coords = np.matmul(
                    mat,
                    np.vstack([x, y, zeros])
                ) + center[:, None]

                df = ak.to_dataframe(akf_plane)

                df["time_transformed"] = (akf_plane["time"] - tmin) / (tmax - tmin)
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

                del akf, akf_filtered, akf_plane, x, y, zeros, coords, pconfig

            except Exception as e:
                logger.error(f"{event_id}: Error processing plane {i}: {e}")
                continue

        if not all_planes:
            return False, f"{event_id}: no planes processed"

        event_df = pd.concat(all_planes, ignore_index=True)

        for col in event_df.columns:
            if col != "name":
                try:
                    event_df[col] = pd.to_numeric(event_df[col])
                except Exception:
                    pass

        if len(event_df) > 10_000:
            event_df = event_df.sample(n=10_000, random_state=42)

        out_file = os.path.join(output_dir, f"{event_id}.parquet")
        event_df.to_parquet(out_file, index=False)

        del event_df, all_planes
        gc.collect()
        return True, f"{event_id}: success"

    except Exception as e:
        return False, f"{event_id}: {e}"


def worker(event_queue: mp.Queue, result_queue: mp.Queue):
    """Process loop for each CPU."""
    while True:
        event_id = event_queue.get()
        if event_id is None:
            break
        ok, msg = process_one_event(event_id)
        result_queue.put((ok, msg))


if __name__ == "__main__":
    n_cpus = len(os.sched_getaffinity(0))
    print(f"Number of CPUs allocated: {n_cpus}")

    try:
        event_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        logger.error(f"Error listing base directory {base_dir}: {e}")
        event_dirs = []

    logger.info(f"Found {len(event_dirs)} events")

    event_queue = mp.Queue()
    result_queue = mp.Queue()

    # Spawn one worker per available CPU
    processes = [
        mp.Process(target=worker, args=(event_queue, result_queue))
        for _ in range(n_cpus)
    ]

    for p in processes:
        p.start()

    # Enqueue all events
    for ev in event_dirs:
        event_queue.put(ev)

    # Send stop signals
    for _ in processes:
        event_queue.put(None)

    # Progress + collect results
    successes = 0
    with tqdm(total=len(event_dirs), desc="Processing events") as pbar:
        for _ in range(len(event_dirs)):
            ok, msg = result_queue.get()
            if ok:
                successes += 1
            else:
                logger.warning(msg)
            pbar.update(1)

    for p in processes:
        p.join()

    logger.info(f"✅ All events processed: {successes}/{len(event_dirs)} successful")
    logger.info(f"Output directory: {output_dir}")

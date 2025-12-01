import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import logging
import gc
from pathlib import Path

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler("process_cone_params.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Input and output directories
input_dir = "../ml/processed_events_50k"
output_file = "../ml/processed_events_50k/event_cone_parameters.parquet"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def calculate_cone_parameters(event_data: pd.DataFrame) -> pd.Series:
    """Calculate cone parameters for a single event."""
    try:
        batch_identifier = "event_id"
        
        # Filter for planes > 0 to find min_plane
        valid_planes = event_data[event_data['plane'] > 0]
        if valid_planes.empty:
            return pd.Series({
                'event_id': event_data['event_id'].iloc[0],
                'min_plane': 0, 'max_plane': 0,
                'X_mean_min': 0, 'Y_mean_min': 0, 'Z_mean_min': 0,
                'X_mean_max': 0, 'Y_mean_max': 0, 'Z_mean_max': 0,
                'radius': 0
            })
        
        # min plane is the lowest non-zero plane
        min_plane = valid_planes['plane'].min()
        max_plane = 0  # Assuming plane 0 is the max plane (ground/detector)
        
        # get average x,y,z from min plane
        X_mean_min = event_data[event_data['plane'] == min_plane]['X_transformed'].mean()
        Y_mean_min = event_data[event_data['plane'] == min_plane]['Y_transformed'].mean()
        Z_mean_min = event_data[event_data['plane'] == min_plane]['Z_transformed'].mean()
        
        # get average x, y, z from max plane (plane 0)
        X_mean_max = event_data[event_data['plane'] == max_plane]['X_transformed'].mean()
        Y_mean_max = event_data[event_data['plane'] == max_plane]['Y_transformed'].mean()
        Z_mean_max = event_data[event_data['plane'] == max_plane]['Z_transformed'].mean()
        
        # radius is 3* l2 norm of std (x,y,z) at max plane
        max_plane_data = event_data[event_data['plane'] == max_plane]
        if len(max_plane_data) > 0:
            stds = [
                max_plane_data['X_transformed'].std(),
                max_plane_data['Y_transformed'].std(),
                max_plane_data['Z_transformed'].std()
            ]
            radius = 3 * np.linalg.norm(stds)
        else:
            radius = 0
        
        return pd.Series({
            'event_id': event_data['event_id'].iloc[0],
            'min_plane': min_plane,
            'max_plane': max_plane,
            'X_mean_min': X_mean_min,
            'Y_mean_min': Y_mean_min,
            'Z_mean_min': Z_mean_min,
            'X_mean_max': X_mean_max,
            'Y_mean_max': Y_mean_max,
            'Z_mean_max': Z_mean_max,
            'radius': radius
        })
    except Exception as e:
        logger.error(f"Error calculating cone params: {e}")
        return pd.Series({'event_id': event_data['event_id'].iloc[0] if len(event_data)>0 else 'unknown'})

def process_one_parquet(parquet_file: str) -> pd.DataFrame:
    """Process a single parquet file and extract cone parameters."""
    try:
        logger.info(f"Processing {parquet_file}")
        
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        
        if df.empty:
            logger.warning(f"{parquet_file}: empty dataframe")
            return pd.DataFrame()
        
        # Group by event_id and calculate cone parameters
        event_cone_params = df.groupby('event_id').apply(calculate_cone_parameters).reset_index(drop=True)
        
        logger.info(f"{parquet_file}: processed {len(event_cone_params)} events")
        return event_cone_params
        
    except Exception as e:
        logger.error(f"Error processing {parquet_file}: {e}")
        return pd.DataFrame()

def worker(file_queue: mp.Queue, result_queue: mp.Queue):
    """Worker process for multiprocessing."""
    while True:
        parquet_file = file_queue.get()
        if parquet_file is None:
            break
        result = process_one_parquet(parquet_file)
        result_queue.put(result)

if __name__ == "__main__":
    # Find all parquet files
    parquet_files = list(Path(input_dir).glob("*.parquet"))
    parquet_files = [str(f) for f in parquet_files if f.name != "event_cone_parameters.parquet"]  # Exclude output file
    
    logger.info(f"Found {len(parquet_files)} parquet files to process")
    
    if not parquet_files:
        logger.warning("No parquet files found!")
        exit(1)
    
    # Setup multiprocessing
    n_cpus = len(os.sched_getaffinity(0))
    logger.info(f"Using {n_cpus} CPUs")
    
    file_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = [mp.Process(target=worker, args=(file_queue, result_queue)) 
                 for _ in range(min(n_cpus, len(parquet_files)))]
    
    for p in processes:
        p.start()
    
    # Enqueue all files
    for parquet_file in parquet_files:
        file_queue.put(parquet_file)
    
    # Send stop signals
    for _ in processes:
        file_queue.put(None)
    
    # Collect results with progress bar
    all_cone_params = []
    with tqdm(total=len(parquet_files), desc="Processing files") as pbar:
        for _ in range(len(parquet_files)):
            result_df = result_queue.get()
            if not result_df.empty:
                all_cone_params.append(result_df)
            pbar.update(1)
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    # Combine all results and save
    if all_cone_params:
        final_df = pd.concat(all_cone_params, ignore_index=True)
        final_df.drop_duplicates(subset=['event_id'], keep='first', inplace=True)
        final_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(final_df)} unique cone parameters to {output_file}")
    else:
        logger.error("No cone parameters computed!")
    
    logger.info("✅ Processing complete")

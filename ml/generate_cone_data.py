import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import logging
import gc
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler("process_cone_params.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Input and output directories
input_dir = "ml/processed_events_50k"
output_file = "ml/processed_events_50k/event_cone_parameters_normalized.parquet"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def calculate_cone_parameters(event_data):
    """Calculate cone parameters for a single event (takes group as input)."""
    try:
        event_id = event_data.name  # group key from groupby
        
        # Get primary energy (take first value, assuming it's constant per event)
        primary_energy = event_data['primary_kinetic_energy'].iloc[0] if 'primary_kinetic_energy' in event_data.columns else 0
        sin_azimuth = event_data['sin_azimuth'].iloc[0]
        cos_azimuth = event_data['cos_azimuth'].iloc[0]
        sin_zenith = event_data['sin_zenith'].iloc[0]
        cos_zenith = event_data['cos_zenith'].iloc[0]
        
        # Filter for planes > 0 to find min_plane
        valid_planes = event_data[event_data['plane'] > 0]
        if valid_planes.empty:
            return pd.Series({
                'primary_kinetic_energy': primary_energy,
                'min_plane': 0, 'max_plane': 0,
                'X_mean_min': 0, 'Y_mean_min': 0, 'Z_mean_min': 0,
                'X_mean_max': 0, 'Y_mean_max': 0, 'Z_mean_max': 0,
                'radius': 0, 
                'sin_azimuth': sin_azimuth, 'cos_azimuth': cos_azimuth,
                'sin_zenith': sin_zenith, 'cos_zenith': cos_zenith
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
            'primary_kinetic_energy': primary_energy,
            'min_plane': min_plane,
            'max_plane': max_plane,
            'X_mean_min': X_mean_min,
            'Y_mean_min': Y_mean_min,
            'Z_mean_min': Z_mean_min,
            'X_mean_max': X_mean_max,
            'Y_mean_max': Y_mean_max,
            'Z_mean_max': Z_mean_max,
            'radius': radius,
            'sin_azimuth':sin_azimuth,
            'cos_azimuth':cos_azimuth,
            'sin_zenith':sin_zenith,
            'cos_zenith':cos_zenith
        })
    except Exception as e:
        logger.error(f"Error calculating cone params for {event_data.name if hasattr(event_data, 'name') else 'unknown'}: {e}")
        return pd.Series({'primary_kinetic_energy': 0})

def process_one_parquet(parquet_file: str) -> pd.DataFrame:
    """Process a single parquet file and extract RAW cone parameters (no normalization)."""
    try:
        logger.info(f"Processing {parquet_file}")
        
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        
        if df.empty:
            logger.warning(f"{parquet_file}: empty dataframe")
            return pd.DataFrame()
        
        # EXACTLY like your example: groupby().apply().reset_index()
        batch_identifier = "event_id"
        event_cone_params = df.groupby(by=[batch_identifier]).apply(calculate_cone_parameters).reset_index()
        
        # Add event_id back after groupby.apply (no duplication)
        event_cone_params['event_id'] = event_cone_params[batch_identifier]
        
        if event_cone_params.empty:
            logger.warning(f"{parquet_file}: no cone parameters computed")
            return pd.DataFrame()
        
        logger.info(f"{parquet_file}: processed {len(event_cone_params)} raw events")
        return event_cone_params
        
    except Exception as e:
        logger.error(f"Error processing {parquet_file}: {e}")
        return pd.DataFrame()

def normalize_cone_features(final_df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization and standardization to ALL cone parameters at once (final step)."""
    try:
        logger.info("Applying global normalization to all events...")
        
        # Define feature groups for cone parameters
        energy_features = ["primary_kinetic_energy"]
        spatial_features = ["X_mean_min", "Y_mean_min", "Z_mean_min", 
                          "X_mean_max", "Y_mean_max", "Z_mean_max", "radius"]
        trig_features = ["sin_azimuth", "cos_azimuth", "sin_zenith", "cos_zenith"]

        
        # Outlier detection (z-score < 2) on energy + spatial features ACROSS ALL EVENTS
        numeric_cols = energy_features + spatial_features
        initial_count = len(final_df)
        
        if len(final_df) > 1:  # Need at least 2 points for z-score
            z_scores = np.abs(stats.zscore(final_df[numeric_cols]))
            outlier_mask = (z_scores < 2).all(axis=1)
            final_df = final_df[outlier_mask].reset_index(drop=True)
            logger.info(f"Outlier removal: {initial_count} -> {len(final_df)} events")
        
        # Define transformations (GLOBAL fit across all events)
        log_scaler = Pipeline([
            ("log", FunctionTransformer(
                func=np.log1p,
                inverse_func=np.expm1,
                validate=False,
                feature_names_out="one-to-one"
            )),
            ("scaler", StandardScaler())
        ])
        
        spatial_scaler = StandardScaler()
        
        # Keep non-scaled columns as passthrough
        metadata_cols = ["event_id", "min_plane", "max_plane"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("energy", log_scaler, energy_features),
                ("spatial", spatial_scaler, spatial_features),
                ("metadata", "passthrough", metadata_cols),
                ("trig", "passthrough", trig_features)

            ]
        )
        
        # Apply transformations (fit on ALL data)
        feature_cols = energy_features + spatial_features + metadata_cols + trig_features
        normalized_data = preprocessor.fit_transform(final_df[feature_cols])
        
        # Reconstruct DataFrame with proper column names
        normalized_df = pd.DataFrame(
            normalized_data, 
            columns=feature_cols
        )
        
        logger.info("Global normalization applied successfully")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error in global normalization: {e}")
        return final_df  # Return raw data if normalization fails

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
    parquet_files = [str(f) for f in parquet_files 
                    if f.name not in ["event_cone_parameters.parquet", 
                                    "event_cone_parameters_normalized.parquet"]]  # Exclude output files
    
    # parquet_files = parquet_files[:200]
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
    
    # Collect RAW results with progress bar
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
    
    # Combine all RAW results and deduplicate
    if all_cone_params:
        final_raw_df = pd.concat(all_cone_params, ignore_index=True)
        final_raw_df.drop_duplicates(subset=['event_id'], keep='first', inplace=True)
        final_raw_df.to_parquet('ml/processed_events_50k/tmp_10k_2.parquet', index=False)
        
        
        # # GLOBAL NORMALIZATION (fit on all events together)
        # final_normalized_df = normalize_cone_features(final_raw_df)
        # final_normalized_df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved {len(final_raw_df)} unique normalized cone parameters to {output_file}")
        logger.info(f"Final columns: {final_raw_df.columns.tolist()}")
        logger.info(f"Shape: {final_raw_df.shape}")
    else:
        logger.error("No cone parameters computed!")
    
    logger.info("✅ Processing complete")

import dask.dataframe as dd

df = dd.read_parquet("../ml/processed_events_50k/*.parquet")

# Write to a single file
df.compute().to_parquet("../ml/processed_events_50k/all_showers.parquet", 
                        index=False, 
                        engine="pyarrow")

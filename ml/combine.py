import dask.dataframe as dd

df = dd.read_parquet("../ml/processed_events/*.parquet")

# Write to a single file
df.compute().to_parquet("../ml/processed_events/all_showers.parquet", 
                        index=False, 
                        engine="pyarrow")

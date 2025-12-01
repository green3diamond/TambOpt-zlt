#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# Load data
df = pd.read_parquet("processed_events/all_showers.parquet")

# Drop missing values
df = df.dropna()

# Add 'charge' column: 1 if pdg is 11 or 13, 0 if pdg is 22, else 0
df["charge"] = np.where(df["pdg"].isin([11, 13]), 1, 0)

# Standardize (z-score normalization)
df["r_normalized"] = zscore(np.log1p(df["r"].values))
df["kinetic_energy_normalized"] = zscore(np.log1p(df["kinetic_energy"].values))
df["z_normalized"] = zscore(np.log1p(abs(df["Z_transformed"].values)))
df["y_normalized"] = zscore(np.log1p(abs(df["Y_transformed"].values)))
df["x_normalized"] = zscore(np.log1p(abs(df["X_transformed"].values)))
df["time_normalized"] = zscore(df["time_transformed"].values)

# Standardize the new 'charge' column as well
df["charge_normalized"] = zscore(df["charge"].values)
df["distance_normalized"] = zscore(df["distance"].values)


# Save processed file
df.to_parquet("processed_events/normalized.parquet", index=False)

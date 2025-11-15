import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from scipy import stats

# --- Step 1: Load parquet file ---
df = pd.read_parquet("../ml/processed_events/all_showers.parquet")

# --- Step 2: Define feature groups ---
energy_features = ["kinetic_energy", "primary_kinetic_energy"]
trig_features = ["sin_azimuth", "cos_azimuth", "sin_zenith", "cos_zenith"]
spatial_features = ["X_transformed", "Y_transformed", "Z_transformed", "distance"]
time_features = ["time_transformed"]

# Select only numeric columns for outlier detection
numeric_cols = energy_features + spatial_features + time_features
z_scores = np.abs(stats.zscore(df[numeric_cols]))
# remove data beyond 2 standard deviations
df = df[(z_scores < 2).all(axis=1)].reset_index(drop=True)

# --- Step 3: Define transformations ---
log_scaler = Pipeline([
    ("log", FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        validate=False,
        feature_names_out="one-to-one"  # keeps feature names consistent
    )),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("energy", log_scaler, energy_features),
        ("spatial", "passthrough", spatial_features),
        ("time", MinMaxScaler(), time_features),
        ("trig", "passthrough", trig_features)  # keep as is
    ]
)

# --- Step 4: Apply transformations ---
normalized_data = preprocessor.fit_transform(df)

# --- Step 5: Put back into DataFrame ---
all_columns = energy_features + spatial_features + time_features + trig_features
normalized_df = pd.DataFrame(normalized_data, columns=all_columns)
normalized_df["event_id"]=df["event_id"]
normalized_df["pdg"]=abs(df["pdg"])
normalized_df["plane"]=df["plane"]

# --- Step 6: Save result ---
normalized_df.to_parquet("../ml/processed_events/normalized_features.parquet", index=False)


print("Columns:", normalized_df.columns.tolist())

print(df.shape)

print(normalized_df.head())

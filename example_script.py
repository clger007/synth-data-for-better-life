# If needed: pip install ctgan pandas numpy
# !pip install ctgan pandas numpy

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ctgan import CTGAN

# ----------------------------
# 1) Build a toy training set
# ----------------------------
rng = np.random.default_rng(7)

n = 1_000
base_date = pd.Timestamp("2022-01-01")

df = pd.DataFrame({
    "amount": rng.normal(loc=120.0, scale=35.0, size=n).round(2),
    "category": rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1]),
    "user_segment": rng.choice(["free", "pro"], size=n, p=[0.7, 0.3]),
    "event_date": [base_date + pd.to_timedelta(int(x), unit="D") for x in rng.integers(0, 365, size=n)],
})

# Add a little missingness to make it realistic
mask = rng.random(n) < 0.05
df.loc[mask, "amount"] = np.nan  # CTGAN can handle missing numerics; feel free to impute if you prefer

print(df.head())

# ---------------------------------------------------
# 2) Convert datetimes to numeric (unix timestamp ms)
#    (Pure `ctgan` treats datetimes as numeric.)
# ---------------------------------------------------
def to_ts_ms(s: pd.Series) -> pd.Series:
    return (s.view("int64") // 10**6).astype("float")  # ms

def from_ts_ms(s: pd.Series) -> pd.Series:
    return pd.to_datetime((s.round().astype("int64")) * 10**6)

df_model = df.copy()
df_model["event_date"] = to_ts_ms(df_model["event_date"])

# ---------------------------------------------
# 3) Identify discrete (categorical) columns
# ---------------------------------------------
# CTGAN needs to know which columns are discrete.
discrete_columns = ["category", "user_segment"]

# ---------------------------------------------
# 4) Train CTGAN
#    (Adjust hyperparams as your data grows.)
# ---------------------------------------------
ctgan = CTGAN(
    epochs=300,            # start small; increase for quality
    batch_size=512,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    pac=10,                # good default for mode collapse on tabular data
    cuda=False             # set True if you have a CUDA GPU available
)

ctgan.fit(df_model, discrete_columns=discrete_columns)

# ---------------------------------------------
# 5) Sample synthetic rows
# ---------------------------------------------
num_samples = 5_000
synthetic = ctgan.sample(num_samples)

# Convert timestamps back to datetimes
synthetic["event_date"] = from_ts_ms(synthetic["event_date"])

# Optional: clip/repair simple numeric quirks (e.g., negative amount)
synthetic["amount"] = synthetic["amount"].clip(lower=0)

print(synthetic.head())

# ---------------------------------------------
# 6) Quick sanity checks (optional)
# ---------------------------------------------
print("\nReal vs Synthetic category distribution:")
print(pd.concat([
    df["category"].value_counts(normalize=True).rename("real"),
    synthetic["category"].value_counts(normalize=True).rename("synthetic")
], axis=1))

print("\nReal vs Synthetic amount summary:")
print(pd.DataFrame({
    "real": df["amount"].describe(percentiles=[.05, .5, .95]),
    "synthetic": synthetic["amount"].describe(percentiles=[.05, .5, .95]),
}))

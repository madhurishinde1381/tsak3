# tsak 3
# Cell 1: Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
%matplotlib inline
# Cell 2: Load dataset (try some common paths; fallback guidance)
possible_paths = [
    "housing.csv",
    "Housing.csv",
    "house_prices.csv",
    "/mnt/data/housing.csv",        # common in CODEX / notebook sandboxes
    "/content/housing.csv"
]

df = None
for p in possible_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Loaded dataset from: {p}")
        break

if df is None:
    print("No local CSV found in common locations.")
    print("If you have the Kaggle dataset, either:")
    print("  1) Upload `housing.csv` to the notebook directory, or")
    print("  2) Configure Kaggle API and use the kaggle CLI to download (not included here).")
    # create a small synthetic dataset fallback so cells can run for demo:
    np.random.seed(0)
    df = pd.DataFrame({
        "sqft_living": np.random.randint(500, 4000, 300),
        "bedrooms": np.random.randint(1,6,300),
        "bathrooms": np.round(np.random.uniform(1,4,300),1),
        "floors": np.random.choice([1,2,3],300),
        "waterfront": np.random.choice([0,1],300, p=[0.95,0.05]),
        "grade": np.random.randint(1,13,300),
        "zipcode": np.random.choice(['98001','98002','98003','98004'],300),
        "price": lambda x: None
    })
    # make price correlated with sqft_living, grade, bathrooms
    df["price"] = (df["sqft_living"] * 200) + (df["grade"] * 10000) + (df["bathrooms"] * 5000) \
                  + (df["waterfront"] * 100000) + np.random.normal(0, 20000, df.shape[0])
    print("Using synthetic demo dataset (for quick run). Replace with real CSV for actual results.")

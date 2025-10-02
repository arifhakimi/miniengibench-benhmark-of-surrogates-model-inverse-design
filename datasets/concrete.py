from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# UCI Concrete Compressive Strength dataset
# Original is an Excel file: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
# We load it directly via pandas; requires openpyxl.
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

COLS = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
    "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age", "Strength"
]

def load_concrete(test_size=0.2, val_size=0.1, random_state=42, local_path: str | None = None):
    """
    Loads the Concrete Compressive Strength dataset and returns standardized splits.
    Target: Strength (MPa)
    """
    if local_path and Path(local_path).exists():
        df = pd.read_excel(local_path, header=0)
    else:
        df = pd.read_excel(URL, header=0)

    df.columns = COLS  # ensure consistent naming
    X = df.drop(columns=["Strength"]).to_numpy(dtype=float)
    y = df["Strength"].to_numpy(dtype=float)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    rel = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

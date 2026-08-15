"""
Feature pipeline for the Purchase-Intent-GCP baseline.

Design decisions:
- Month is KEPT (v0 dropped it) and cyclically encoded - purchase intent is
  seasonal (Nov/Dec spike is a known pattern in this dataset), so dropping
  it throws away real signal for no reason.
- Numeric features are scaled - not needed for trees, but needed once we
  benchmark against a neural net in Phase 2, so building it in now saves
  a rewrite later.
- Categorical columns are one-hot encoded with unknown categories ignored
  at inference time rather than raising - this matters once this is served
  as an API and gets a browser/traffic type the training data never saw.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BOOLEAN_FEATURES = ["Weekend"]
MONTH_COL = "Month"

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def add_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month_num = df[MONTH_COL].map(MONTH_MAP)
    df["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
    df["Month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return df.drop(columns=[MONTH_COL])


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = NUMERIC_FEATURES + ["Month_sin", "Month_cos"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ]
    )

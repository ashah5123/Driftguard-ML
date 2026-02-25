"""Data loading and preprocessor construction for the training pipeline."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from disk. Supports .csv and .parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {suffix}. Use .csv or .parquet.")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer: median + StandardScaler for numeric, constant + OneHotEncoder for categorical."""
    numeric_transformer = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    categorical_transformer = [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            ("num", Pipeline(numeric_transformer), numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("cat", Pipeline(categorical_transformer), categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

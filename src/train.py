"""Training script: load data, cross-validate, fit pipeline, save model and metadata to disk and MLflow."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_preprocessing import build_preprocessor, load_data


def _detect_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical by dtype."""
    numeric = []
    categorical = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost classifier with preprocessing.")
    parser.add_argument("--input", required=True, help="Path to input data (.csv or .parquet)")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--mlflow_experiment", default="default", help="MLflow experiment name")
    args = parser.parse_args()

    target = args.target
    df = load_data(args.input)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in data. Columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c not in (target, "flight_date", "year")]
    if not feature_cols:
        raise ValueError("No feature columns left after excluding target and non-features.")
    X = df[feature_cols]
    y = df[target]

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    numeric_features, categorical_features = _detect_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier()),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    cv_auc_mean = float(cv_scores.mean())
    cv_auc_std = float(cv_scores.std())

    # MLflow (local)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_metric("cv_auc_mean", cv_auc_mean)
        mlflow.log_metric("cv_auc_std", cv_auc_std)

        # Fit on full data
        pipeline.fit(X, y)

        # Save to disk
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.pkl"
        joblib.dump(pipeline, model_path)

        metadata = {
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "feature_cols": feature_cols,
            "n_samples": int(len(X)),
            "n_numeric": len(numeric_features),
            "n_categorical": len(categorical_features),
        }
        metadata_path = models_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")

    # Print key outputs
    print("Training complete.")
    print(f"  CV ROC-AUC (mean ± std): {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    print(f"  Model saved: {model_path}")
    print(f"  Metadata saved: {metadata_path}")
    print(f"  MLflow experiment: {args.mlflow_experiment}")


if __name__ == "__main__":
    main()

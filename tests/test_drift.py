"""Test drift detection: detect_drift returns expected keys and numeric PSI results."""

import pandas as pd
import pytest

from src.drift import detect_drift


def test_detect_drift_returns_keys_and_numeric_psi():
    reference = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    current = pd.DataFrame({
        "a": [1.1, 2.1, 3.1, 4.1, 5.1],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    numeric_cols = ["a", "b"]
    result = detect_drift(reference, current, numeric_cols)

    assert set(result.keys()) == set(numeric_cols)
    for col in numeric_cols:
        assert col in result
        row = result[col]
        assert "psi" in row
        assert "ks_stat" in row
        assert "ks_pvalue" in row
        assert isinstance(row["psi"], (int, float))
        assert isinstance(row["ks_stat"], (int, float))
        assert isinstance(row["ks_pvalue"], (int, float))

"""Drift detection: PSI and KS test, robust to constant arrays and NaNs."""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# Minimum proportion per bin to avoid log(0); keep PSI finite
_EPS = 1e-10


def psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    buckets: int = 10,
) -> float:
    """
    Population Stability Index between expected and actual distributions.
    Robust to constant arrays and NaNs (NaNs dropped; constant arrays yield 0 or finite PSI).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Constant expected: single bin at that value
    if np.ptp(expected) == 0:
        c = float(np.min(expected))
        expected_pct = np.array([1.0])
        in_bin = np.sum(np.abs(actual - c) <= _EPS) if len(actual) > 0 else 0
        p = in_bin / len(actual)
        actual_pct = np.array([max(_EPS, min(p, 1.0))])
    else:
        _, bin_edges = np.histogram(expected, bins=buckets)
        if np.ptp(bin_edges) == 0:
            expected_pct = np.array([1.0])
            actual_pct = np.array([1.0])
        else:
            expected_counts, _ = np.histogram(expected, bins=bin_edges)
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            expected_pct = expected_counts / max(expected_counts.sum(), 1)
            actual_pct = actual_counts / max(actual_counts.sum(), 1)
            expected_pct = np.clip(expected_pct, _EPS, 1.0)
            actual_pct = np.clip(actual_pct, _EPS, 1.0)

    # PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    ratio = np.clip(actual_pct / expected_pct, _EPS, 1.0 / _EPS)
    psi_val = np.sum((actual_pct - expected_pct) * np.log(ratio))
    return float(psi_val)


def ks_test(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """
    Two-sample Kolmogorov–Smirnov test. Returns (statistic, p-value).
    NaNs are dropped from both samples.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan"), float("nan")

    stat, pvalue = ks_2samp(expected, actual)
    return float(stat), float(pvalue)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Compute drift metrics per numeric feature.
    Returns dict[feature_name, {"psi": float, "ks_stat": float, "ks_pvalue": float}].
    """
    results: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col]
        cur = current_df[col]
        psi_val = psi(ref, cur)
        ks_stat, ks_pvalue = ks_test(ref, cur)
        results[col] = {
            "psi": psi_val,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
        }
    return results

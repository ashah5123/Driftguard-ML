"""Data quality gate: validate data/processed/current.csv. Exit 0 pass, 1 fail. Uses Great Expectations minimally."""

import sys
from pathlib import Path

import pandas as pd

from data_preprocessing import load_data

REFERENCE_PATH = Path("data/processed/reference.csv")
CURRENT_PATH = Path("data/processed/current.csv")
DEFAULT_REQUIRED_COLUMNS = ["feature1", "feature2", "target"]


def _get_required_columns() -> list[str]:
    """Required columns from reference.csv if it exists, else default list."""
    if REFERENCE_PATH.exists():
        df = load_data(REFERENCE_PATH)
        return list(df.columns)
    return list(DEFAULT_REQUIRED_COLUMNS)


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


def _check_target_binary(df: pd.DataFrame, target: str) -> tuple[bool, str]:
    if target not in df.columns:
        return False, f"Target column '{target}' not in data"
    vals = df[target].dropna()
    if len(vals) == 0:
        return False, "Target column is all null"
    try:
        uniq = set(vals.astype(int).unique())
    except (ValueError, TypeError):
        return False, "Target has non-numeric values"
    if not uniq.issubset({0, 1}):
        return False, f"Target must be binary (0/1); found values: {uniq}"
    return True, ""


def _check_no_all_null_columns(df: pd.DataFrame) -> tuple[bool, str]:
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        return False, f"Completely null columns: {all_null}"
    return True, ""


def _check_numeric_min_distinct(df: pd.DataFrame, min_distinct: int = 2) -> tuple[bool, str]:
    bad = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() < min_distinct:
            bad.append(c)
    if bad:
        return False, f"Numeric columns with < {min_distinct} distinct values: {bad}"
    return True, ""


def _run_great_expectations(df: pd.DataFrame, target: str) -> tuple[bool, str]:
    """Run minimal GE expectations (target binary) using programmatic expectations."""
    try:
        import great_expectations as gx
    except ImportError:
        return True, ""  # skip if GE not installed
    try:
        # Ephemeral context and pandas datasource (GE 1.x)
        context = gx.get_context(mode="ephemeral")
        datasource = context.sources.add_pandas("pandas")
        batch_request = datasource.read_dataframe(df)
        suite_name = "data_quality_gate"
        context.add_expectation_suite(expectation_suite_name=suite_name)
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )
        validator.expect_column_to_exist(target)
        validator.expect_column_values_to_be_in_set(column=target, value_set=[0, 1])
        validator.save_expectation_suite()
        result = validator.validate()
        if not result.success:
            parts = [
                getattr(r, "expectation_config", {}) and r.expectation_config.get("expectation_type", "")
                for r in result.results
                if not getattr(r, "success", True)
            ]
            return False, f"Great Expectations failed: {parts}"
    except Exception as e:
        return False, f"Great Expectations error: {e}"
    return True, ""


def main() -> int:
    print("DriftGuard data quality gate")
    print("  Current data:", CURRENT_PATH)

    if not CURRENT_PATH.exists():
        print("[FAIL] Current file not found:", CURRENT_PATH)
        return 1

    try:
        df = load_data(CURRENT_PATH)
    except Exception as e:
        print("[FAIL] Failed to load data:", e)
        return 1

    required = _get_required_columns()
    target = "target" if "target" in required else (required[0] if required else "target")

    checks = [
        ("Required columns", lambda d: _check_required_columns(d, required)),
        ("Target binary (0/1)", lambda d: _check_target_binary(d, target)),
        ("No all-null columns", _check_no_all_null_columns),
        ("Numeric columns ≥2 distinct", _check_numeric_min_distinct),
    ]
    for name, run in checks:
        ok, err = run(df)
        if not ok:
            print(f"[FAIL] {name}: {err}")
            return 1

    ok, err = _run_great_expectations(df, target)
    if not ok:
        print(f"[FAIL] {err}")
        return 1

    print("[PASS] Data quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

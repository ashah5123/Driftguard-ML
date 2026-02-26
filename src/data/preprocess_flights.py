"""Preprocess flight delay CSV: load largest from data/raw, add year/target/features, split into reference and current."""

import argparse
from pathlib import Path

import pandas as pd

# Column name candidates (first match wins)
DATE_CANDIDATES = ["FL_DATE", "Flight_Date", "flight_date"]
YEAR_COL_CANDIDATES = ["YEAR", "year"]
MONTH_COL_CANDIDATES = ["MONTH", "month"]
DAY_COL_CANDIDATES = ["DAY", "day"]
ARR_DELAY_CANDIDATES = ["ARR_DELAY", "ArrDelay", "arr_delay", "arrival_delay"]
AIRLINE_CANDIDATES = ["OP_CARRIER", "AIRLINE", "Carrier", "carrier", "op_carrier"]
ORIGIN_CANDIDATES = ["ORIGIN", "Origin", "origin"]
DEST_CANDIDATES = ["DEST", "Dest", "dest", "destination", "Destination"]
DISTANCE_CANDIDATES = ["DISTANCE", "Distance", "distance"]
DEP_TIME_CANDIDATES = ["DEP_TIME", "CRS_DEP_TIME", "DepTime", "dep_time", "crs_dep_time"]


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _largest_csv(raw_dir: Path) -> Path:
    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")
    return max(csvs, key=lambda p: p.stat().st_size)


def _extract_year(df: pd.DataFrame, date_col: str | None, year_col: str | None, month_col: str | None, day_col: str | None) -> pd.Series:
    if year_col and year_col in df.columns:
        return df[year_col].astype(int)
    if date_col and date_col in df.columns:
        ser = pd.to_datetime(df[date_col], errors="coerce")
        return ser.dt.year
    if month_col and day_col and month_col in df.columns and day_col in df.columns:
        # synthetic year from MONTH/DAY: use a reference year (e.g. 2020) when only month/day exist
        return pd.Series(2020, index=df.index)
    raise ValueError("Could not determine year: no FL_DATE or YEAR/MONTH/DAY columns found.")


def _extract_dep_hour(df: pd.DataFrame, dep_time_col: str) -> pd.Series:
    # BTS style: DEP_TIME as HHMM (e.g. 830 = 8:30) or minutes since midnight
    raw = pd.to_numeric(df[dep_time_col], errors="coerce")
    # If values > 2400 or very large, treat as minutes; else HHMM
    if raw.max() > 2400 or raw.min() > 2400:
        return (raw // 60).astype("Int64")
    return (raw // 100).astype("Int64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess flight delay CSV and split into reference/current.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw CSV(s)")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for reference.csv and current.csv")
    parser.add_argument("--reference-year", type=int, default=2021, help="Year for reference set")
    parser.add_argument("--current-year", type=int, default=2022, help="Year for current set")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to read (default: all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    path = _largest_csv(raw_dir)
    print(f"Loading largest CSV: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(path, nrows=args.max_rows)
    print(f"Loaded shape: {df.shape}")
    cols = list(df.columns)

    date_col = _pick(cols, DATE_CANDIDATES)
    year_col = _pick(cols, YEAR_COL_CANDIDATES)
    month_col = _pick(cols, MONTH_COL_CANDIDATES)
    day_col = _pick(cols, DAY_COL_CANDIDATES)
    arr_delay_col = _pick(cols, ARR_DELAY_CANDIDATES)
    airline_col = _pick(cols, AIRLINE_CANDIDATES)
    origin_col = _pick(cols, ORIGIN_CANDIDATES)
    dest_col = _pick(cols, DEST_CANDIDATES)
    distance_col = _pick(cols, DISTANCE_CANDIDATES)
    dep_time_col = _pick(cols, DEP_TIME_CANDIDATES)

    df["year"] = _extract_year(df, date_col, year_col, month_col, day_col)
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    print(f"Detected year range: {year_min}–{year_max}")

    if not arr_delay_col:
        raise ValueError("No arrival delay column found (ARR_DELAY, ArrDelay, etc.).")
    df["Delayed"] = (df[arr_delay_col] > 15).astype(int)

    keep = {"year", "Delayed"}
    renames = {}
    if airline_col:
        keep.add(airline_col)
        renames[airline_col] = "airline"
    if origin_col:
        keep.add(origin_col)
        renames[origin_col] = "origin"
    if dest_col:
        keep.add(dest_col)
        renames[dest_col] = "destination"
    if distance_col:
        keep.add(distance_col)
        renames[distance_col] = "distance"
    if dep_time_col:
        keep.add(dep_time_col)
        df["dep_hour"] = _extract_dep_hour(df, dep_time_col)
        keep.add("dep_hour")
        keep.discard(dep_time_col)

    existing = [c for c in keep if c in df.columns]
    out = df[existing].copy()
    out.rename(columns=renames, inplace=True)
    out = out.dropna(how="all", axis=1)

    ref = out[out["year"] == args.reference_year].drop(columns=["year"])
    cur = out[out["year"] == args.current_year].drop(columns=["year"])
    ref_path = out_dir / "reference.csv"
    cur_path = out_dir / "current.csv"
    ref.to_csv(ref_path, index=False)
    cur.to_csv(cur_path, index=False)
    print(f"Reference ({args.reference_year}): {ref.shape} -> {ref_path}")
    print(f"Current   ({args.current_year}): {cur.shape} -> {cur_path}")


if __name__ == "__main__":
    main()

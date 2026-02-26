"""Download flights_sample_3m.csv from Kaggle dataset via CLI."""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

DATASET = "patrickzel/flight-delay-and-cancellation-dataset-2019-2023"
FILE_NAME = "flights_sample_3m.csv"
DEFAULT_OUTPUT_DIR = "data/raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download flights_sample_3m.csv from Kaggle into data/raw.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-f", FILE_NAME,
        "-p", str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    # Kaggle downloads a zip; extract the CSV into the output dir if present
    zips = list(out_dir.glob("*.zip"))
    if zips:
        target = out_dir / FILE_NAME
        with zipfile.ZipFile(zips[0], "r") as zf:
            for name in zf.namelist():
                if name.endswith(FILE_NAME) or name == FILE_NAME:
                    zf.extract(name, out_dir)
                    extracted = out_dir / name
                    if extracted.resolve() != target.resolve():
                        shutil.move(str(extracted), str(target))
                    break
        zips[0].unlink(missing_ok=True)


if __name__ == "__main__":
    main()

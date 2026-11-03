"""Time-based train / validation / test split.

    train: arrivals up to 2016-12-31   (model fitting)
    val:   2017-01-01 .. 2017-03-31    (model and threshold selection)
    test:  2017-04-01 onward           ("production" data, held out)

The test split is written to its own directory and nothing before the production
validation stage reads it. `load_design_data` returns train + val only, so code that
uses it cannot touch test rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MONTHS = {
    name: i
    for i, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
VAL_START = pd.Timestamp("2017-01-01")
TEST_START = pd.Timestamp("2017-04-01")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "hotels.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"  # train.csv, val.csv
HOLDOUT_DIR = REPO_ROOT / "data" / "holdout"  # test.csv, read only at production validation


def add_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an `arrival_date` column built from year, month name and day."""
    out = df.copy()
    out["arrival_date"] = pd.to_datetime(
        {
            "year": out["arrival_date_year"],
            "month": out["arrival_date_month"].map(MONTHS),
            "day": out["arrival_date_day_of_month"],
        }
    )
    return out


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Label every row train / val / test from its `arrival_date`."""
    labels = pd.Series("train", index=df.index, dtype="object")
    labels[df["arrival_date"] >= VAL_START] = "val"
    labels[df["arrival_date"] >= TEST_START] = "test"
    return labels


def load_design_data(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load train + val (never test) with a `split` column."""
    frames = []
    for name in ("train", "val"):
        part = pd.read_csv(Path(processed_dir) / f"{name}.csv", parse_dates=["arrival_date"])
        part["split"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

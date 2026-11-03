"""Split the raw CSV by arrival date and record a manifest with file hashes.

Writes
    data/processed/train.csv, data/processed/val.csv
    data/holdout/test.csv                       (not read by training or EDA code)
    docs/split_manifest.json                    (rows, date range, SHA-256 per file; committed)

The manifest holds no label statistics for the test split.

Usage:
    python -m src.data.make_splits              # write splits + manifest
    python -m src.data.make_splits --check      # recompute and compare with the manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.splits import (
    HOLDOUT_DIR,
    PROCESSED_DIR,
    RAW_PATH,
    REPO_ROOT,
    add_arrival_date,
    assign_split,
)
from src.ingestion.download import sha256_of

MANIFEST_PATH = REPO_ROOT / "docs" / "split_manifest.json"


def _write(df: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "rows": len(df),
        "first_arrival": df["arrival_date"].min().strftime("%Y-%m-%d"),
        "last_arrival": df["arrival_date"].max().strftime("%Y-%m-%d"),
        "sha256": sha256_of(path),
    }


def build(raw_path: Path = RAW_PATH) -> dict:
    """Write the three split files and return the manifest dict."""
    raw = pd.read_csv(raw_path)
    df = add_arrival_date(raw)
    df["_split"] = assign_split(df)
    assert df["_split"].notna().all() and len(df) == len(raw)
    parts = {name: g.drop(columns="_split") for name, g in df.groupby("_split")}
    return {
        "raw": {"path": str(raw_path.relative_to(REPO_ROOT)), "rows": len(raw),
                "sha256": sha256_of(raw_path)},
        "train": _write(parts["train"], PROCESSED_DIR / "train.csv"),
        "val": _write(parts["val"], PROCESSED_DIR / "val.csv"),
        "test": _write(parts["test"], HOLDOUT_DIR / "test.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with committed manifest")
    args = parser.parse_args()
    manifest = build()
    if args.check:
        committed = json.loads(MANIFEST_PATH.read_text())
        if committed != manifest:
            print("MISMATCH between recomputed splits and docs/split_manifest.json")
            return 1
        print("OK splits match docs/split_manifest.json")
        return 0
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("train", "val", "test"):
        m = manifest[name]
        print(f"{name:5s} rows={m['rows']:>6} {m['first_arrival']}..{m['last_arrival']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

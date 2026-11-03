"""Download the hotel bookings CSV and verify it byte-for-byte.

Source: TidyTuesday 2020-02-11 (Thomas Mock, Antoine Bichat), a cleaned, combined
version of the Hotel Booking Demand datasets of Antonio, de Almeida and Nunes (2019),
Data in Brief 22, 41-49, https://doi.org/10.1016/j.dib.2018.11.126.

The file is pinned by SHA-256, so a silent upstream change fails loudly instead of
changing results.

Usage:
    python -m src.ingestion.download            # -> data/raw/hotels.csv
    python -m src.ingestion.download --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)
EXPECTED_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "raw" / "hotels.csv"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise ValueError if the file's SHA-256 differs from `expected`."""
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}. "
            "The upstream file changed or the download is corrupt."
        )


def download(
    dest: Path = DEFAULT_PATH,
    url: str = URL,
    expected_sha256: str = EXPECTED_SHA256,
    force: bool = False,
) -> Path:
    """Fetch `url` to `dest` (unless a verified copy exists) and return the path."""
    dest = Path(dest)
    if dest.exists() and not force:
        verify_sha256(dest, expected_sha256)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        verify_sha256(tmp_path, expected_sha256)  # only a verified file becomes dest
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    path = download(args.dest, force=args.force)
    print(f"OK {path} sha256={sha256_of(path)}")


if __name__ == "__main__":
    main()

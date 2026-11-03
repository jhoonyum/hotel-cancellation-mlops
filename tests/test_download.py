import hashlib

import pytest

from src.ingestion.download import EXPECTED_SHA256, download, sha256_of, verify_sha256


def test_pinned_hash_is_a_valid_sha256():
    assert len(EXPECTED_SHA256) == 64
    int(EXPECTED_SHA256, 16)  # raises if not hex


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hotel" * 1000)
    assert sha256_of(f) == hashlib.sha256(b"hotel" * 1000).hexdigest()


def test_verify_accepts_match_and_rejects_mismatch(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n")
    verify_sha256(f, hashlib.sha256(b"a,b\n1,2\n").hexdigest())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(f, "0" * 64)


def test_download_from_file_url_verifies_and_leaves_no_partial_file(tmp_path):
    src = tmp_path / "src.csv"
    src.write_bytes(b"a,b\n1,2\n")
    good = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    dest = tmp_path / "out" / "hotels.csv"

    with pytest.raises(ValueError):
        download(dest, url=src.as_uri(), expected_sha256="0" * 64)
    assert not dest.exists()
    assert list(dest.parent.glob("*.part")) == []

    assert download(dest, url=src.as_uri(), expected_sha256=good) == dest
    assert dest.read_bytes() == b"a,b\n1,2\n"

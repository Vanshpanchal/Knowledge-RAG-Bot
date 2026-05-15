"""Security-related helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Compute SHA256 hex digest for given bytes payload."""
    hasher = hashlib.sha256()
    hasher.update(payload)
    return hasher.hexdigest()

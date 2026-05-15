"""Validation helpers for uploads."""

from __future__ import annotations

from pathlib import Path


def sniff_signature(file_head: bytes) -> str:
    if file_head.startswith(b"%PDF"):
        return "application/pdf"
    if file_head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if file_head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if file_head.startswith(b"PK"):
        return "application/zip-based"
    if file_head.startswith(b"RIFF") and b"WAVE" in file_head[:16]:
        return "audio/wav"
    if file_head.startswith(b"ID3"):
        return "audio/mpeg"
    if len(file_head) >= 2 and file_head[0] == 0xFF and (file_head[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    if file_head.startswith(b"fLaC"):
        return "audio/flac"
    if file_head.startswith(b"OggS"):
        return "audio/ogg"
    if len(file_head) >= 8 and file_head[4:8] == b"ftyp":
        return "audio/mp4"
    return "unknown"


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower().strip()

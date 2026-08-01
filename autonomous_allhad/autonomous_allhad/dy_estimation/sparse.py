"""Small deterministic helpers for sparse NanoAOD reads."""

from __future__ import annotations

import hashlib
from typing import Iterable


def stable_file_id(path: str) -> int:
    """Return the 31-bit file ID used by the current feature worker."""

    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def group_sparse_windows(
    entries: Iterable[int],
    *,
    max_span: int = 50_000,
    max_gap: int = 5_000,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Group sparse entry indices into bounded half-open read windows."""

    if max_span <= 0 or max_gap < 0:
        raise ValueError("max_span must be positive and max_gap nonnegative")
    ordered = sorted({int(value) for value in entries})
    if any(value < 0 for value in ordered):
        raise ValueError("entry indices must be nonnegative")
    if not ordered:
        return []
    output: list[tuple[int, int, tuple[int, ...]]] = []
    start = ordered[0]
    previous = ordered[0]
    members = [ordered[0]]
    for entry in ordered[1:]:
        if entry - previous > max_gap or entry - start + 1 > max_span:
            output.append((start, previous + 1, tuple(members)))
            start = entry
            members = [entry]
        else:
            members.append(entry)
        previous = entry
    output.append((start, previous + 1, tuple(members)))
    return output

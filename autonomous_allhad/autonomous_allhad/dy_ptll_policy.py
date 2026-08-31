#!/usr/bin/env python3
"""Pure policy helpers for selecting mutually exclusive DY sample families."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import re
from typing import Any


DY_PTLL_EXCLUSIVE_WINDOWS_GEV: dict[str, tuple[float, float | None]] = {
    "100": (100.0, 200.0),
    "200": (200.0, 400.0),
    "400": (400.0, 600.0),
    "600": (600.0, None),
}


def dy_ptll_exclusive_window(
    dataset: str,
    process: str,
) -> tuple[float, float | None] | None:
    """Return the exclusive generator-level pT(ll) window for a DY sample.

    The 2024 ``DYto2L`` PTLL samples are inclusive above the threshold encoded
    in their names.  They must therefore be made exclusive at event level
    using ``LHE_Vpt``: 100--200, 200--400, 400--600, and >=600 GeV.
    """
    if process != "DY":
        return None
    match = re.search(r"PTLL-(100|200|400|600)_", dataset)
    if match is None:
        return None
    return DY_PTLL_EXCLUSIVE_WINDOWS_GEV[match.group(1)]


def dy_ptll_in_exclusive_window(
    dataset: str,
    process: str,
    generator_ptll: float,
) -> bool:
    """Return whether an event belongs to its sample's exclusive PTLL slice."""
    window = dy_ptll_exclusive_window(dataset, process)
    if window is None:
        return True
    lower, upper = window
    value = float(generator_ptll)
    return value >= lower and (upper is None or value < upper)


def dy_ptll_dataset_allowed(dataset: str, process: str, policy: str) -> bool:
    """Return whether a dataset belongs in the requested DY sample policy.

    The ``ptll100_200`` policy is the adopted composition for the 2026-07-27
    downstream rerun: PTLL-400 and PTLL-600 are explicitly excluded.
    """
    if policy in {"all", "exclusive_gen_ptll"} or process != "DY":
        return True
    if policy == "ptll200_only":
        return "PTLL-200_" in dataset
    if policy == "ptll100_only":
        return "PTLL-100_" in dataset
    if policy == "ptll100_200":
        return "PTLL-100_" in dataset or "PTLL-200_" in dataset
    if policy == "dy2x":
        # Adopt the Run-3 flavour-exclusive DY production and reject every
        # legacy pT(ll)-binned DY artifact.  The token boundary avoids
        # accidentally accepting a similarly named private sample.
        return re.search(r"DYto2(?:E|Mu|Tau)-4Jets(?:_|$)", dataset) is not None
    raise ValueError(f"unknown DY pT(ll) dataset policy: {policy}")


def true_ranges(mask: Sequence[Any]) -> list[tuple[int, int]]:
    """Return half-open contiguous ranges containing truthy mask values."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def dataset_id_prefilter_plan(
    dataset_ids: Iterable[int],
    allowed_ids: set[int],
) -> dict[str, Any]:
    """Build the allowed read ranges and excluded-ID counts for one chunk."""
    ids = [int(dataset_id) for dataset_id in dataset_ids]
    keep = [dataset_id in allowed_ids for dataset_id in ids]
    excluded = Counter(
        dataset_id
        for dataset_id, allowed in zip(ids, keep)
        if not allowed
    )
    return {
        "ranges": true_ranges(keep),
        "excluded_counts": dict(sorted(excluded.items())),
        "entries_scanned": len(ids),
        "entries_loaded": sum(stop - start for start, stop in true_ranges(keep)),
    }

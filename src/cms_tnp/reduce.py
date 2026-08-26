"""Deterministic reduction of data and simulation histogram shards."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

import numpy as np

IDENTITY = (
    "measurement",
    "year",
    "probe_collection",
    "probe_selection",
    "pass_selection",
    "probe_abseta_edges",
    "probe_pt_edges_gev",
    "mass_edges_gev",
    "fit",
    "correction",
)


def merge(shards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(shards)
    if not items:
        raise ValueError("no shards were supplied")
    first = items[0]
    for item in items[1:]:
        mismatches = [key for key in IDENTITY if item.get(key) != first.get(key)]
        if mismatches:
            raise ValueError(f"incompatible shards: {mismatches}")
    merged: dict[str, dict[str, np.ndarray]] = {}
    for item in items:
        for sample, histograms in item["samples"].items():
            if sample not in merged:
                merged[sample] = {
                    key: np.asarray(value, dtype=float)
                    for key, value in histograms.items()
                }
            else:
                for key, value in histograms.items():
                    merged[sample][key] += np.asarray(value, dtype=float)
    if "data" not in merged or "mc" not in merged:
        raise ValueError("reduction requires both nominal data and MC histograms")
    processing = {
        "files_expected": sum(
            int(item["processing"]["files_expected"]) for item in items
        ),
        "files_processed": sum(
            int(item["processing"]["files_processed"]) for item in items
        ),
        "files_failed": [
            failure for item in items for failure in item["processing"]["files_failed"]
        ],
        "events_read": sum(int(item["processing"]["events_read"]) for item in items),
        "pairs_selected": sum(
            int(item["processing"]["pairs_selected"]) for item in items
        ),
    }
    output = {key: copy.deepcopy(first[key]) for key in ("schema_version", *IDENTITY)}
    output["samples"] = {
        sample: {key: value.tolist() for key, value in values.items()}
        for sample, values in merged.items()
    }
    output["processing"] = processing
    blockers = [
        blocker for item in items for blocker in item.get("adoption_blockers", [])
    ]
    if processing["files_expected"] != processing["files_processed"]:
        blockers.append(
            f"ROOT coverage is {processing['files_processed']}/{processing['files_expected']}"
        )
    output["adoption_blockers"] = list(dict.fromkeys(blockers))
    output["status"] = (
        "complete"
        if processing["files_expected"] == processing["files_processed"]
        else "incomplete"
    )
    return output

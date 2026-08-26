#!/usr/bin/env python3
"""Build and validate analysis-owned correctionlib scale-factor payloads.

The measurement writes fit/count results first. Only results whose status is
explicitly ``adopted`` are exported without the CLI's candidate flag.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import correctionlib


ADOPTED_STATUS = "adopted"
VARIATIONS = ("nominal", "up", "down")


def ensure_adopted(result: dict[str, Any], source: Path) -> None:
    status = str(result.get("status") or "")
    if status != ADOPTED_STATUS:
        raise RuntimeError(
            f"refusing to install {source}: status is {status!r}, expected {ADOPTED_STATUS!r}"
        )


def _finite_list(values: Iterable[float], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _axis(name: str, edges: Iterable[float]) -> dict[str, Any]:
    parsed = _finite_list(edges, label=f"{name} edges")
    if len(parsed) < 2 or any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} edges must be strictly increasing")
    return {"name": str(name), "edges": parsed}


def _multibinning(axes: list[dict[str, Any]], content: list[float]) -> dict[str, Any]:
    expected = 1
    for axis in axes:
        expected *= len(axis["edges"]) - 1
    if len(content) != expected:
        raise ValueError(f"multibinning content has {len(content)} values; expected {expected}")
    return {
        "nodetype": "multibinning",
        "inputs": [axis["name"] for axis in axes],
        "edges": [axis["edges"] for axis in axes],
        "content": _finite_list(content, label="correction content"),
        "flow": "clamp",
    }


def correction(
    *,
    name: str,
    description: str,
    axes: Iterable[tuple[str, Iterable[float]]],
    nominal: Iterable[float],
    uncertainty: Iterable[float],
    version: int = 1,
) -> dict[str, Any]:
    """Return a variation-category correction with symmetric total uncertainty.

    The flattened content follows correctionlib multibinning order: the last
    axis varies fastest.  All analysis-owned corrections use ``flow=clamp``;
    callers must still mask the intended physics domain before evaluation.
    """

    parsed_axes = [_axis(axis_name, edges) for axis_name, edges in axes]
    nominal_values = _finite_list(nominal, label=f"{name} nominal")
    uncertainty_values = _finite_list(uncertainty, label=f"{name} uncertainty")
    if len(nominal_values) != len(uncertainty_values):
        raise ValueError("nominal and uncertainty arrays must have equal length")
    if any(value <= 0.0 for value in nominal_values):
        raise ValueError(f"{name} nominal scale factors must be positive")
    if any(value < 0.0 for value in uncertainty_values):
        raise ValueError(f"{name} uncertainties must be non-negative")
    values = {
        "nominal": nominal_values,
        "up": [value + error for value, error in zip(nominal_values, uncertainty_values)],
        "down": [max(1.0e-6, value - error) for value, error in zip(nominal_values, uncertainty_values)],
    }
    return {
        "name": name,
        "description": description,
        "version": int(version),
        "inputs": [
            {"name": "variation", "type": "string", "description": "nominal, up, or down"},
            *[
                {"name": axis["name"], "type": "real", "description": axis["name"]}
                for axis in parsed_axes
            ],
        ],
        "output": {"name": "weight", "type": "real", "description": "data/MC scale factor"},
        "data": {
            "nodetype": "category",
            "input": "variation",
            "content": [
                {"key": variation, "value": _multibinning(parsed_axes, values[variation])}
                for variation in VARIATIONS
            ],
        },
    }


def correction_set(description: str, corrections: Iterable[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "description": description,
        "corrections": list(corrections),
        "compound_corrections": [],
    }
    if not payload["corrections"]:
        raise ValueError("at least one correction is required")
    return payload


def write_json_gz(path: Path, payload: dict[str, Any]) -> str:
    """Validate with correctionlib, write deterministically, and return SHA256."""

    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    # Validate the exact JSON before writing it.
    correctionlib.CorrectionSet.from_string(encoded.decode())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(encoded)
    # Re-open the compressed artifact so installation failures are caught here.
    correctionlib.CorrectionSet.from_file(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_adopted_result(
    result_path: Path,
    output_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    ensure_adopted(result, result_path)
    digest = write_json_gz(output_path, payload)
    return {
        "status": "installed",
        "source_result": str(result_path),
        "output": str(output_path),
        "sha256": digest,
        "corrections": [item["name"] for item in payload["corrections"]],
    }

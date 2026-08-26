"""Correctionlib export."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def build_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    blockers = list(result.get("adoption_blockers", []))
    if blockers:
        raise ValueError(f"cannot export a blocked result: {blockers}")
    invalid = [
        int(item["flat_index"]) for item in result["bins"] if not item.get("valid")
    ]
    if invalid:
        raise ValueError(f"cannot export invalid bins: {invalid}")
    nominal = [float(item["scale_factor"]) for item in result["bins"]]
    uncertainty = [float(item["scale_factor_uncertainty"]) for item in result["bins"]]
    values = {
        "nominal": nominal,
        "up": [value + error for value, error in zip(nominal, uncertainty)],
        "down": [
            max(1.0e-6, value - error) for value, error in zip(nominal, uncertainty)
        ],
    }
    correction = result["correction"]
    axes = [result["probe_abseta_edges"], result["probe_pt_edges_gev"]]
    return {
        "schema_version": 2,
        "description": str(correction["description"]),
        "corrections": [
            {
                "name": str(correction["name"]),
                "description": str(correction["description"]),
                "version": 1,
                "inputs": [
                    {"name": "variation", "type": "string"},
                    {"name": "abseta", "type": "real"},
                    {"name": "pt", "type": "real"},
                ],
                "output": {"name": "weight", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "variation",
                    "content": [
                        {
                            "key": variation,
                            "value": {
                                "nodetype": "multibinning",
                                "inputs": ["abseta", "pt"],
                                "edges": axes,
                                "content": content,
                                "flow": str(correction.get("flow", "clamp")),
                            },
                        }
                        for variation, content in values.items()
                    ],
                },
            }
        ],
        "compound_corrections": [],
    }


def write_payload(path: Path | str, payload: Mapping[str, Any]) -> str:
    import correctionlib

    path = Path(path)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    correctionlib.CorrectionSet.from_string(encoded.decode())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as compressed:
        compressed.write(encoded)
    correctionlib.CorrectionSet.from_file(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()

#!/usr/bin/env python3
"""Build the unified RZ(Nb) measurement from merged histogram inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import finalize_rz


CHANNELS = ("DY2E", "DY2M")
GROUPS = ("Nb1", "Nb2plus")
WINDOWS = ("on", "off")
COMPONENTS = ("data", "zll", "other")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_leaf(node: dict[str, Any], context: str) -> dict[str, Any]:
    leaf = (node or {}).get("nominal") or {}
    output: dict[str, Any] = {}
    for quantity in ("sumw", "sumw2", "entries"):
        values = leaf.get(quantity)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"{context}/{quantity}: expected one histogram bin")
        output[quantity] = (
            int(values[0]) if quantity == "entries" else float(values[0])
        )
    return output


def histogram_leaf(
    node: dict[str, Any], edges: list[float], context: str
) -> dict[str, Any]:
    leaf = (node or {}).get("nominal") or {}
    nbin = len(edges) - 1
    output: dict[str, Any] = {"edges": list(edges)}
    for quantity in ("sumw", "sumw2", "entries"):
        values = leaf.get(quantity)
        if not isinstance(values, list) or len(values) != nbin:
            raise ValueError(
                f"{context}/{quantity}: expected {nbin} histogram bins"
            )
        output[quantity] = list(values)
    return output


def convert_regime(
    source: dict[str, Any], regime: str, edges: list[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw: dict[str, Any] = {}
    mll: dict[str, Any] = {}
    node = ((source.get("dy_rz") or {}).get(regime) or {})
    yields = node.get("yields") or {}
    histograms = node.get("mll") or {}
    for channel in CHANNELS:
        for group in GROUPS:
            for window in WINDOWS:
                for component in COMPONENTS:
                    raw.setdefault(channel, {}).setdefault(group, {}).setdefault(
                        window, {}
                    )[component] = scalar_leaf(
                        (((yields.get(channel) or {}).get(group) or {}).get(window) or {}).get(component) or {},
                        f"{regime}/yields/{channel}/{group}/{window}/{component}",
                    )
            for component in COMPONENTS:
                mll.setdefault(channel, {}).setdefault(group, {})[component] = (
                    histogram_leaf(
                        (((histograms.get(channel) or {}).get(group) or {}).get(component) or {}),
                        edges,
                        f"{regime}/mll/{channel}/{group}/{component}",
                    )
                )
    return raw, mll


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-year", choices=("2024", "2025"), default="2024")
    args = parser.parse_args(argv)

    source = json.loads(args.hist_input.read_text())
    if source.get("status") != "complete":
        raise ValueError(f"histogram input is not complete: {source.get('status')}")
    if source.get("schema_version") != "background_estimation_histograms_v1":
        raise ValueError(
            "unsupported histogram boundary: "
            f"{source.get('schema_version')!r}"
        )
    policy = source.get("category_policy") or {}
    for regime in ("highdm", "lowdm"):
        if list(policy.get(regime) or []) != list(GROUPS):
            raise ValueError(f"{regime}: RZ requires Nb1/Nb2plus only")

    edges = [float(value) for value in ((source.get("dy_rz") or {}).get("mll_edges") or [])]
    if len(edges) < 2:
        raise ValueError("DY RZ mll edges are absent")
    high_raw, high_mll = convert_regime(source, "highdm", edges)
    low_raw, low_mll = convert_regime(source, "lowdm", edges)
    output = {
        "schema_version": f"dy_estimation_measurement_{args.campaign_year}_v3",
        "status": "complete",
        "rz_high_raw": high_raw,
        "rz_high": finalize_rz(high_raw),
        "mll_high": high_mll,
        "rz_low_raw": low_raw,
        "rz_low": finalize_rz(low_raw),
        "mll_low": low_mll,
        "summary": {
            "source_kind": "merged normalized histograms",
            "expected_partitions": 0,
            "completed_partitions": 0,
            "candidate_events": 0,
            "matched_events": 0,
            "failures": [],
        },
        "provenance": {
            "hist_input": str(args.hist_input),
            "hist_input_sha256": sha256_file(args.hist_input),
            "campaign_year": args.campaign_year,
            "intermediate_root_reread": False,
            "mass_windows": (source.get("dy_rz") or {}).get("mass_windows"),
            "category_policy": {
                "highdm": list(GROUPS),
                "lowdm": list(GROUPS),
                "removed_lowdm_axes": ["Njet", "pTb", "ISR"],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "combined": {
                    "highdm": output["rz_high"]["combined"],
                    "lowdm": output["rz_low"]["combined"],
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

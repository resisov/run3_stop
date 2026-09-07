#!/usr/bin/env python3
"""Build compact transfer-factor inputs from the merged histogram boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SAMPLES = ("ST", "TT", "WtoLNu", "QCD")
REGIONS = ("SR", "LLCR", "QCDCR")
GROUPS = ("Nb1", "Nb2plus")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_leaf(value: Any, context: str, nbin: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: histogram leaf is absent")
    output: dict[str, Any] = {}
    for quantity in ("sumw", "sumw2", "entries"):
        values = value.get(quantity)
        if not isinstance(values, list) or len(values) != nbin:
            raise ValueError(
                f"{context}/{quantity}: expected {nbin} bins, found "
                f"{len(values) if isinstance(values, list) else 'none'}"
            )
        output[quantity] = list(values)
    return output


def select_regime(source: dict[str, Any], regime: str) -> dict[str, Any]:
    node = source.get(regime) or {}
    edges = [float(value) for value in node.get("recoil_edges") or []]
    if len(edges) < 2:
        raise ValueError(f"{regime}: recoil edges are absent")
    if list(node.get("nb_groups") or []) != list(GROUPS):
        raise ValueError(
            f"{regime}: only the Nb1/Nb2plus category policy is supported"
        )
    nbin = len(edges) - 1
    recoil = node.get("recoil") or {}
    selected: dict[str, Any] = {}
    for region in REGIONS:
        selected[region] = {}
        for group in GROUPS:
            selected[region][group] = {}
            by_sample = ((recoil.get(region) or {}).get(group) or {})
            for sample in SAMPLES:
                variations = by_sample.get(sample) or {}
                if "nominal" not in variations:
                    raise ValueError(
                        f"{regime}/{region}/{group}/{sample}: nominal histogram absent"
                    )
                selected[region][group][sample] = {
                    variation: require_leaf(
                        leaf,
                        f"{regime}/{region}/{group}/{sample}/{variation}",
                        nbin,
                    )
                    for variation, leaf in variations.items()
                }
    return {
        "recoil_edges": edges,
        "nb_groups": list(GROUPS),
        "recoil": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hist-input",
        type=Path,
        required=True,
        help="Compact *_background_estimation.json from histogram merging.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-year", choices=("2024", "2025"), default="2024")
    args = parser.parse_args()

    source = json.loads(args.hist_input.read_text())
    if source.get("status") != "complete":
        raise ValueError(f"histogram input is not complete: {source.get('status')}")
    if source.get("schema_version") != "background_estimation_histograms_v1":
        raise ValueError(
            "unsupported histogram boundary: "
            f"{source.get('schema_version')!r}"
        )

    output = {
        "schema_version": f"histogram_tf_inputs_{args.campaign_year}_v3",
        "status": "complete",
        "highdm": select_regime(source, "highdm"),
        "lowdm": select_regime(source, "lowdm"),
        "summary": {
            "selected_processes": list(SAMPLES),
            "source_kind": "merged normalized histograms",
        },
        "provenance": {
            "hist_input": str(args.hist_input),
            "hist_input_sha256": file_sha256(args.hist_input),
            "campaign_year": args.campaign_year,
            "intermediate_root_reread": False,
            "include_data": False,
            "regions": list(REGIONS),
            "category_policy": {
                "highdm": "Nb=1 and Nb>=2 in native U_T bins",
                "lowdm": "Nb=1 and Nb>=2 in native U_T bins",
                "removed_lowdm_axes": ["Njet", "pTb", "ISR"],
            },
            "sample_policy": {
                "top": "TT + ST",
                "qcd": "QCD-4Jets HT-binned current merged process",
                "dy": "DYto2E/Mu/Tau-4Jets; PTLL excluded upstream",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

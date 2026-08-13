#!/usr/bin/env python3
"""Audit normalized photon-origin composition in photon-fake sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from measure_photon_fakes_2024 import (
    _distribution_record,
    _merge_dataset_record,
    _sum_stratified,
    aggregate_component,
    read_payload,
)


PROBES = ("target", "application", "measurement_pass", "measurement_fail")
ORIGINS = ("all", "prompt", "electron", "fake")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--nominal", type=Path)
    parser.add_argument("--processes", nargs="+", default=["QCD", "GJ"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        paths.extend(sorted(path.rglob("*.json.gz")) if path.is_dir() else [path])
    datasets: dict[str, Any] = {}
    for path in sorted(set(paths)):
        payload = read_payload(path)
        for physical, incoming in (payload.get("datasets") or {}).items():
            if physical not in datasets:
                datasets[physical] = incoming
            else:
                _merge_dataset_record(datasets[physical], incoming)

    normalization = read_payload(args.normalization)
    process_results: dict[str, Any] = {}
    for process in args.processes:
        origin_channels: dict[str, Any] = {}
        origin_audits: dict[str, Any] = {}
        for origin in ORIGINS:
            channels, audit = aggregate_component(
                datasets,
                normalization,
                origin,
                {process},
            )
            origin_channels[origin] = channels
            origin_audits[origin] = audit
        yields: dict[str, Any] = {}
        for probe in PROBES:
            probe_yields: dict[str, float] = {}
            probe_variances: dict[str, float] = {}
            for origin in ORIGINS:
                record = _distribution_record(
                    origin_channels[origin],
                    probe,
                    origin,
                    "GCR",
                    "recoil",
                )
                value, variance = _sum_stratified(record)
                probe_yields[origin] = value
                probe_variances[origin] = variance
            partition = (
                probe_yields["prompt"]
                + probe_yields["electron"]
                + probe_yields["fake"]
            )
            yields[probe] = {
                "sumw": probe_yields,
                "sumw2": probe_variances,
                "classified_sum": partition,
                "all_minus_classified": probe_yields["all"] - partition,
                "fractions_of_all": {
                    origin: (
                        probe_yields[origin] / probe_yields["all"]
                        if probe_yields["all"] != 0.0
                        else None
                    )
                    for origin in ("prompt", "electron", "fake")
                },
            }
        process_results[process] = {
            "yields": yields,
            "normalization_audits": origin_audits,
        }

    nominal_comparison = None
    if args.nominal is not None:
        nominal = read_payload(args.nominal)
        variable = nominal["highdm_variable_histograms"]["GCR"]["ut"]
        nominal_comparison = {}
        for process in args.processes:
            leaf = variable.get(process, {}).get("nominal")
            if leaf is None:
                continue
            values = np.asarray(leaf["sumw"], dtype=float)
            nominal_comparison[process] = {
                "ut_bins": values.tolist(),
                "integral": float(np.sum(values)),
                "sidecar_target_all": process_results[process]["yields"]["target"][
                    "sumw"
                ]["all"],
                "sidecar_target_fake": process_results[process]["yields"]["target"][
                    "sumw"
                ]["fake"],
            }

    payload = {
        "schema_version": "photon_fake_2024_origin_audit_v1",
        "status": "complete",
        "inputs": [str(path) for path in sorted(set(paths))],
        "normalization": str(args.normalization),
        "processes": process_results,
        "nominal_comparison": nominal_comparison,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                process: {
                    probe: record["sumw"]
                    for probe, record in result["yields"].items()
                }
                for process, result in process_results.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

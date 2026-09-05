#!/usr/bin/env python3
"""Merge all-signal partials and extend the frozen 30-bin Low-dM inputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import ROOT

try:
    from .build_datacard import write_cards
except ImportError:
    from build_datacard import write_cards  # type: ignore[no-redef]


TOPOLOGIES = ("T2tt", "T2bW", "T2tb")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def empty_counter(bins: int) -> dict[str, np.ndarray]:
    return {
        "sumw": np.zeros(bins, dtype=float),
        "sumw2": np.zeros(bins, dtype=float),
        "entries": np.zeros(bins, dtype=np.int64),
    }


def merge_partials(paths: list[Path], expected_partials: int):
    if len(paths) != expected_partials:
        raise RuntimeError(
            f"expected {expected_partials} partials, found {len(paths)}"
        )
    merged: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    configuration = None
    input_events = test_events = 0
    exclusions = []
    sources = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") not in {"complete", "complete_with_exclusions"}:
            raise RuntimeError(f"incomplete partial: {path}")
        if configuration is None:
            configuration = payload["configuration"]
        elif payload["configuration"] != configuration:
            raise RuntimeError("partial category/edge configuration mismatch")
        input_events += int(payload["input_events_selected"])
        test_events += int(payload["test_events"])
        exclusions.extend(payload.get("exclusions", []))
        sources.append({
            "path": str(path),
            "input_root": payload["input_root"],
            "test_events": int(payload["test_events"]),
            "mass_points": int(payload["mass_points"]),
        })
        for sample, by_category in payload["histograms"].items():
            target_sample = merged.setdefault(sample, {})
            for label, counter in by_category.items():
                target = target_sample.setdefault(label, empty_counter(len(counter["sumw"])))
                for key in ("sumw", "sumw2", "entries"):
                    values = np.asarray(counter[key], dtype=target[key].dtype)
                    if values.shape != target[key].shape:
                        raise RuntimeError(f"bin mismatch for {sample}/{label}/{key}")
                    target[key] += values
    assert configuration is not None
    return merged, configuration, input_events, test_events, exclusions, sources


def parse_sample(sample: str):
    topology, stop, lsp = sample.split("_")
    return topology, int(stop.removeprefix("mStop")), int(lsp.removeprefix("mLSP"))


def root_signals(merged, topology: str, labels: list[str]):
    signals = {}
    for sample, by_category in merged.items():
        point_topology, mstop, mlsp = parse_sample(sample)
        if point_topology != topology:
            continue
        mass_key = f"mStop{mstop}_mLSP{mlsp}"
        signals[mass_key] = {
            label: (
                np.asarray(by_category[label]["sumw"], dtype=float),
                np.asarray(by_category[label]["sumw2"], dtype=float),
            )
            for label in labels
        }
    return signals


def update_template(path: Path, signals, labels: list[str]) -> None:
    root_file = ROOT.TFile(str(path), "UPDATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"cannot update {path}")
    try:
        for label in labels:
            directory = root_file.GetDirectory(f"SR_{label}")
            if not directory:
                raise RuntimeError(f"missing SR_{label} in {path}")
            for mass_key, by_category in signals.items():
                values, variances = by_category[label]
                edges = array("d", np.arange(len(values) + 1, dtype=float))
                name = "signal_" + mass_key
                histogram = ROOT.TH1D(name, name, len(values), edges)
                histogram.SetDirectory(directory)
                for index, (value, variance) in enumerate(zip(values, variances), start=1):
                    histogram.SetBinContent(index, float(max(value, 0.0)))
                    histogram.SetBinError(index, math.sqrt(max(float(variance), 0.0)))
                directory.cd()
                histogram.Write(name, ROOT.TObject.kOverwrite)
    finally:
        root_file.Close()


def compare_existing_signals(old_root: Path, signals, labels: list[str]):
    root_file = ROOT.TFile.Open(str(old_root), "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"cannot read {old_root}")
    compared = 0
    max_abs = max_rel = 0.0
    try:
        for mass_key, by_category in signals.items():
            for label in labels:
                histogram = root_file.Get(f"SR_{label}/signal_{mass_key}")
                if not histogram:
                    continue
                old = np.asarray(
                    [histogram.GetBinContent(index) for index in range(1, histogram.GetNbinsX() + 1)],
                    dtype=float,
                )
                new = np.asarray(by_category[label][0], dtype=float)
                difference = np.abs(old - new)
                max_abs = max(max_abs, float(np.max(difference, initial=0.0)))
                denominator = np.maximum(np.abs(old), 1.0e-12)
                max_rel = max(max_rel, float(np.max(difference / denominator, initial=0.0)))
                compared += len(old)
    finally:
        root_file.Close()
    return {"bins_compared": compared, "maximum_absolute_difference": max_abs,
            "maximum_relative_difference": max_rel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partials", required=True, type=Path)
    parser.add_argument("--existing-low", required=True, type=Path)
    parser.add_argument("--high-grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eos-reference-base", required=True)
    parser.add_argument("--campaign-year", required=True, choices=("2024", "2025"))
    parser.add_argument("--expected-partials", type=int, default=11)
    args = parser.parse_args()
    if args.expected_partials < 1:
        parser.error("--expected-partials must be positive")
    paths = sorted(args.partials.glob("signal_cache_*.json"))
    merged, configuration, input_events, test_events, exclusions, sources = merge_partials(
        paths, args.expected_partials
    )
    labels = list(configuration["category_labels"])
    high_campaign = json.loads(args.high_grid.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    topology_summary = {}
    all_records = []
    for topology in TOPOLOGIES:
        signals = root_signals(merged, topology, labels)
        source_dir = args.existing_low / topology
        old_manifest = json.loads((source_dir / "manifest.json").read_text())
        old_template = source_dir / f"templates_{topology}.root"
        target = args.output / topology
        target.mkdir(parents=True, exist_ok=True)
        template = target / f"lowdm_{args.campaign_year}.root"
        shutil.copy2(old_template, template)
        parity = compare_existing_signals(old_template, signals, labels)
        update_template(template, signals, labels)
        summary = old_manifest["channel_summary"]
        reference = (
            args.eos_reference_base.rstrip("/") + f"/{topology}/{template.name}"
        )
        cards_dir = target / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        for stale_card in cards_dir.glob("datacard_mStop*_mLSP*.txt"):
            stale_card.unlink()
        cards = write_cards(cards_dir, reference, signals, labels, summary)
        high_points = set(high_campaign["topologies"][topology]["mass_points"])
        low_points = set(signals)
        missing_high = sorted(high_points - low_points)
        if missing_high:
            raise RuntimeError(f"{topology}: {len(missing_high)} High-dM points lack Low-dM signal")
        records = []
        for mass_key in sorted(signals):
            mstop, mlsp = (
                int(part.removeprefix(prefix))
                for part, prefix in zip(mass_key.split("_"), ("mStop", "mLSP"))
            )
            records.append({
                "mass_key": mass_key, "topology": topology, "mStop": mstop, "mLSP": mlsp,
                "deltaM": mstop - mlsp,
                "yield": float(sum(np.sum(signals[mass_key][label][0]) for label in labels)),
                "in_highdm_grid": mass_key in high_points,
            })
        all_records.extend(records)
        manifest = dict(old_manifest)
        manifest.update({
            "status": "combine_inputs_ready",
            "signal_domain": "all generated mass points passing diagonal-v3 selection",
            "mass_points": sorted(signals),
            "mass_point_count": len(signals),
            "high_grid_mass_point_count": len(high_points),
            "high_grid_missing_low_signal": missing_high,
            "template_root": str(template),
            "template_reference": reference,
            "cards": cards,
            "existing_diagonal_parity": parity,
        })
        write_json(target / "manifest.json", manifest)
        topology_summary[topology] = {
            "all_lowdm_points": len(signals),
            "high_grid_points": len(high_points),
            "existing_diagonal_parity": parity,
            "template": str(template),
        }
    campaign = {
        "schema_version": "gnn_lowdm_all_signal_combine_inputs_v1",
        "status": "combine_inputs_ready",
        "selection": (
            "feature_lowdm_preselection && Nt=0 && NW=0 && Nres=0 && Nb>=1; "
            "no deltaM, mStop, mLSP, MET/sqrtHT, NISR, ISR-dphi, or High-dM veto"
        ),
        "partition": "deterministic 70% test; signal normalized by 1/0.70",
        "input_selected_events": input_events,
        "test_signal_events": test_events,
        "partial_count": len(paths),
        "exclusions": exclusions,
        "configuration": configuration,
        "topologies": topology_summary,
        "signal_records": all_records,
        "sources": sources,
    }
    write_json(args.output / "campaign_manifest.json", campaign)
    write_json(args.output / "all_signal_histograms.json", {
        "status": "complete", "configuration": configuration,
        "histograms": {
            sample: {
                label: {key: values.tolist() for key, values in counter.items()}
                for label, counter in by_category.items()
            }
            for sample, by_category in merged.items()
        },
    })
    print(json.dumps({"status": campaign["status"], "topologies": topology_summary,
                      "test_signal_events": test_events, "exclusions": len(exclusions)},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

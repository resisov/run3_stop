#!/usr/bin/env python3
"""Merge low-pT lepton pass/fail histogram shard JSONs."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .reference_trigger_counts import json_safe


HISTOGRAM_KEYS = ("pass_sumw", "pass_sumw2", "fail_sumw", "fail_sumw2")
SAMPLES = ("data", "mc", "mc_pileup_up", "mc_pileup_down")


def use_mc_reference(
    payload: dict[str, Any],
    reference: dict[str, Any],
    *,
    source: Path,
    reference_year: str | None = None,
) -> dict[str, Any]:
    """Replace only the MC histograms with an explicitly audited reference.

    This keeps the target-year data measurement intact while allowing a
    high-statistics MC reference when an otherwise identical target-year
    production is unavailable.  The compatibility checks deliberately cover
    every object and fit definition that affects the pass/fail efficiency.
    """

    compatibility_keys = (
        "kind",
        "probe_definition",
        "denominator_selection",
        "target_selection",
        "tag_pt_min_gev",
        "tag_miniiso_max",
        "external_reference_muon",
        "tag_trigger_match_required",
        "reference_trigger_object_kind",
        "probe_abseta_edges",
        "probe_pt_edges_gev",
        "mass_edges_gev",
        "fit_window_gev",
    )
    mismatches = [
        key for key in compatibility_keys
        if payload.get(key) != reference.get(key)
    ]
    if mismatches:
        raise ValueError(
            "MC-reference histograms are incompatible with the target "
            f"measurement: {mismatches}"
        )
    resolved_reference_year = str(reference_year or reference.get("year") or "")
    if not resolved_reference_year:
        raise ValueError(
            "MC-reference year is absent; provide --mc-reference-year explicitly"
        )
    result = dict(payload)
    samples = dict(payload["samples"])
    for name in ("mc", "mc_pileup_up", "mc_pileup_down"):
        reference_sample = reference.get("samples", {}).get(name)
        if reference_sample is None:
            raise ValueError(f"MC-reference payload is missing samples/{name}")
        for key in HISTOGRAM_KEYS:
            target_shape = np.asarray(samples[name][key], dtype=float).shape
            reference_shape = np.asarray(reference_sample[key], dtype=float).shape
            if target_shape != reference_shape:
                raise ValueError(
                    f"MC-reference histogram shape mismatch for {name}/{key}: "
                    f"{reference_shape} != {target_shape}"
                )
        samples[name] = reference_sample
    result["samples"] = samples
    result["mc_physical_datasets"] = list(reference.get("mc_physical_datasets") or [])
    result["pileup_correction"] = reference.get("pileup_correction")
    result["mc_reference"] = {
        "source": str(source),
        "year": resolved_reference_year,
        "files_expected": reference.get("files_expected"),
        "files_processed": reference.get("files_processed"),
        "files_failed": reference.get("files_failed"),
        "mc_physical_datasets": list(reference.get("mc_physical_datasets") or []),
        "policy": (
            "Use the identically selected high-statistics reference MC because "
            "no equivalent target-year SPS double-J/psi NanoAOD production exists; "
            "the target-year data histograms are unchanged. Future POG validation "
            "of this cross-year simulation reference is required."
        ),
    }
    return result


def _add_histograms(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in HISTOGRAM_KEYS:
        left = np.asarray(target[key], dtype=float)
        right = np.asarray(source[key], dtype=float)
        if left.shape != right.shape:
            raise ValueError(f"TnP histogram shape mismatch for {key}: {left.shape} vs {right.shape}")
        target[key] = (left + right).tolist()


def rebin_probe_histograms(
    payload: dict[str, Any],
    *,
    target_eta_edges: list[float],
    target_pt_edges: list[float],
) -> dict[str, Any]:
    """Merge existing probe bins without revisiting NanoAOD events."""

    source_eta = np.asarray(payload["probe_abseta_edges"], dtype=float)
    source_pt = np.asarray(payload["probe_pt_edges_gev"], dtype=float)
    target_eta = np.asarray(target_eta_edges, dtype=float)
    target_pt = np.asarray(target_pt_edges, dtype=float)
    for label, source, target in (
        ("eta", source_eta, target_eta),
        ("pt", source_pt, target_pt),
    ):
        if len(target) < 2 or np.any(np.diff(target) <= 0.0):
            raise ValueError(f"target {label} edges must be strictly increasing")
        if not all(np.any(np.isclose(edge, source, rtol=0.0, atol=1.0e-9)) for edge in target):
            raise ValueError(f"target {label} edges must be a subset of source edges")
        if target[0] < source[0] - 1.0e-9 or target[-1] > source[-1] + 1.0e-9:
            raise ValueError(f"target {label} range must lie inside the source range")

    source_n_eta = len(source_eta) - 1
    source_n_pt = len(source_pt) - 1
    rebinned_samples: dict[str, Any] = {}
    for sample_name, sample in payload["samples"].items():
        rebinned_samples[sample_name] = {}
        for key in HISTOGRAM_KEYS:
            values = np.asarray(sample[key], dtype=float)
            if values.shape[0] != source_n_eta * source_n_pt:
                raise ValueError(f"{sample_name}/{key} probe-bin count does not match source edges")
            shaped = values.reshape(source_n_eta, source_n_pt, *values.shape[1:])
            merged = []
            for eta_low, eta_high in zip(target_eta[:-1], target_eta[1:]):
                eta_indices = np.flatnonzero(
                    (source_eta[:-1] >= eta_low - 1.0e-9)
                    & (source_eta[1:] <= eta_high + 1.0e-9)
                )
                for pt_low, pt_high in zip(target_pt[:-1], target_pt[1:]):
                    pt_indices = np.flatnonzero(
                        (source_pt[:-1] >= pt_low - 1.0e-9)
                        & (source_pt[1:] <= pt_high + 1.0e-9)
                    )
                    block = shaped[np.ix_(eta_indices, pt_indices)]
                    merged.append(np.sum(block, axis=(0, 1)))
            rebinned_samples[sample_name][key] = np.asarray(merged).tolist()
    result = dict(payload)
    result["samples"] = rebinned_samples
    result["probe_abseta_edges"] = target_eta.tolist()
    result["probe_pt_edges_gev"] = target_pt.tolist()
    result["probe_binning_reduction"] = {
        "source_probe_abseta_edges": source_eta.tolist(),
        "source_probe_pt_edges_gev": source_pt.tolist(),
        "target_probe_abseta_edges": target_eta.tolist(),
        "target_probe_pt_edges_gev": target_pt.tolist(),
        "method": "exact selection and sum of existing pass/fail mass histograms",
    }
    return result


def merge_tnp_shards(
    paths: list[Path],
    *,
    kind: str,
    config: dict[str, Any],
    expected_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_measurement = config.get("source_measurement", config["measurement"])
    first = None
    samples = None
    seen_files: set[str] = set()
    successful_files: set[str] = set()
    mc_datasets: set[str] = set()
    unresolved_failures: dict[str, dict[str, Any]] = {}
    shard_audit = []
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        if payload.get("measurement") != expected_measurement or payload.get("kind") != kind:
            raise ValueError(f"TnP measurement mismatch in {path}")
        if config.get("year") is not None and str(payload.get("year")) != str(config["year"]):
            raise ValueError(
                f"TnP campaign-year mismatch in {path}: "
                f"{payload.get('year')!r} != {config.get('year')!r}"
            )
        if payload.get("probe_definition") != config.get("probe_definition"):
            raise ValueError(
                f"TnP probe-definition mismatch in {path}: "
                f"{payload.get('probe_definition')!r} != {config.get('probe_definition')!r}"
            )
        expected_trigger_match = bool(config.get("tag_trigger_match_required", True))
        if bool(payload.get("tag_trigger_match_required", True)) != expected_trigger_match:
            raise ValueError(
                f"TnP tag-trigger topology mismatch in {path}: "
                f"{payload.get('tag_trigger_match_required')!r} != {expected_trigger_match!r}"
            )
        expected_reference_kind = config.get("reference_trigger_object_kind", kind)
        if payload.get("reference_trigger_object_kind", kind) != expected_reference_kind:
            raise ValueError(
                f"TnP reference-object mismatch in {path}: "
                f"{payload.get('reference_trigger_object_kind')!r} != {expected_reference_kind!r}"
            )
        expected_reference_application = {
            "data": True,
            "mc": bool(config.get("apply_reference_trigger_to_mc", True)),
        }
        payload_reference_application = payload.get(
            "reference_trigger_application",
            {"data": True, "mc": True},
        )
        if payload_reference_application != expected_reference_application:
            raise ValueError(
                f"TnP reference-trigger application mismatch in {path}: "
                f"{payload_reference_application!r} != "
                f"{expected_reference_application!r}"
            )
        expected_tag_pt = config.get("tag_pt_min_gev")
        if expected_tag_pt is not None and not np.isclose(
            float(payload.get("tag_pt_min_gev", math.nan)),
            float(expected_tag_pt),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"TnP tag-threshold mismatch in {path}: "
                f"{payload.get('tag_pt_min_gev')!r} != {config.get('tag_pt_min_gev')!r}"
            )
        if payload.get("tag_miniiso_max", 0.1) != config.get("tag_miniiso_max", 0.1):
            raise ValueError(
                f"TnP tag-isolation mismatch in {path}: "
                f"{payload.get('tag_miniiso_max')!r} != "
                f"{config.get('tag_miniiso_max')!r}"
            )
        if payload.get("external_reference_muon") != config.get("external_reference_muon"):
            raise ValueError(
                f"TnP external-reference topology mismatch in {path}: "
                f"{payload.get('external_reference_muon')!r} != "
                f"{config.get('external_reference_muon')!r}"
            )
        if first is None:
            first = payload
            samples = {name: payload["samples"][name] for name in SAMPLES}
        else:
            for name in SAMPLES:
                _add_histograms(samples[name], payload["samples"][name])
        records_by_path = {
            str(record.get("file_path") or ""): record
            for record in payload.get("input_records", [])
            if str(record.get("file_path") or "")
        }
        for file_path, record in records_by_path.items():
            seen_files.add(file_path)
            if record.get("sample") == "mc" and record.get("dataset"):
                mc_datasets.add(str(record["dataset"]).split("____", 1)[0])
        shard_successful: set[str] = set()
        shard_failures: list[dict[str, Any]] = []
        for sample_stats in payload.get("processing", {}).values():
            shard_successful.update(str(path) for path in sample_stats.get("files_successful") or [])
            shard_failures.extend(sample_stats.get("files_failed") or [])
        duplicate_success = successful_files & shard_successful
        if duplicate_success:
            raise RuntimeError(
                f"duplicate successful ROOT file across TnP shards: {sorted(duplicate_success)[:5]}"
            )
        successful_files.update(shard_successful)
        for file_path in shard_successful:
            unresolved_failures.pop(file_path, None)
        for failure in shard_failures:
            file_path = str(failure.get("path") or "")
            if file_path and file_path not in successful_files:
                unresolved_failures[file_path] = failure
        shard_audit.append({
            "path": str(path),
            "status": payload.get("status"),
            "files_expected": payload.get("files_expected"),
            "files_processed": payload.get("files_processed"),
            "files_failed": payload.get("files_failed"),
        })
    if first is None or samples is None:
        raise ValueError("no TnP shards supplied")
    if expected_records is not None:
        expected_files = {str(record["file_path"]) for record in expected_records}
        if len(expected_files) != len(expected_records):
            raise RuntimeError("duplicate ROOT file in expected TnP campaign records")
        unexpected = successful_files - expected_files
        if unexpected:
            raise RuntimeError(f"successful ROOT files absent from expected campaign: {sorted(unexpected)[:5]}")
        # The explicit expected-records file is the frozen campaign scope.
        # Failed inputs that were permanently skipped may still be mentioned
        # in their original shard metadata, but must not re-enter coverage or
        # unresolved-failure accounting after the audited exclusion.
        seen_files = set(expected_files)
        unresolved_failures = {
            path: failure for path, failure in unresolved_failures.items()
            if path in expected_files
        }
        for file_path in expected_files - successful_files:
            unresolved_failures.setdefault(file_path, {
                "path": file_path,
                "error": "no successful TnP shard or recovery output",
            })
    failures = [unresolved_failures[path] for path in sorted(unresolved_failures)]
    blockers = list(first.get("adoption_blockers") or [])
    if failures:
        blockers.append(f"{len(failures)} ROOT files failed")
    if len(mc_datasets) > 1:
        blockers.append(f"multiple MC physical datasets require explicit relative normalization: {sorted(mc_datasets)}")
    return json_safe({
        "schema_version": 1,
        "measurement": config["measurement"],
        "source_measurement": expected_measurement,
        "year": str(config.get("year") or first.get("year") or "2024"),
        "status": "candidate_histograms" if not blockers else "blocked",
        "kind": kind,
        "probe_definition": first.get("probe_definition"),
        "denominator_selection": config.get("denominator", first.get("denominator_selection")),
        "target_selection": config.get("target_selection", first.get("target_selection")),
        "binning_policy": config.get("binning_policy"),
        "tag_pt_min_gev": first.get("tag_pt_min_gev"),
        "tag_miniiso_max": first.get("tag_miniiso_max", 0.1),
        "tag_selection": first.get("tag_selection"),
        "external_reference_muon": first.get("external_reference_muon"),
        "tag_trigger_match_required": first.get("tag_trigger_match_required", True),
        "reference_trigger_object_kind": first.get("reference_trigger_object_kind", kind),
        "reference_trigger_application": first.get(
            "reference_trigger_application",
            {"data": True, "mc": True},
        ),
        "probe_abseta_edges": first["probe_abseta_edges"],
        "probe_pt_edges_gev": first["probe_pt_edges_gev"],
        "mass_edges_gev": first["mass_edges_gev"],
        "fit_window_gev": first["fit_window_gev"],
        "nominal_mass_rebin_factor": int(config.get("nominal_mass_rebin_factor", 1)),
        "nominal_background_model": str(
            config.get("nominal_background_model", "chebyshev")
        ),
        "samples": samples,
        "pileup_correction": first.get("pileup_correction"),
        "trigger_object_filter_bits": first.get("trigger_object_filter_bits"),
        "files_expected": len(seen_files),
        "files_processed": len(successful_files),
        "files_failed": failures,
        "mc_physical_datasets": sorted(mc_datasets),
        "shards": shard_audit,
        "adoption_blockers": blockers,
        "created_unix": time.time(),
    })


def cli(argv: list[str] | None = None, *, default_kind: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        help=(
            "input filename pattern; repeat to add patterns "
            "(default: shard_*.json plus shard_recovery_*.json)"
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("electron", "muon"), default=default_kind, required=default_kind is None)
    parser.add_argument("--target-eta-edges", type=float, nargs="+")
    parser.add_argument("--target-pt-edges", type=float, nargs="+")
    parser.add_argument(
        "--mc-reference-histograms",
        type=Path,
        help=(
            "replace only MC and MC pileup-variation histograms with a fully "
            "compatible, explicitly recorded reference payload"
        ),
    )
    parser.add_argument(
        "--mc-reference-year",
        help="explicit year of --mc-reference-histograms when absent from that payload",
    )
    args = parser.parse_args(argv)
    globs = args.globs or ["shard_*.json", "shard_recovery_*.json"]
    paths = sorted({
        path
        for directory in args.input_dir
        for pattern in globs
        for path in directory.glob(pattern)
    })
    if not paths:
        raise FileNotFoundError(
            f"no TnP shard JSONs match {globs!r} in {', '.join(map(str, args.input_dir))}"
        )
    expected_records = None
    if args.records is not None:
        expected_records = list(json.loads(args.records.read_text()).get("records") or [])
    config = json.loads(args.config.read_text())
    payload = merge_tnp_shards(
        paths,
        kind=args.kind,
        config=config,
        expected_records=expected_records,
    )
    target_eta = (
        [float(value) for value in args.target_eta_edges]
        if args.target_eta_edges
        else [float(value) for value in config["probe_abseta_edges"]]
    )
    target_pt = (
        [float(value) for value in args.target_pt_edges]
        if args.target_pt_edges
        else [float(value) for value in config["probe_pt_edges_gev"]]
    )
    if (
        not np.array_equal(np.asarray(payload["probe_abseta_edges"]), np.asarray(target_eta))
        or not np.array_equal(np.asarray(payload["probe_pt_edges_gev"]), np.asarray(target_pt))
    ):
        payload = rebin_probe_histograms(
            payload,
            target_eta_edges=target_eta,
            target_pt_edges=target_pt,
        )
    if args.mc_reference_year is not None and args.mc_reference_histograms is None:
        raise ValueError("--mc-reference-year requires --mc-reference-histograms")
    if args.mc_reference_histograms is not None:
        reference = json.loads(args.mc_reference_histograms.read_text())
        payload = use_mc_reference(
            payload,
            reference,
            source=args.mc_reference_histograms,
            reference_year=args.mc_reference_year,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if payload["files_processed"] else 2


if __name__ == "__main__":
    raise SystemExit(cli())

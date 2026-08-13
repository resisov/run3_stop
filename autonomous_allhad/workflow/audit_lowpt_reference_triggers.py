#!/usr/bin/env python3
"""Audit low-pT lepton reference HLT rates and NanoAOD trigger-object bits.

The output is deliberately diagnostic rather than an SF input.  In particular,
it records the filter bits on trigger objects geometrically matched to the
prospective tight tag, separately for every fired reference path.  A filter bit
is adopted in the TnP configuration only after this audit is checked against
the 2024 HLT menu and an unbiased-probe test.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot


def _delta_phi(left: Any, right: Any) -> Any:
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def _branches(kind: str, paths: list[str]) -> list[str]:
    prefix = "Electron" if kind == "electron" else "Muon"
    fields = ["pt", "eta", "phi", "miniPFRelIso_all"]
    fields += ["cutBased"] if kind == "electron" else ["tightId", "looseId"]
    return [
        *paths,
        *(f"{prefix}_{field}" for field in fields),
        "TrigObj_pt",
        "TrigObj_eta",
        "TrigObj_phi",
        "TrigObj_id",
        "TrigObj_filterBits",
    ]


def _tag_mask(arrays: Any, kind: str, tag_pt_min_gev: float) -> Any:
    if kind == "electron":
        eta = arrays["Electron_eta"]
        fiducial = (abs(eta) < 1.4442) | ((abs(eta) > 1.5660) & (abs(eta) < 2.5))
        return (
            (arrays["Electron_pt"] > tag_pt_min_gev)
            & fiducial
            & (arrays["Electron_cutBased"] >= 4)
            & (arrays["Electron_miniPFRelIso_all"] < 0.1)
        )
    return (
        (arrays["Muon_pt"] > tag_pt_min_gev)
        & (abs(arrays["Muon_eta"]) < 2.4)
        & arrays["Muon_tightId"]
        & (arrays["Muon_miniPFRelIso_all"] < 0.1)
    )


def _matched_filter_bits(arrays: Any, kind: str, tag_pt_min_gev: float) -> Any:
    prefix = "Electron" if kind == "electron" else "Muon"
    object_id = 11 if kind == "electron" else 13
    tag = _tag_mask(arrays, kind, tag_pt_min_gev)
    deta = arrays[f"{prefix}_eta"][:, :, None] - arrays["TrigObj_eta"][:, None, :]
    dphi = _delta_phi(
        arrays[f"{prefix}_phi"][:, :, None], arrays["TrigObj_phi"][:, None, :]
    )
    matched = (
        tag[:, :, None]
        & (abs(arrays["TrigObj_id"][:, None, :]) == object_id)
        & ((deta * deta + dphi * dphi) < 0.1**2)
    )
    trigger_bits, matched = ak.broadcast_arrays(
        arrays["TrigObj_filterBits"][:, None, :], matched
    )
    return ak.flatten(trigger_bits[matched], axis=None)


def _accumulate_bits(values: Any, exact: Counter[int], individual: Counter[int]) -> int:
    flat = np.asarray(ak.to_numpy(ak.flatten(values, axis=None)), dtype=np.int64)
    exact.update(int(value) for value in flat)
    for bit in range(32):
        individual[bit] += int(np.count_nonzero((flat & (1 << bit)) != 0))
    return len(flat)


def audit(
    *,
    kind: str,
    files: list[str],
    paths: list[str],
    step_size: int,
    max_events: int | None,
    tag_pt_min_gev: float = 5.0,
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    exact_by_path = {path: Counter() for path in paths}
    bits_by_path = {path: Counter() for path in paths}
    matched_by_path: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    events_read = 0
    files_processed = 0
    present_by_file: dict[str, list[str]] = {}

    for file_path in files:
        if max_events is not None and events_read >= max_events:
            break
        try:
            tree = uproot.open(file_path)["Events"]
            present = set(tree.keys())
            available = [path for path in paths if path in present]
            present_by_file[file_path] = available
            if not available:
                raise RuntimeError("none of the requested HLT paths is present")
            requested = _branches(kind, available)
            missing = sorted(set(requested) - present)
            if missing:
                raise RuntimeError(f"required branches missing: {', '.join(missing)}")
            for arrays in tree.iterate(requested, step_size=step_size, library="ak"):
                if max_events is not None:
                    remaining = max_events - events_read
                    if remaining <= 0:
                        break
                    arrays = arrays[:remaining]
                n_events = len(arrays)
                events_read += n_events
                event_counts["all"] += n_events
                fired_or = np.zeros(n_events, dtype=bool)
                for path in available:
                    fired = np.asarray(arrays[path], dtype=bool)
                    event_counts[path] += int(np.count_nonzero(fired))
                    fired_or |= fired
                    if np.any(fired):
                        matched = _matched_filter_bits(arrays[fired], kind, tag_pt_min_gev)
                        matched_by_path[path] += _accumulate_bits(
                            matched, exact_by_path[path], bits_by_path[path]
                        )
                event_counts["reference_or"] += int(np.count_nonzero(fired_or))
            files_processed += 1
        except Exception as exc:
            failures.append({"path": file_path, "error": f"{type(exc).__name__}: {exc}"})

    def serialise_counter(counter: Counter[Any]) -> dict[str, int]:
        return {str(key): int(value) for key, value in sorted(counter.items()) if value}

    return {
        "schema_version": 1,
        "kind": kind,
        "status": "diagnostic_not_for_adoption",
        "files_requested": len(files),
        "files_processed": files_processed,
        "file_failures": failures,
        "events_read": events_read,
        "event_counts": serialise_counter(event_counts),
        "paths_present_by_file": present_by_file,
        "matched_trigger_objects": serialise_counter(matched_by_path),
        "exact_filter_bits_by_path": {
            path: serialise_counter(exact_by_path[path]) for path in paths
        },
        "individual_bit_index_counts_by_path": {
            path: serialise_counter(bits_by_path[path]) for path in paths
        },
        "matching": {
            "delta_r_max": 0.1,
            "tag_pt_min_gev": tag_pt_min_gev,
            "electron_tag": f"pt>{tag_pt_min_gev:g}, fiducial, cutBased>=4, miniPFRelIso_all<0.1",
            "muon_tag": f"pt>{tag_pt_min_gev:g}, abs(eta)<2.4, tightId, miniPFRelIso_all<0.1",
        },
        "created_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("electron", "muon"), required=True)
    parser.add_argument("--file", action="append", dest="files", required=True)
    parser.add_argument("--path", action="append", dest="paths", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-size", type=int, default=100_000)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--tag-pt-min-gev", type=float, default=5.0)
    args = parser.parse_args()
    result = audit(
        kind=args.kind,
        files=args.files,
        paths=args.paths,
        step_size=args.step_size,
        max_events=args.max_events,
        tag_pt_min_gev=args.tag_pt_min_gev,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["files_processed"] and not result["file_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

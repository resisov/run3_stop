#!/usr/bin/env python3
"""Validate every diagonal-v3 cache shard and write machine-readable state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import uproot


EXPECTED_BRANCHES = {
    "feature_lowdm_diagonal_v3_SR",
    "nresolved_top_trota",
    "physical_dataset_id",
    "run",
    "luminosityBlock",
    "event",
    "is_signal",
    "is_background",
    "gen_weight",
    "jet_corrected_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_corrected_mass",
    "jet_btag_upart_all",
    "lowdm_met_sqrt_ht",
    "n_lowdm_isr",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tree_fields(tree: Any) -> set[str]:
    names = tree.keys() if hasattr(tree, "keys") else tree.fields
    return {str(name).split(";", 1)[0] for name in names}


def validate_one(request: dict[str, Any], outputs: Path) -> dict[str, Any]:
    expected = Path(str(request["output"]))
    root_path = outputs / expected.name
    sidecar_path = root_path.with_suffix(".json")
    row: dict[str, Any] = {
        "kind": request["kind"],
        "batch": int(request["batch"]),
        "output": str(root_path),
        "sidecar": str(sidecar_path),
        "status": "failed",
    }
    if not root_path.is_file() or not sidecar_path.is_file():
        row["error"] = "ROOT output or sidecar is missing"
        return row
    try:
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("status") != "complete":
            raise RuntimeError("sidecar status is not complete")
        if sidecar.get("selection_mode") != "diagonal_v3":
            raise RuntimeError("sidecar selection_mode is not diagonal_v3")
        events_selected = int(sidecar["events_selected"])
        with uproot.open(root_path) as root_file:
            if events_selected:
                if "Events" not in root_file:
                    raise RuntimeError("nonempty output has no Events tree")
                tree = root_file["Events"]
                if int(tree.num_entries) != events_selected:
                    raise RuntimeError("ROOT entry count differs from sidecar")
                missing = sorted(EXPECTED_BRANCHES - tree_fields(tree))
                if missing:
                    raise RuntimeError("missing branches: " + ", ".join(missing))
            elif "Events" in root_file and int(root_file["Events"].num_entries):
                raise RuntimeError("zero-event sidecar has a nonempty Events tree")
        requested_inputs = {str(item["root"]) for item in request["inputs"]}
        valid_inputs = set(map(str, sidecar.get("input_files") or []))
        if not valid_inputs <= requested_inputs:
            raise RuntimeError("sidecar contains an unrequested input")
        row.update(
            status="complete",
            events_selected=events_selected,
            output_size_bytes=root_path.stat().st_size,
            input_files_requested=len(requested_inputs),
            input_files_valid=len(valid_inputs),
            audit=sidecar.get("audit") or {},
            bad_files=sidecar.get("bad_files") or [],
        )
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--max-skipped-file-fraction", type=float, default=0.0)
    parser.add_argument("--max-skipped-event-fraction", type=float, default=0.0)
    opts = parser.parse_args()
    campaign = json.loads(opts.campaign_manifest.read_text())
    if campaign.get("schema_version") != "gnn_lowdm_diagonal_v3_condor_campaign_v1":
        raise RuntimeError("unexpected Condor campaign schema")
    request_paths = sorted(opts.requests.glob("*.json"))
    if len(request_paths) != int(campaign["jobs"]):
        raise RuntimeError("request count differs from campaign manifest")
    started = time.time()
    requests = [json.loads(path.read_text()) for path in request_paths]
    results = [validate_one(request, opts.outputs) for request in requests]
    failures = [row for row in results if row["status"] != "complete"]
    bad_files = [
        bad for row in results for bad in (row.get("bad_files") or [])
    ]
    metadata_by_root = {
        str(item["root"]): item
        for request in requests
        for item in request["inputs"]
    }
    for bad in bad_files:
        metadata = metadata_by_root.get(str(bad.get("file")), {})
        bad["events_read"] = int(metadata.get("events_read", 0))
        bad["events_written"] = int(metadata.get("events_written", 0))
        bad["root_size_bytes"] = int(metadata.get("root_size_bytes", 0))
        bad["physical_dataset_ids"] = metadata.get("physical_dataset_ids", [])
        bad["processes"] = metadata.get("processes", [])
    audit: dict[str, int] = {}
    for row in results:
        for key, value in (row.get("audit") or {}).items():
            audit[key] = int(audit.get(key, 0)) + int(value)
    unique_requested_inputs = set(metadata_by_root)
    total_input_events = sum(
        int(item.get("events_written", 0)) for item in metadata_by_root.values()
    )
    lost_input_events = sum(int(item.get("events_written", 0)) for item in bad_files)
    skipped_file_fraction = len(bad_files) / max(len(unique_requested_inputs), 1)
    skipped_event_fraction = lost_input_events / max(total_input_events, 1)
    exceeds_skip_threshold = (
        skipped_file_fraction > opts.max_skipped_file_fraction
        or skipped_event_fraction > opts.max_skipped_event_fraction
    )
    campaign_complete = not failures and not exceeds_skip_threshold
    state = {
        "schema_version": "gnn_lowdm_diagonal_v3_feature_cache_campaign_v1",
        "status": "complete" if campaign_complete else "incomplete",
        "selection_mode": "diagonal_v3",
        "selection_policy": campaign["selection"],
        "manifest": str(opts.campaign_manifest),
        "batches_total": len(results),
        "batches_complete": len(results) - len(failures),
        "batches_failed": len(failures),
        "events_selected": int(audit.get("selected_events", 0)),
        "input_files_requested": len(unique_requested_inputs),
        "input_files_valid": sum(
            int(row.get("input_files_valid", 0)) for row in results
        ),
        "input_files_skipped": len(bad_files),
        "skipped_file_fraction": skipped_file_fraction,
        "intermediate_events_requested": total_input_events,
        "intermediate_events_lost": lost_input_events,
        "skipped_intermediate_event_fraction": skipped_event_fraction,
        "skip_thresholds": {
            "maximum_file_fraction": opts.max_skipped_file_fraction,
            "maximum_intermediate_event_fraction": opts.max_skipped_event_fraction,
            "exceeded": exceeds_skip_threshold,
        },
        "generator_sumw_loss": {
            "status": "not_available_per_intermediate_shard",
            "claim_complete_normalization": False if bad_files else True,
        },
        "output_size_bytes": sum(
            int(row.get("output_size_bytes", 0)) for row in results
        ),
        "audit": audit,
        "results": results,
        "test_partition_touched": False,
        "runtime_seconds": time.time() - started,
    }
    write_json(opts.outputs / "campaign_state.json", state)
    write_json(
        opts.outputs / "bad_files.json",
        {
            "schema_version": "gnn_lowdm_bad_files_v1",
            "status": "complete",
            "records": bad_files,
        },
    )
    (opts.outputs / "bad_files.txt").write_text(
        "".join(
            f"{row.get('file', '')}\t{row.get('error', '')}\n"
            for row in bad_files
        )
    )
    write_json(
        opts.outputs / "file_validation_summary.json",
        {
            "schema_version": "gnn_lowdm_diagonal_v3_file_validation_summary_v1",
            "status": state["status"],
            "intermediate_files_requested": len(unique_requested_inputs),
            "intermediate_files_valid": state["input_files_valid"],
            "intermediate_files_skipped": len(bad_files),
            "skipped_file_fraction": skipped_file_fraction,
            "intermediate_events_requested": total_input_events,
            "intermediate_events_lost": lost_input_events,
            "skipped_intermediate_event_fraction": skipped_event_fraction,
            "skip_thresholds": state["skip_thresholds"],
            "failed_cache_batches": len(failures),
            "bad_files_manifest": str(opts.outputs / "bad_files.json"),
        },
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

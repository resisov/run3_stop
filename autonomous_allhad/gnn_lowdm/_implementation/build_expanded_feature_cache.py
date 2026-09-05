#!/usr/bin/env python3
"""Build resumable Low-dM feature caches from nominal intermediate ROOT.

The diagonal-v3 mode applies the zero-lepton preselection, stored boosted
top/W veto, Nb>=1, and a recomputed TROTA Nres==0 veto.  It deliberately does
not select on MET/sqrt(HT), NISR, ISR delta-phi, or the original High-dM SR
flag; those quantities remain available as features or overlap audits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from autonomous_allhad.autonomous_allhad.highdm_resolved_categories import (
        boosted_overlap_vetoed_ak4_indices,
        map_candidates_to_events,
        map_candidates_to_events_rle,
        select_exclusive_resolved_candidates,
    )
except ImportError:
    from highdm_resolved_categories import (  # type: ignore[no-redef]
        boosted_overlap_vetoed_ak4_indices,
        map_candidates_to_events,
        map_candidates_to_events_rle,
        select_exclusive_resolved_candidates,
    )


EXPECTED_TROTA_SCHEMA = "trota_topresolved_2024_inplace_sparse_v1"
EXPECTED_TROTA_MODEL_SHA256 = (
    "ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3"
)
DERIVED_SELECTION_BRANCH = "feature_lowdm_expanded_SR"
DIAGONAL_V3_SELECTION_BRANCH = "feature_lowdm_diagonal_v3_SR"
DERIVED_NRES_BRANCH = "nresolved_top_trota"

SELECTION_MODES = {
    "expanded_v1": {
        "derived_branch": DERIVED_SELECTION_BRANCH,
        "require_met_sqrt_ht": True,
        "description": (
            "kind && feature_lowdm_preselection && pass_lowdm_topology_veto "
            "&& pass_lowdm_met_sqrt_ht && nb_medium_lowdm>=1 "
            "&& nresolved_top_trota==0; no NISR or ISR-dphi requirement; "
            "no !feature_SR veto"
        ),
    },
    "diagonal_v3": {
        "derived_branch": DIAGONAL_V3_SELECTION_BRANCH,
        "require_met_sqrt_ht": False,
        "description": (
            "kind && feature_lowdm_preselection && pass_lowdm_topology_veto "
            "&& nb_medium_lowdm>=1 && nresolved_top_trota==0; "
            "MET/sqrt(HT), NISR, ISR-dphi, and !feature_SR are not selected"
        ),
    },
}

FEATURE_BRANCHES = (
    "physical_dataset_id",
    "run",
    "luminosityBlock",
    "event",
    "file_id",
    "is_signal",
    "is_background",
    "mStop",
    "mLSP",
    "signal_topology_id",
    "process_id",
    "dataset_id",
    "gen_weight",
    "feature_lowdm_preselection",
    "feature_lowdm_SR",
    "feature_SR",
    "lowdm_search_bin_SR",
    "pass_lowdm_topology_veto",
    "pass_lowdm_isr",
    "pass_lowdm_met_sqrt_ht",
    "n_lowdm_isr",
    "nb_medium_lowdm",
    "n_sv_softb",
    "j1_met_dphi",
    "j2_met_dphi",
    "j3_met_dphi",
    "j4_met_dphi",
    "min_dphi4",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
    "lowdm_met_sqrt_ht",
    "lowdm_ptb",
    "lowdm_mtb",
    "lowdm_isr_subjet_btag_max",
    "lowdm_fatjet_pt",
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
    "lowdm_fatjet_msd",
    "met",
    "met_phi",
    "ht",
    "njet",
    "nb_medium",
    "jet_corrected_pt",
    "jet_eta_all",
    "jet_phi_all",
    "jet_corrected_mass",
    "jet_btag_upart_all",
    "jet_id_all",
)

NRES_EVENT_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    "file_id",
    "entry",
    "feature_lowdm_preselection",
    "pass_lowdm_topology_veto",
    "pass_lowdm_met_sqrt_ht",
    "nb_medium_lowdm",
    "jet_source_index_all",
    "jet_eta_all",
    "jet_phi_all",
    "fatjet_eta_all",
    "fatjet_phi_all",
    "fatjet_subjet_index1_all",
    "fatjet_subjet_index2_all",
    "fatjet_boosted_top_pass_all",
    "fatjet_boosted_w_pass_all",
    "subjet_eta_all",
    "subjet_phi_all",
)
TROTA_PRIMARY_BRANCHES = (
    "file_id",
    "entry",
    "TopResolved1pct_candidateIndex",
    "TopResolved1pct_sourceJetIdx0",
    "TopResolved1pct_sourceJetIdx1",
    "TopResolved1pct_sourceJetIdx2",
    "TopResolved1pct_eta",
    "TopResolved1pct_mass",
    "TopResolved1pct_QCDDiscriminant",
)
TROTA_FALLBACK_BRANCHES = (
    "run",
    "luminosityBlock",
    "event",
    *TROTA_PRIMARY_BRANCHES[2:],
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_trota_provenance(sidecar_path: str) -> dict[str, object]:
    sidecar = json.loads(Path(sidecar_path).read_text())
    payload = sidecar.get("trota_topresolved_2024") or {}
    marker = payload.get("marker") or {}
    if (
        payload.get("status") != "complete"
        or payload.get("schema_version") != EXPECTED_TROTA_SCHEMA
        or marker.get("status") != "complete"
        or marker.get("model_sha256") != EXPECTED_TROTA_MODEL_SHA256
    ):
        raise RuntimeError(f"invalid TROTA provenance: {sidecar_path}")
    return {
        "schema_version": payload["schema_version"],
        "model_sha256": marker["model_sha256"],
        "selected_working_point": marker.get("selected_working_point"),
    }


def compute_expanded_nres(
    light: Any,
    trota_tree: Any,
    *,
    require_met_sqrt_ht: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute exclusive TROTA Nres for the expanded Low-dM seed."""
    import awkward as ak

    event_fields = set(light.fields)
    missing = sorted(set(NRES_EVENT_BRANCHES) - event_fields)
    if missing:
        raise RuntimeError("missing Nres Events branches: " + ", ".join(missing))
    event_count = len(light)
    eligible = (
        np.asarray(light["feature_lowdm_preselection"], dtype=bool)
        & np.asarray(light["pass_lowdm_topology_veto"], dtype=bool)
        & (np.asarray(light["nb_medium_lowdm"], dtype=np.int32) >= 1)
    )
    if require_met_sqrt_ht:
        eligible &= np.asarray(light["pass_lowdm_met_sqrt_ht"], dtype=bool)
    counts = np.zeros(event_count, dtype=np.int16)

    trota_fields = set(trota_tree.keys())
    identity_fallback = 0
    if set(TROTA_PRIMARY_BRANCHES) <= trota_fields:
        fields = TROTA_PRIMARY_BRANCHES
        arrays_ak = trota_tree.arrays(fields, library="ak")
        arrays = {name: np.asarray(ak.to_numpy(arrays_ak[name])) for name in fields}
        candidate_event = map_candidates_to_events(
            np.asarray(light["file_id"]),
            np.asarray(light["entry"]),
            arrays["file_id"],
            arrays["entry"],
        )
    elif set(TROTA_FALLBACK_BRANCHES) <= trota_fields:
        fields = TROTA_FALLBACK_BRANCHES
        arrays_ak = trota_tree.arrays(fields, library="ak")
        arrays = {name: np.asarray(ak.to_numpy(arrays_ak[name])) for name in fields}
        candidate_event = map_candidates_to_events_rle(
            np.asarray(light["run"]),
            np.asarray(light["luminosityBlock"]),
            np.asarray(light["event"]),
            arrays["run"],
            arrays["luminosityBlock"],
            arrays["event"],
        )
        identity_fallback = 1
    else:
        raise RuntimeError("TROTA tree has neither validated identity schema")

    fiducial = (
        eligible[candidate_event]
        & np.isfinite(arrays["TopResolved1pct_eta"])
        & np.isfinite(arrays["TopResolved1pct_mass"])
        & np.isfinite(arrays["TopResolved1pct_QCDDiscriminant"])
        & (np.abs(arrays["TopResolved1pct_eta"]) < 2.0)
        & (arrays["TopResolved1pct_mass"] >= 100.0)
        & (arrays["TopResolved1pct_mass"] <= 250.0)
    )
    selected_rows = np.flatnonzero(fiducial)
    rejected_boosted = 0
    rejected_resolved = 0
    if selected_rows.size:
        order = np.argsort(candidate_event[selected_rows], kind="stable")
        selected_rows = selected_rows[order]
        selected_events = candidate_event[selected_rows]
        boundaries = np.flatnonzero(np.diff(selected_events)) + 1
        for rows in np.split(selected_rows, boundaries):
            event_index = int(candidate_event[rows[0]])
            vetoed = boosted_overlap_vetoed_ak4_indices(
                jet_source_indices=ak.to_list(light["jet_source_index_all"][event_index]),
                jet_eta=ak.to_list(light["jet_eta_all"][event_index]),
                jet_phi=ak.to_list(light["jet_phi_all"][event_index]),
                fatjet_eta=ak.to_list(light["fatjet_eta_all"][event_index]),
                fatjet_phi=ak.to_list(light["fatjet_phi_all"][event_index]),
                fatjet_subjet_index1=ak.to_list(light["fatjet_subjet_index1_all"][event_index]),
                fatjet_subjet_index2=ak.to_list(light["fatjet_subjet_index2_all"][event_index]),
                fatjet_top_pass=ak.to_list(light["fatjet_boosted_top_pass_all"][event_index]),
                fatjet_w_pass=ak.to_list(light["fatjet_boosted_w_pass_all"][event_index]),
                subjet_eta=ak.to_list(light["subjet_eta_all"][event_index]),
                subjet_phi=ak.to_list(light["subjet_phi_all"][event_index]),
            )
            result = select_exclusive_resolved_candidates(
                candidate_indices=arrays["TopResolved1pct_candidateIndex"][rows],
                candidate_scores=arrays["TopResolved1pct_QCDDiscriminant"][rows],
                candidate_source_jets=np.stack(
                    [
                        arrays["TopResolved1pct_sourceJetIdx0"][rows],
                        arrays["TopResolved1pct_sourceJetIdx1"][rows],
                        arrays["TopResolved1pct_sourceJetIdx2"][rows],
                    ],
                    axis=1,
                ),
                boosted_vetoed_ak4_indices=vetoed,
            )
            counts[event_index] = result.nres
            rejected_boosted += len(result.rejected_by_boosted_overlap)
            rejected_resolved += len(result.rejected_by_resolved_overlap)
    return counts, {
        "events": event_count,
        "eligible_events": int(np.count_nonzero(eligible)),
        "nres_positive_events": int(np.count_nonzero(eligible & (counts > 0))),
        "trota_rows": int(trota_tree.num_entries),
        "fiducial_rows": int(selected_rows.size),
        "rejected_by_boosted_overlap": int(rejected_boosted),
        "rejected_by_resolved_overlap": int(rejected_resolved),
        "identity_fallback_files": identity_fallback,
    }


def lowdm_cache_selection_mask(
    arrays: Any,
    nres: np.ndarray,
    kind: str,
    *,
    require_met_sqrt_ht: bool,
) -> np.ndarray:
    """Return the auditable cache-domain mask for one source shard."""
    if kind not in {"mc", "signal"}:
        raise ValueError(f"unsupported supervised kind: {kind}")
    selected = (
        np.asarray(arrays["is_background" if kind == "mc" else "is_signal"], dtype=bool)
        & np.asarray(arrays["feature_lowdm_preselection"], dtype=bool)
        & np.asarray(arrays["pass_lowdm_topology_veto"], dtype=bool)
        & (np.asarray(arrays["nb_medium_lowdm"], dtype=np.int32) >= 1)
        & (np.asarray(nres, dtype=np.int16) == 0)
    )
    if require_met_sqrt_ht:
        selected &= np.asarray(arrays["pass_lowdm_met_sqrt_ht"], dtype=bool)
    return selected


def process_source(
    record: dict[str, object],
    kind: str,
    *,
    selection_mode: str = "expanded_v1",
    root_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    import awkward as ak
    import uproot

    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"unknown cache selection mode: {selection_mode}")
    mode = SELECTION_MODES[selection_mode]
    require_met_sqrt_ht = bool(mode["require_met_sqrt_ht"])
    root_path = root_override or str(record["root"])
    provenance = validate_trota_provenance(str(record["sidecar"]))
    with uproot.open(root_path) as root_file:
        if "Events" not in root_file or "TROTA" not in root_file:
            raise RuntimeError("required Events/TROTA tree is missing")
        event_tree = root_file["Events"]
        read_branches = tuple(dict.fromkeys((*FEATURE_BRANCHES, *NRES_EVENT_BRANCHES)))
        missing = sorted(set(read_branches) - set(event_tree.keys()))
        if missing:
            raise RuntimeError("missing cache branches: " + ", ".join(missing))
        arrays = event_tree.arrays(read_branches, library="ak")
        nres, nres_stats = compute_expanded_nres(
            arrays,
            root_file["TROTA"],
            require_met_sqrt_ht=require_met_sqrt_ht,
        )
    kind_mask = np.asarray(
        arrays["is_background" if kind == "mc" else "is_signal"], dtype=bool
    )
    selected = lowdm_cache_selection_mask(
        arrays,
        nres,
        kind,
        require_met_sqrt_ht=require_met_sqrt_ht,
    )
    payload = {name: arrays[name][selected] for name in FEATURE_BRANCHES}
    payload[str(mode["derived_branch"])] = ak.Array(
        np.ones(int(np.count_nonzero(selected)), dtype=np.bool_)
    )
    payload[DERIVED_NRES_BRANCH] = ak.Array(nres[selected])
    n_isr = np.asarray(arrays["n_lowdm_isr"], dtype=np.int32)
    old_sr = np.asarray(arrays["feature_lowdm_SR"], dtype=bool)
    highdm = np.asarray(arrays["feature_SR"], dtype=bool)
    stats = {
        **nres_stats,
        "selected_events": int(np.count_nonzero(selected)),
        "selected_nisr0": int(np.count_nonzero(selected & (n_isr == 0))),
        "selected_nisr1": int(np.count_nonzero(selected & (n_isr == 1))),
        "selected_nisr2plus": int(np.count_nonzero(selected & (n_isr >= 2))),
        "selected_old_lowdm_sr": int(np.count_nonzero(selected & old_sr)),
        "selected_pass_met_sqrt_ht": int(
            np.count_nonzero(
                selected
                & np.asarray(arrays["pass_lowdm_met_sqrt_ht"], dtype=bool)
            )
        ),
        "selected_fail_met_sqrt_ht": int(
            np.count_nonzero(
                selected
                & ~np.asarray(arrays["pass_lowdm_met_sqrt_ht"], dtype=bool)
            )
        ),
        "selected_highdm_sr_overlap": int(np.count_nonzero(selected & highdm)),
        "kind_highdm_sr_events": int(np.count_nonzero(kind_mask & highdm)),
        "trota_provenance_valid": int(bool(provenance)),
    }
    return payload, stats


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def alternate_eos_endpoint(path: str) -> str | None:
    """Return an independent XRootD view for a local EOS FUSE path."""
    if path.startswith("/eos/user/"):
        return "root://eosuser.cern.ch/" + path
    return None


def process_source_with_retries(
    record: dict[str, object],
    kind: str,
    *,
    selection_mode: str,
) -> tuple[dict[str, Any], dict[str, int], dict[str, object]]:
    """Retry the FUSE path once, then try the independent EOS endpoint."""
    original = str(record["root"])
    alternate = alternate_eos_endpoint(original)
    failures: list[dict[str, str]] = []
    for attempt, endpoint in enumerate((original, original, alternate), start=1):
        if endpoint is None:
            continue
        try:
            payload, stats = process_source(
                record,
                kind,
                selection_mode=selection_mode,
                root_override=endpoint,
            )
            return payload, stats, {
                "attempts": attempt,
                "access_endpoint": endpoint,
                "failures_before_success": failures,
            }
        except Exception as error:
            failures.append(
                {
                    "time": utc_now(),
                    "endpoint": endpoint,
                    "exception_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
            if attempt == 1:
                time.sleep(2.0)
    error = RuntimeError(f"source failed after {len(failures)} attempts: {original}")
    setattr(error, "source_failures", failures)
    setattr(error, "alternate_access_attempted", bool(alternate))
    raise error


def sum_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def worker(request_path: Path) -> int:
    import awkward as ak
    import uproot

    request = json.loads(request_path.read_text())
    selection_mode = str(request.get("selection_mode", "expanded_v1"))
    if selection_mode not in SELECTION_MODES:
        raise RuntimeError(f"unknown request selection mode: {selection_mode}")
    mode = SELECTION_MODES[selection_mode]
    derived_selection_branch = str(mode["derived_branch"])
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output_branches = (
        *FEATURE_BRANCHES,
        derived_selection_branch,
        DERIVED_NRES_BRANCH,
    )
    totals: dict[str, int] = {}
    bad_files: list[dict[str, object]] = []
    valid_files: list[str] = []
    access_audit = {
        "source_attempts": 0,
        "sources_retried": 0,
        "sources_recovered_by_alternate": 0,
    }
    started = time.time()
    output_tree = None
    with uproot.recreate(output, compression=uproot.LZ4(4)) as root_file:
        for record in request["inputs"]:
            try:
                payload, stats, access = process_source_with_retries(
                    record,
                    str(request["kind"]),
                    selection_mode=selection_mode,
                )
            except Exception as error:
                failures = list(getattr(error, "source_failures", []))
                access_audit["source_attempts"] += len(failures)
                access_audit["sources_retried"] += int(len(failures) > 1)
                bad_files.append(
                    {
                        "dataset": None,
                        "file": str(record["root"]),
                        "failure_stage": "expanded_cache_source_processing",
                        "exception_type": (
                            failures[-1]["exception_type"]
                            if failures
                            else type(error).__name__
                        ),
                        "error": (
                            failures[-1]["error"]
                            if failures
                            else str(error)[:500]
                        ),
                        "first_failure_time": (
                            failures[0]["time"] if failures else utc_now()
                        ),
                        "last_failure_time": (
                            failures[-1]["time"] if failures else utc_now()
                        ),
                        "alternate_access_attempted": bool(
                            getattr(error, "alternate_access_attempted", False)
                        ),
                        "permanently_skipped": True,
                        "attempts": failures,
                    }
                )
                continue
            source_events = int(stats["selected_events"])
            if source_events:
                try:
                    if output_tree is None:
                        branch_types = {
                            name: (
                                values.dtype
                                if isinstance(values, np.ndarray)
                                else ak.type(values).content
                            )
                            for name, values in payload.items()
                        }
                        output_tree = root_file.mktree("Events", branch_types)
                    output_tree.extend(payload)
                except Exception as error:
                    raise RuntimeError(
                        f"failed to append {record['root']} to cache output"
                    ) from error
            sum_stats(totals, stats)
            valid_files.append(str(record["root"]))
            access_audit["source_attempts"] += int(access["attempts"])
            access_audit["sources_retried"] += int(access["attempts"] > 1)
            access_audit["sources_recovered_by_alternate"] += int(
                str(access["access_endpoint"]) != str(record["root"])
            )
    sum_stats(totals, access_audit)
    selected_events = int(totals.get("selected_events", 0))
    if selected_events:
        with uproot.open(output) as root_file:
            if "Events" not in root_file or int(root_file["Events"].num_entries) != selected_events:
                raise RuntimeError("expanded cache output failed integrity validation")
    sidecar = {
        "schema_version": (
            "gnn_lowdm_diagonal_v3_feature_cache_shard_v1"
            if selection_mode == "diagonal_v3"
            else "gnn_lowdm_expanded_feature_cache_shard_v1"
        ),
        "status": "complete",
        "kind": request["kind"],
        "batch": int(request["batch"]),
        "selection_mode": selection_mode,
        "selection": mode["description"],
        "input_files_requested": len(request["inputs"]),
        "input_files_valid": len(valid_files),
        "input_files": valid_files,
        "events_selected": selected_events,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "branches": list(output_branches),
        "output_write_mode": "streaming_ttree_extend_per_source",
        "audit": totals,
        "bad_files": bad_files,
        "runtime_seconds": time.time() - started,
    }
    write_json(output.with_suffix(".json"), sidecar)
    print(json.dumps(sidecar, sort_keys=True), flush=True)
    return 0


def run_request(script: Path, request_path: Path, log_path: Path) -> dict[str, object]:
    started = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(
            [sys.executable, str(script), "--worker", str(request_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    request = json.loads(request_path.read_text())
    output = Path(request["output"])
    sidecar = output.with_suffix(".json")
    result: dict[str, object] = {
        "kind": request["kind"],
        "batch": request["batch"],
        "request": str(request_path),
        "output": str(output),
        "log": str(log_path),
        "exit_code": completed.returncode,
        "runtime_seconds": time.time() - started,
        "status": "failed",
    }
    if completed.returncode == 0 and output.exists() and sidecar.exists():
        payload = json.loads(sidecar.read_text())
        if payload.get("status") == "complete":
            result.update(
                status="complete",
                events_selected=int(payload["events_selected"]),
                output_size_bytes=int(payload["output_size_bytes"]),
                input_files_valid=int(payload["input_files_valid"]),
                audit=payload.get("audit") or {},
                bad_files=payload.get("bad_files") or [],
            )
    return result


def manager(opts: argparse.Namespace) -> int:
    manifest = json.loads(opts.manifest.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("full-campaign manifest is incomplete")
    opts.output.mkdir(parents=True, exist_ok=True)
    request_dir = opts.output / "requests"
    log_dir = opts.output / "logs"
    request_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    if opts.selection_mode not in SELECTION_MODES:
        raise RuntimeError(f"unknown cache selection mode: {opts.selection_mode}")
    mode = SELECTION_MODES[opts.selection_mode]
    pending: list[tuple[Path, Path]] = []
    results: list[dict[str, object]] = []
    all_requests: list[Path] = []
    for kind in ("mc", "signal"):
        records = [
            {"root": item["root"], "sidecar": item["sidecar"]}
            for item in manifest["shards"]
            if item["kind"] == kind
        ]
        for batch, start in enumerate(range(0, len(records), opts.files_per_batch)):
            request_path = request_dir / f"{kind}_{batch:04d}.json"
            output_path = opts.output / f"{kind}_cache_{batch:04d}.root"
            write_json(
                request_path,
                {
                    "schema_version": (
                        "gnn_lowdm_diagonal_v3_feature_cache_request_v1"
                        if opts.selection_mode == "diagonal_v3"
                        else "gnn_lowdm_expanded_feature_cache_request_v1"
                    ),
                    "kind": kind,
                    "batch": batch,
                    "selection_mode": opts.selection_mode,
                    "inputs": records[start : start + opts.files_per_batch],
                    "output": str(output_path),
                },
            )
            all_requests.append(request_path)
            sidecar = output_path.with_suffix(".json")
            if output_path.exists() and sidecar.exists():
                payload = json.loads(sidecar.read_text())
                if payload.get("status") == "complete":
                    results.append(
                        {
                            "kind": kind,
                            "batch": batch,
                            "request": str(request_path),
                            "output": str(output_path),
                            "log": str(log_dir / f"{kind}_{batch:04d}.log"),
                            "exit_code": 0,
                            "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
                            "status": "complete",
                            "events_selected": int(payload["events_selected"]),
                            "output_size_bytes": int(payload["output_size_bytes"]),
                            "input_files_valid": int(payload["input_files_valid"]),
                            "audit": payload.get("audit") or {},
                            "bad_files": payload.get("bad_files") or [],
                            "resumed_existing_output": True,
                        }
                    )
                    continue
            pending.append((request_path, log_dir / f"{kind}_{batch:04d}.log"))

    state_path = opts.output / "campaign_state.json"
    state: dict[str, object] = {
        "schema_version": (
            "gnn_lowdm_diagonal_v3_feature_cache_campaign_v1"
            if opts.selection_mode == "diagonal_v3"
            else "gnn_lowdm_expanded_feature_cache_campaign_v1"
        ),
        "status": "running",
        "manifest": str(opts.manifest),
        "selection_mode": opts.selection_mode,
        "selection_policy": mode["description"],
        "files_per_batch": opts.files_per_batch,
        "workers": opts.workers,
        "batches_total": len(all_requests),
        "batches_pending_at_start": len(pending),
        "batches_complete": len(results),
        "batches_failed": 0,
        "results": results,
        "started_at": time.time(),
    }
    write_json(state_path, state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.workers) as executor:
        futures = {
            executor.submit(run_request, script, request, log): request
            for request, log in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            state["results"].append(result)
            key = "batches_complete" if result["status"] == "complete" else "batches_failed"
            state[key] = int(state[key]) + 1
            state["updated_at"] = time.time()
            write_json(state_path, state)
            print(
                json.dumps(
                    {
                        "kind": result["kind"],
                        "batch": result["batch"],
                        "status": result["status"],
                        "complete": state["batches_complete"],
                        "failed": state["batches_failed"],
                        "total": state["batches_total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    state["status"] = "complete" if not state["batches_failed"] else "incomplete"
    state["completed_at"] = time.time()
    audit: dict[str, int] = {}
    bad_files = []
    for result in state["results"]:
        if result.get("status") == "complete":
            sum_stats(audit, result.get("audit") or {})
        bad_files.extend(result.get("bad_files") or [])
    state["audit"] = audit
    state["events_selected"] = int(audit.get("selected_events", 0))
    state["output_size_bytes"] = sum(
        int(result.get("output_size_bytes", 0)) for result in state["results"]
    )
    write_json(
        opts.output / "bad_files.json",
        {"schema_version": "gnn_lowdm_bad_files_v1", "status": "complete", "records": bad_files},
    )
    (opts.output / "bad_files.txt").write_text(
        "".join(f"{record['file']}\t{record['error']}\n" for record in bad_files)
    )
    write_json(
        opts.output / "file_validation_summary.json",
        {
            "schema_version": (
                "gnn_lowdm_diagonal_v3_file_validation_summary_v1"
                if opts.selection_mode == "diagonal_v3"
                else "gnn_lowdm_expanded_file_validation_summary_v1"
            ),
            "status": state["status"],
            "intermediate_files_requested": sum(len(json.loads(path.read_text())["inputs"]) for path in all_requests),
            "intermediate_files_valid": sum(int(result.get("input_files_valid", 0)) for result in state["results"]),
            "intermediate_files_skipped": len(bad_files),
            "bad_files_manifest": str(opts.output / "bad_files.json"),
        },
    )
    write_json(state_path, state)
    return 0 if state["status"] == "complete" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--files-per-batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--selection-mode",
        choices=tuple(SELECTION_MODES),
        default="expanded_v1",
    )
    parser.add_argument("--worker", type=Path)
    return parser.parse_args()


def main() -> int:
    opts = parse_args()
    if opts.worker is not None:
        return worker(opts.worker)
    if opts.manifest is None or opts.output is None:
        raise SystemExit("--manifest and --output are required in manager mode")
    return manager(opts)


if __name__ == "__main__":
    raise SystemExit(main())

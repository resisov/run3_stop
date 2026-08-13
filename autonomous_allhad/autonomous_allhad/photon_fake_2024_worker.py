from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import awkward as ak
import numpy as np

from . import flat_ntuple_worker as flat
from . import intermediate_2024_worker as intermediate
from . import real_subset_worker as baseline
from .object_corrections_2024 import (
    branch_audit,
    calibrate_jets_and_met,
    validate_payloads,
)
from .shape_histogram_2024_worker import (
    load_histogram_builder,
    physical_dataset,
    record_entry_bounds,
    required_read_branches,
)


OUTPUT_SCHEMA = "photon_fake_2024_sidecar_shard_v2"
PROBE_KINDS = (
    "target",
    "measurement_pass",
    "measurement_fail",
    "application",
    "plj_other",
)
ORIGINS = ("all", "prompt", "electron", "fake")
GCR_REGIONS = (
    "GCR",
    "GCR_Nt0",
    "GCR_Nt1",
    "GCR_DPhiVR_Low",
    "GCR_DPhiVR_High",
)
TRANSFER_PT_EDGES = (220.0, 300.0, 400.0, 600.0, 1_000_000.0)
TRANSFER_ETA_LABELS = ("EB", "EE")
VID_CUT_INDEX = {
    "pt": 0,
    "eta": 1,
    "sieie": 2,
    "hoe": 3,
    "charged_iso": 4,
    "ecal_iso": 5,
    "hcal_iso": 6,
}
EXTRA_BRANCHES = (
    "Photon_vidNestedWPBitmap",
    "Photon_genPartFlav",
    "Photon_pixelSeed",
    "Photon_sieie",
    "Photon_pfRelIso03_chg_quadratic",
)
PREFILTER_BRANCHES = (
    "run",
    "luminosityBlock",
    "Photon_pt",
    "Photon_eta",
    "Photon_electronVeto",
    "Photon_vidNestedWPBitmap",
    "Photon_cutBased",
    "Jet_btagUParTAK4B",
)
_PARALLEL_BUILDER: Any | None = None
_PARALLEL_PAYLOAD_STATUS: dict[str, Any] | None = None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vid_level(bitmap: Any, cut: str) -> Any:
    index = VID_CUT_INDEX[cut]
    values = ak.values_astype(bitmap, np.int64)
    return (values >> (2 * index)) & 0b11


def probe_masks(arrays: Any) -> dict[str, Any]:
    fields = set(arrays.fields)
    if "Photon_vidNestedWPBitmap" not in fields:
        raise RuntimeError("Photon_vidNestedWPBitmap is required for photon fake sidebands")
    p_pt = arrays["Photon_pt"]
    p_eta = arrays["Photon_eta"]
    p_electron_veto = ak.values_astype(arrays["Photon_electronVeto"], np.bool_)
    bitmap = arrays["Photon_vidNestedWPBitmap"]
    fiducial = (abs(p_eta) < 1.4442) | (
        (abs(p_eta) > 1.5660) & (abs(p_eta) < 2.5)
    )
    external = (p_pt > 220.0) & fiducial & p_electron_veto
    kinematic_medium = (vid_level(bitmap, "pt") >= 2) & (
        vid_level(bitmap, "eta") >= 2
    )
    shape_level = vid_level(bitmap, "sieie")
    charged_level = vid_level(bitmap, "charged_iso")
    hoe_level = vid_level(bitmap, "hoe")
    ecal_level = vid_level(bitmap, "ecal_iso")
    hcal_level = vid_level(bitmap, "hcal_iso")
    base = external & kinematic_medium
    other_medium = (
        (hoe_level >= 2) & (ecal_level >= 2) & (hcal_level >= 2)
    )
    # The ABCD sidebands deliberately fail the *loose* working point.  Values
    # at level 1 (pass loose, fail medium) form a guard band and are not used.
    # This follows the established CMS photon-like-jet construction and avoids
    # treating candidates just below medium as a hadronic-fake template.
    shape_fail_loose = shape_level < 1
    charged_fail_loose = charged_level < 1
    plj_other = base & (shape_level >= 2) & (charged_level >= 2) & (
        (
            (hoe_level < 1)
            & (ecal_level >= 2)
            & (hcal_level >= 2)
        )
        | (
            (hoe_level >= 2)
            & (ecal_level < 1)
            & (hcal_level >= 2)
        )
        | (
            (hoe_level >= 2)
            & (ecal_level >= 2)
            & (hcal_level < 1)
        )
    )
    return {
        "target": (
            base
            & other_medium
            & (shape_level >= 2)
            & (charged_level >= 2)
        ),
        "measurement_pass": (
            base
            & other_medium
            & shape_fail_loose
            & (charged_level >= 2)
        ),
        "measurement_fail": (
            base
            & other_medium
            & shape_fail_loose
            & charged_fail_loose
        ),
        "application": (
            base
            & other_medium
            & (shape_level >= 2)
            & charged_fail_loose
        ),
        "plj_other": plj_other,
    }


def transfer_stratum(probe_pt: float, probe_eta: float) -> int:
    eta_index = 0 if abs(float(probe_eta)) < 1.4442 else 1
    pt_index = int(
        np.searchsorted(
            np.asarray(TRANSFER_PT_EDGES, dtype=float),
            float(probe_pt),
            side="right",
        )
        - 1
    )
    pt_index = max(0, min(pt_index, len(TRANSFER_PT_EDGES) - 2))
    return eta_index * (len(TRANSFER_PT_EDGES) - 1) + pt_index


def transfer_labels() -> list[str]:
    labels: list[str] = []
    for eta in TRANSFER_ETA_LABELS:
        for low, high in zip(TRANSFER_PT_EDGES[:-1], TRANSFER_PT_EDGES[1:]):
            upper = "inf" if high >= 1_000_000.0 else f"{high:g}"
            labels.append(f"{eta}_pt{low:g}to{upper}")
    return labels


def photon_origin(gen_part_flavour: int) -> str:
    flavour = abs(int(gen_part_flavour))
    if flavour == 1:
        return "prompt"
    if flavour == 11:
        return "electron"
    return "fake"


def _empty_leaf(edges: list[float]) -> dict[str, Any]:
    bins = max(0, len(edges) - 1)
    return {
        "bin_edges": [float(value) for value in edges],
        "sumw": [0.0] * bins,
        "sumw2": [0.0] * bins,
        "entries": [0] * bins,
    }


def _empty_stratified(edges: list[float]) -> dict[str, Any]:
    return {
        "transfer_labels": transfer_labels(),
        "strata": [_empty_leaf(edges) for _ in transfer_labels()],
    }


def _add_value(
    target: dict[str, Any],
    stratum: int,
    value: float,
    weight: float,
) -> None:
    if not np.isfinite(value) or not np.isfinite(weight):
        return
    leaf = target["strata"][int(stratum)]
    edges = np.asarray(leaf["bin_edges"], dtype=float)
    index = int(np.searchsorted(edges, float(value), side="right") - 1)
    if index < 0 or index >= len(edges) - 1:
        return
    leaf["sumw"][index] += float(weight)
    leaf["sumw2"][index] += float(weight) * float(weight)
    leaf["entries"][index] += 1


def _dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_dataset": physical_dataset(str(record.get("dataset") or "unknown")),
        "process": str(record.get("process_group") or "unknown"),
        "xsec_pb": record.get("xsec_pb"),
        "dataset_splits": [],
        "files_processed": 0,
        "events_read": 0,
        "selected_events": 0,
        "channels": {},
    }


def _register_dataset(
    datasets: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "unknown")
    key = physical_dataset(dataset)
    target = datasets.setdefault(key, _dataset_record(record))
    if target["process"] != str(record.get("process_group") or "unknown"):
        raise RuntimeError(f"process mismatch for physical dataset {key}")
    old_xsec = target.get("xsec_pb")
    new_xsec = record.get("xsec_pb")
    if (
        old_xsec is not None
        and new_xsec is not None
        and abs(float(old_xsec) - float(new_xsec)) > 1.0e-12
    ):
        raise RuntimeError(f"cross-section mismatch for physical dataset {key}")
    if dataset not in target["dataset_splits"]:
        target["dataset_splits"].append(dataset)
    return target


def _channel_origin_record(
    dataset_record: dict[str, Any],
    probe: str,
    origin: str,
    builder: Any,
) -> dict[str, Any]:
    target = (
        dataset_record["channels"]
        .setdefault(probe, {})
        .setdefault(
            origin,
            {
                "transfer": _empty_stratified([0.0, 1.0]),
                "region_transfers": {
                    region: _empty_stratified([0.0, 1.0])
                    for region in GCR_REGIONS
                },
                "distributions": {},
            },
        )
    )
    distributions = target["distributions"]
    for region in GCR_REGIONS:
        region_record = distributions.setdefault(region, {})
        region_record.setdefault(
            "recoil",
            _empty_stratified([float(x) for x in builder.RECOIL_PT_BINS]),
        )
        if region == "GCR":
            for variable, spec in builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items():
                region_record.setdefault(
                    variable,
                    _empty_stratified([float(x) for x in spec["bins"]]),
                )
    return target


def _row_value(row: dict[str, Any], spec: dict[str, Any]) -> float:
    branch = (spec.get("branch_by_region") or {}).get("GCR", spec["branch"])
    source = spec.get("source", "scalar")
    if source == "masked_first":
        mask_branch = str(spec["mask_branch"])
        for value, keep in zip(row.get(branch) or [], row.get(mask_branch) or []):
            if bool(keep):
                return float(value)
        return float(spec.get("fill", -99.0))
    value = row.get(branch, spec.get("fill", -99.0))
    if isinstance(value, list):
        if not value:
            return float(spec.get("fill", -99.0))
        return float(value[0])
    return float(value)


def compact_data_event(
    row: dict[str, Any],
    probe: str,
    source_dataset: str,
    builder: Any,
) -> dict[str, Any]:
    values = {
        "recoil": float(row["recoil_gcr"]),
        **{
            variable: _row_value(row, spec)
            for variable, spec in builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items()
        },
    }
    return {
        "run": int(row["run"]),
        "luminosityBlock": int(row["luminosityBlock"]),
        "event": int(row["event"]),
        "source_dataset": source_dataset,
        "probe": probe,
        "transfer_stratum": int(row["transfer_stratum"]),
        "nboosted_top": int(row["nboosted_top"]),
        "regions": row_regions(row),
        "values": values,
    }


def row_regions(row: dict[str, Any]) -> list[str]:
    """Return mutually exclusive nominal/validation GCR memberships."""
    if bool(row.get("pass_gcr_open_high")):
        return [
            "GCR",
            (
                "GCR_Nt0"
                if int(row.get("nboosted_top") or 0) == 0
                else "GCR_Nt1"
            ),
        ]
    minimum = float(row.get("gcr_min_recoil_dphi4", -1.0))
    if 0.30 <= minimum < 0.50:
        return ["GCR_DPhiVR_High"]
    if 0.10 <= minimum < 0.30:
        return ["GCR_DPhiVR_Low"]
    return []


def fill_mc_row(
    dataset_record: dict[str, Any],
    row: dict[str, Any],
    probe: str,
    builder: Any,
) -> None:
    origin = str(row["probe_origin"])
    stratum = int(row["transfer_stratum"])
    weight = float(row["nominal_weight"])
    regions = row_regions(row)
    for origin_key in ("all", origin):
        target = _channel_origin_record(
            dataset_record,
            probe,
            origin_key,
            builder,
        )
        if "GCR" in regions:
            _add_value(target["transfer"], stratum, 0.5, weight)
        for region in regions:
            _add_value(
                target["region_transfers"][region],
                stratum,
                0.5,
                weight,
            )
            region_target = target["distributions"][region]
            _add_value(
                region_target["recoil"],
                stratum,
                float(row["recoil_gcr"]),
                weight,
            )
            if region == "GCR":
                for variable, spec in builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS.items():
                    _add_value(
                        region_target[variable],
                        stratum,
                        _row_value(row, spec),
                        weight,
                    )


def _selected_probe_values(
    arrays: Any,
    mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_pt = ak.fill_none(ak.firsts(arrays["Photon_pt"][mask]), np.nan)
    selected_eta = ak.fill_none(ak.firsts(arrays["Photon_eta"][mask]), np.nan)
    if "Photon_genPartFlav" in set(arrays.fields):
        selected_flavour = ak.fill_none(
            ak.firsts(arrays["Photon_genPartFlav"][mask]),
            0,
        )
    else:
        selected_flavour = ak.zeros_like(selected_pt)
    return (
        np.asarray(ak.to_numpy(selected_pt), dtype=float),
        np.asarray(ak.to_numpy(selected_eta), dtype=float),
        np.asarray(ak.to_numpy(selected_flavour), dtype=int),
    )


def _probe_assignment(arrays: Any) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    masks = probe_masks(arrays)
    target_mask = masks["target"]
    target_count = ak.sum(target_mask, axis=1)
    sideband_mask = masks["measurement_pass"]
    for probe in ("measurement_fail", "application", "plj_other"):
        sideband_mask = sideband_mask | masks[probe]

    # Match the trusted nominal GCR target definition exactly: one medium target
    # photon is sufficient even when additional anti-ID candidates exist, while
    # events with multiple medium target photons remain rejected.  Preserve the
    # established event-level sideband definition: in the absence of a target,
    # exactly one pass/fail/application candidate is required.  Selecting a
    # leading candidate from a multi-probe sideband event would change the A/P/F
    # regions and requires a separate combinatorial fake-rate prescription.
    sideband_count = ak.sum(sideband_mask, axis=1)
    selected_target = target_mask & (target_count == 1)
    selected_sideband = (
        sideband_mask
        & (target_count == 0)
        & (sideband_count == 1)
    )
    selected_mask = selected_target | selected_sideband

    probe_codes = np.full(len(arrays), -1, dtype=int)
    for code, probe in enumerate(PROBE_KINDS):
        has_probe = np.asarray(
            ak.to_numpy(ak.any(selected_mask & masks[probe], axis=1)),
            dtype=bool,
        )
        probe_codes[has_probe] = code
    has_selected_probe = np.asarray(
        ak.to_numpy(ak.sum(selected_mask, axis=1) == 1),
        dtype=bool,
    )
    return masks, selected_mask, probe_codes, has_selected_probe


def _prefilter_read_branches(present: set[str]) -> list[str]:
    branches = set(PREFILTER_BRANCHES)
    branches.update(set(baseline.PHOTON_HLT) & present)
    branches.update(set(baseline.FILTERS) & present)
    if "nJet" in present:
        branches.add("nJet")
    else:
        branches.add("Jet_pt")
    return sorted(branches)


def _necessary_gcr_event_mask(
    arrays: Any,
    photon_candidate: np.ndarray,
    is_data: bool,
    process: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply only exact necessary GCR conditions before the trusted selection."""
    n = len(arrays)
    photon_trigger = baseline.bool_branch(arrays, list(baseline.PHOTON_HLT), n)
    met_filters, _ = baseline.all_filters(arrays, n)
    if is_data:
        lumi_mask, _ = baseline.golden_lumi_mask(
            arrays,
            process,
            Path.cwd(),
            n,
            "2024",
        )
    else:
        lumi_mask = np.ones(n, dtype=bool)
    if "nJet" in set(arrays.fields):
        enough_raw_jets = np.asarray(arrays["nJet"], dtype=int) >= 5
    else:
        enough_raw_jets = np.asarray(
            ak.to_numpy(ak.num(arrays["Jet_pt"], axis=1)),
            dtype=int,
        ) >= 5
    has_medium_btag_score = np.asarray(
        ak.to_numpy(
            ak.any(
                arrays["Jet_btagUParTAK4B"] > baseline.UPART_AK4_MEDIUM_WP,
                axis=1,
            )
        ),
        dtype=bool,
    )
    stages = (
        ("photon_nminus1_exactly_one", photon_candidate),
        ("photon_trigger", photon_trigger),
        ("met_filters", met_filters),
        ("golden_lumi", lumi_mask),
        ("raw_njet_at_least_five", enough_raw_jets),
        ("any_medium_btag_score", has_medium_btag_score),
    )
    keep = np.ones(n, dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in stages:
        keep &= np.asarray(mask, dtype=bool)
        counts[name] = int(np.count_nonzero(keep))
    return keep, counts


def _open_root_for_record(file_path: str, timeout: int) -> tuple[Any, dict[str, Any]]:
    """Serialize first population of a shared xrdcp cache, then release the lock."""
    keep_shared_cache = (
        os.environ.get("AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE", "0") == "1"
        and str(file_path).startswith("root://")
    )
    if not keep_shared_cache:
        return baseline.open_root_with_xrd_fallback(file_path, timeout=timeout)
    cache_path = baseline._xrd_cache_path(file_path)
    lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        return baseline.open_root_with_xrd_fallback(file_path, timeout=timeout)


@contextlib.contextmanager
def suppress_full_medium_photon_id_sf() -> Iterator[None]:
    """Keep all nominal MC weights except the inapplicable full-medium photon SF."""
    original = baseline.compute_weight_bundle

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if "gcr_mask" in kwargs:
            modified_kwargs = dict(kwargs)
            modified_kwargs["gcr_mask"] = np.zeros_like(
                np.asarray(kwargs["gcr_mask"], dtype=bool)
            )
            return original(*args, **modified_kwargs)
        modified_args = list(args)
        modified_args[-1] = np.zeros_like(
            np.asarray(modified_args[-1], dtype=bool)
        )
        return original(*modified_args)

    baseline.compute_weight_bundle = wrapped
    try:
        yield
    finally:
        baseline.compute_weight_bundle = original


def _modified_cutbased(corrected: Any, mask: Any) -> Any:
    selected = ak.values_astype(ak.where(mask, 2, 0), np.int32)
    return ak.with_field(corrected, selected, "Photon_cutBased")


def _decorate_probe_rows(
    rows: list[dict[str, Any]],
    probe_pt: np.ndarray,
    probe_eta: np.ndarray,
    probe_flavour: np.ndarray,
    probe_codes: np.ndarray,
    entry_start: int,
    is_data: bool,
) -> None:
    for row in rows:
        local = int(row["entry"]) - int(entry_start)
        pt = float(probe_pt[local])
        eta = float(probe_eta[local])
        flavour = 0 if is_data else int(probe_flavour[local])
        row["probe_pt"] = pt
        row["probe_eta"] = eta
        row["probe_gen_part_flavour"] = flavour
        row["probe_origin"] = "data" if is_data else photon_origin(flavour)
        row["transfer_stratum"] = transfer_stratum(pt, eta)
        code = int(probe_codes[local])
        if code < 0 or code >= len(PROBE_KINDS):
            raise RuntimeError(
                f"selected N-1 photon has invalid probe category code {code}"
            )
        row["probe_kind"] = PROBE_KINDS[code]


def process_record(
    record: dict[str, Any],
    builder: Any,
    payload_status: dict[str, Any],
    datasets: dict[str, Any],
    data_events: list[dict[str, Any]],
    summary: dict[str, Any],
    chunk_size: int,
    prefilter_block_size: int,
    max_chunks_per_file: int | None,
) -> bool:
    dataset = str(record.get("dataset") or "unknown")
    process = str(record.get("process_group") or "unknown")
    file_path = str(record.get("file_path") or "")
    is_data = bool(record.get("is_data"))
    if is_data and process != "EGamma":
        raise RuntimeError(f"photon fake data must come from EGamma, got {process}")
    if not is_data and not bool(record.get("is_background")):
        raise RuntimeError(f"photon fake MC must be background MC: {dataset}")
    dataset_record = None if is_data else _register_dataset(datasets, record)
    root_file = None
    access_info: dict[str, Any] = {}
    start_time = time.time()
    try:
        root_file, access_info = _open_root_for_record(
            file_path,
            timeout=60,
        )
        keys = flat.split_keys(root_file)
        if "Events" not in keys:
            raise RuntimeError("Events tree missing")
        tree = root_file["Events"]
        present = set(tree.keys())
        if "Photon_vidNestedWPBitmap" not in present:
            raise RuntimeError("Photon_vidNestedWPBitmap branch missing")
        if not is_data and "Photon_genPartFlav" not in present:
            raise RuntimeError("Photon_genPartFlav branch missing in MC")
        required_prefilter = set(PREFILTER_BRANCHES)
        if "nJet" not in present:
            required_prefilter.add("Jet_pt")
        missing_prefilter = sorted(required_prefilter - present)
        if missing_prefilter:
            raise RuntimeError(
                f"photon prefilter branches missing: {missing_prefilter}"
            )
        audit = branch_audit(present, is_data=is_data)
        if audit.get("status") != "valid":
            raise RuntimeError(
                f"required 2024 object branches missing: {audit.get('missing_required')}"
            )
        branches = sorted(
            set(required_read_branches(tree))
            | (set(EXTRA_BRANCHES) & present)
        )
        prefilter_branches = _prefilter_read_branches(present)
        assigned_start, assigned_stop = record_entry_bounds(
            record,
            int(tree.num_entries),
        )
        events_read = 0
        selected_events = 0
        chunks_seen = 0
        mismatch_objects = 0
        photon_candidate_events = 0
        candidate_events = 0
        fully_evaluated_events = 0
        evaluated_blocks = 0
        event_prefilter_cutflow: dict[str, int] = {}
        for entry_start in range(assigned_start, assigned_stop, chunk_size):
            if max_chunks_per_file is not None and chunks_seen >= max_chunks_per_file:
                break
            entry_stop = min(entry_start + chunk_size, assigned_stop)
            prefilter = tree.arrays(
                prefilter_branches,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="ak",
            )
            events_read += len(prefilter)
            prefilter_masks, _, _, photon_candidate = _probe_assignment(
                prefilter
            )
            nominal = baseline.medium_photon_mask(
                prefilter["Photon_pt"],
                prefilter["Photon_eta"],
                prefilter["Photon_cutBased"],
                prefilter["Photon_electronVeto"],
            )
            mismatch_objects += int(
                ak.sum(
                    ak.values_astype(
                        prefilter_masks["target"] != nominal,
                        np.int64,
                    )
                )
            )
            photon_candidate_events += int(np.count_nonzero(photon_candidate))
            candidate, cutflow = _necessary_gcr_event_mask(
                prefilter,
                photon_candidate,
                is_data,
                process,
            )
            for name, count in cutflow.items():
                event_prefilter_cutflow[name] = (
                    event_prefilter_cutflow.get(name, 0) + count
                )
            candidate_events += int(np.count_nonzero(candidate))
            candidate_offsets = np.flatnonzero(candidate)
            block_offsets = np.unique(
                (candidate_offsets // prefilter_block_size) * prefilter_block_size
            )
            for block_offset in block_offsets:
                block_start = entry_start + int(block_offset)
                block_stop = min(
                    block_start + prefilter_block_size,
                    entry_stop,
                )
                arrays = tree.arrays(
                    branches,
                    entry_start=block_start,
                    entry_stop=block_stop,
                    library="ak",
                )
                fully_evaluated_events += len(arrays)
                evaluated_blocks += 1
                _, union_mask, probe_codes, full_candidate = _probe_assignment(
                    arrays
                )
                expected_candidate = photon_candidate[
                    int(block_offset) : int(block_offset) + len(arrays)
                ]
                if not np.array_equal(full_candidate, expected_candidate):
                    raise RuntimeError(
                        "minimal-branch and full-branch photon prefilter disagree"
                    )
                probe_pt, probe_eta, probe_flavour = _selected_probe_values(
                    arrays,
                    union_mask,
                )
                corrected, calibration = calibrate_jets_and_met(
                    arrays,
                    is_data=is_data,
                    shift="nominal",
                    root=Path.cwd(),
                )
                validation_context = {
                    "object_branch_audit": audit,
                    "payload_status": payload_status,
                }
                modified = _modified_cutbased(corrected, union_mask)
                with suppress_full_medium_photon_id_sf():
                    rows, chunk_summary = intermediate.extract_chunk_2024(
                        arrays,
                        dataset,
                        process,
                        None,
                        str(record.get("year") or "2024"),
                        file_path,
                        block_start,
                        block_stop,
                        fastsim_trigger_bypass=False,
                        shift_name="nominal",
                        compute_weights=True,
                        decorate_rows=False,
                        materialize_skim_flag="feature_GCR_fake_validation",
                        precalibrated=(modified, calibration),
                        validation_context=validation_context,
                    )
                _decorate_probe_rows(
                    rows,
                    probe_pt,
                    probe_eta,
                    probe_flavour,
                    probe_codes,
                    block_start,
                    is_data,
                )
                rows = [row for row in rows if row_regions(row)]
                selected_events += len(rows)
                for row in rows:
                    probe = str(row["probe_kind"])
                    summary["probe_counts"][probe] += 1
                    if is_data:
                        data_events.append(
                            compact_data_event(row, probe, dataset, builder)
                        )
                    else:
                        assert dataset_record is not None
                        fill_mc_row(dataset_record, row, probe, builder)
                btag = (
                    (
                        (chunk_summary.get("scale_factor_status") or {}).get(
                            "components"
                        )
                        or {}
                    ).get("btagSF")
                    or {}
                )
                if not is_data and not btag.get("applied"):
                    raise RuntimeError(f"btagSF unavailable for {dataset}: {btag}")
                if not is_data:
                    summary.setdefault("btag_sf_status", {}).setdefault(
                        process,
                        btag,
                    )
            chunks_seen += 1
        if max_chunks_per_file is None and events_read != assigned_stop - assigned_start:
            raise RuntimeError(
                f"processed Events entries differ from assigned range: "
                f"{events_read} vs {assigned_stop - assigned_start}"
            )
        if dataset_record is not None:
            dataset_record["files_processed"] += 1
            dataset_record["events_read"] += events_read
            dataset_record["selected_events"] += selected_events
        summary["files_processed"] += 1
        summary["events_read"] += events_read
        summary["selected_events"] += selected_events
        summary["target_cutbased_mismatch_objects"] += mismatch_objects
        summary["photon_prefilter_candidate_events"] += photon_candidate_events
        summary["prefilter_candidate_events"] += candidate_events
        summary["fully_evaluated_events"] += fully_evaluated_events
        summary["prefilter_blocks_evaluated"] += evaluated_blocks
        summary.setdefault("file_records", []).append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "entry_start": assigned_start,
                "entry_stop": assigned_stop,
                "events_read": events_read,
                "photon_prefilter_candidate_events": photon_candidate_events,
                "prefilter_candidate_events": candidate_events,
                "event_prefilter_cutflow": event_prefilter_cutflow,
                "fully_evaluated_events": fully_evaluated_events,
                "prefilter_blocks_evaluated": evaluated_blocks,
                "selected_events": selected_events,
                "target_cutbased_mismatch_objects": mismatch_objects,
                "wall_time_s": round(time.time() - start_time, 3),
                "access": access_info,
                "status": "complete",
            }
        )
        return True
    except Exception as exc:
        if isinstance(exc, baseline.RootOpenFailure):
            access_info = dict(exc.access_info)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary.setdefault("bad_files", []).append(
            {
                "dataset": dataset,
                "file_path": file_path,
                "failure_stage": "photon_fake_open_read_or_evaluate",
                "exception_type": type(exc).__name__,
                "concise_error": str(exc)[:500],
                "first_failure_time": now,
                "last_failure_time": now,
                "alternate_access_attempted": bool(
                    access_info.get("alternate_access_attempted")
                ),
                "direct_open_error": access_info.get("direct_open_error"),
                "fallback_status": access_info.get("fallback_status"),
                "xrdcp_exit_status": access_info.get("xrdcp_exit_status"),
                "xrdcp_stderr_tail": str(
                    access_info.get("xrdcp_stderr_tail") or ""
                )[-1000:],
                "xrdcp_attempts": access_info.get("xrdcp_attempts") or [],
                "permanently_skipped": False,
            }
        )
        return False
    finally:
        try:
            if root_file is not None:
                root_file.close()
        except Exception:
            pass
        baseline.cleanup_xrd_cache(access_info)


def _merge_stratified(target: dict[str, Any], source: dict[str, Any]) -> None:
    if target["transfer_labels"] != source["transfer_labels"]:
        raise RuntimeError("transfer-label mismatch while merging record results")
    if len(target["strata"]) != len(source["strata"]):
        raise RuntimeError("transfer-stratum count mismatch while merging record results")
    for target_leaf, source_leaf in zip(target["strata"], source["strata"]):
        if target_leaf["bin_edges"] != source_leaf["bin_edges"]:
            raise RuntimeError("histogram bin-edge mismatch while merging record results")
        for field in ("sumw", "sumw2", "entries"):
            if len(target_leaf[field]) != len(source_leaf[field]):
                raise RuntimeError(
                    f"histogram {field} length mismatch while merging record results"
                )
            target_leaf[field] = [
                left + right
                for left, right in zip(target_leaf[field], source_leaf[field])
            ]


def _merge_channel(target: dict[str, Any], source: dict[str, Any]) -> None:
    _merge_stratified(target["transfer"], source["transfer"])
    for region, transfer in (source.get("region_transfers") or {}).items():
        target_transfers = target.setdefault("region_transfers", {})
        if region not in target_transfers:
            target_transfers[region] = copy.deepcopy(transfer)
        else:
            _merge_stratified(target_transfers[region], transfer)
    for region, variables in source["distributions"].items():
        target_region = target["distributions"].setdefault(region, {})
        for variable, histogram in variables.items():
            if variable not in target_region:
                target_region[variable] = copy.deepcopy(histogram)
            else:
                _merge_stratified(target_region[variable], histogram)


def _merge_datasets(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, dataset in source.items():
        if key not in target:
            target[key] = copy.deepcopy(dataset)
            continue
        merged = target[key]
        if merged["process"] != dataset["process"]:
            raise RuntimeError(f"process mismatch while merging physical dataset {key}")
        old_xsec = merged.get("xsec_pb")
        new_xsec = dataset.get("xsec_pb")
        if (
            old_xsec is not None
            and new_xsec is not None
            and abs(float(old_xsec) - float(new_xsec)) > 1.0e-12
        ):
            raise RuntimeError(
                f"cross-section mismatch while merging physical dataset {key}"
            )
        merged["dataset_splits"] = sorted(
            set(merged["dataset_splits"]) | set(dataset["dataset_splits"])
        )
        for field in ("files_processed", "events_read", "selected_events"):
            merged[field] += int(dataset[field])
        for probe, origins in dataset["channels"].items():
            target_origins = merged["channels"].setdefault(probe, {})
            for origin, channel in origins.items():
                if origin not in target_origins:
                    target_origins[origin] = copy.deepcopy(channel)
                else:
                    _merge_channel(target_origins[origin], channel)


def _merge_record_result(
    datasets: dict[str, Any],
    data_events: list[dict[str, Any]],
    summary: dict[str, Any],
    result: dict[str, Any],
) -> None:
    _merge_datasets(datasets, result["datasets"])
    data_events.extend(result["data_events"])
    record_summary = result["summary"]
    for field in (
        "files_processed",
        "events_read",
        "selected_events",
        "target_cutbased_mismatch_objects",
        "photon_prefilter_candidate_events",
        "prefilter_candidate_events",
        "fully_evaluated_events",
        "prefilter_blocks_evaluated",
    ):
        summary[field] += int(record_summary.get(field) or 0)
    for probe in PROBE_KINDS:
        summary["probe_counts"][probe] += int(
            (record_summary.get("probe_counts") or {}).get(probe) or 0
        )
    summary["bad_files"].extend(record_summary.get("bad_files") or [])
    summary.setdefault("file_records", []).extend(
        record_summary.get("file_records") or []
    )
    for process, status in (
        record_summary.get("btag_sf_status") or {}
    ).items():
        summary.setdefault("btag_sf_status", {}).setdefault(process, status)


def _parallel_initializer() -> None:
    global _PARALLEL_BUILDER, _PARALLEL_PAYLOAD_STATUS
    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload_status = validate_payloads(Path.cwd())
    if payload_status.get("status") != "valid":
        raise RuntimeError(
            "2024 correction payload validation failed in record worker: "
            f"{payload_status.get('errors')}"
        )
    _PARALLEL_PAYLOAD_STATUS = payload_status
    _PARALLEL_BUILDER = load_histogram_builder()


def _process_record_isolated(
    task: tuple[dict[str, Any], int, int, int | None],
) -> dict[str, Any]:
    record, chunk_size, prefilter_block_size, max_chunks_per_file = task
    if _PARALLEL_BUILDER is None or _PARALLEL_PAYLOAD_STATUS is None:
        raise RuntimeError("parallel photon-fake record worker was not initialized")
    child_summary: dict[str, Any] = {
        "files_attempted": 1,
        "files_processed": 0,
        "events_read": 0,
        "selected_events": 0,
        "photon_prefilter_candidate_events": 0,
        "prefilter_candidate_events": 0,
        "fully_evaluated_events": 0,
        "prefilter_blocks_evaluated": 0,
        "probe_counts": {probe: 0 for probe in PROBE_KINDS},
        "target_cutbased_mismatch_objects": 0,
        "bad_files": [],
    }
    child_datasets: dict[str, Any] = {}
    child_data_events: list[dict[str, Any]] = []
    process_record(
        record,
        _PARALLEL_BUILDER,
        _PARALLEL_PAYLOAD_STATUS,
        child_datasets,
        child_data_events,
        child_summary,
        chunk_size,
        prefilter_block_size,
        max_chunks_per_file,
    )
    return {
        "datasets": child_datasets,
        "data_events": child_data_events,
        "summary": child_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build non-destructive 2024 photon-fake sideband histograms."
    )
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--prefilter-block-size", type=int, default=512)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-chunks-per-file", type=int)
    parser.add_argument("--record-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.prefilter_block_size <= 0:
        parser.error("--prefilter-block-size must be positive")
    if args.prefilter_block_size > args.chunk_size:
        parser.error("--prefilter-block-size cannot exceed --chunk-size")
    if args.record_workers <= 0:
        parser.error("--record-workers must be positive")

    intermediate.install_backend()
    os.environ["AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA"] = "0"
    payload_status = validate_payloads(Path.cwd())
    if payload_status.get("status") != "valid":
        raise RuntimeError(
            f"2024 correction payload validation failed: {payload_status.get('errors')}"
        )
    builder = load_histogram_builder()
    shard_path = Path(args.shard)
    shard = read_json(shard_path)
    records = list(shard.get("records") or [])
    if args.max_records is not None:
        records = records[: max(0, args.max_records)]
    if not records:
        raise RuntimeError("photon fake shard has no records")

    start = time.time()
    summary: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_shard": str(shard_path),
        "source_record_digest": shard.get("record_digest"),
        "files_attempted": len(records),
        "record_workers": min(args.record_workers, len(records)),
        "files_processed": 0,
        "events_read": 0,
        "selected_events": 0,
        "photon_prefilter_candidate_events": 0,
        "prefilter_candidate_events": 0,
        "fully_evaluated_events": 0,
        "prefilter_blocks_evaluated": 0,
        "probe_counts": {probe: 0 for probe in PROBE_KINDS},
        "target_cutbased_mismatch_objects": 0,
        "bad_files": [],
        "payload_status": payload_status,
        "selection_source": "real_subset_worker.py via intermediate_2024_worker.py",
        "photon_sf_policy": (
            "full-medium photon ID SF suppressed in target and sidebands; "
            "all other nominal central MC weights retained"
        ),
        "prefilter_policy": {
            "base_branches": list(PREFILTER_BRANCHES),
            "candidate": (
                "exactly one photon in the four-way N-1 union plus exact "
                "necessary GCR trigger/filter/lumi/raw-Njet/btag-score conditions"
            ),
            "full_evaluation_block_size": args.prefilter_block_size,
            "physics_selection_after_prefilter": (
                "unchanged real_subset_worker.py selection"
            ),
        },
    }
    datasets: dict[str, Any] = {}
    data_events: list[dict[str, Any]] = []
    if args.record_workers == 1 or len(records) == 1:
        for record in records:
            process_record(
                record,
                builder,
                payload_status,
                datasets,
                data_events,
                summary,
                args.chunk_size,
                args.prefilter_block_size,
                args.max_chunks_per_file,
            )
    else:
        tasks = [
            (
                record,
                args.chunk_size,
                args.prefilter_block_size,
                args.max_chunks_per_file,
            )
            for record in records
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.record_workers, len(records)),
            initializer=_parallel_initializer,
        ) as executor:
            for result in executor.map(_process_record_isolated, tasks):
                _merge_record_result(datasets, data_events, summary, result)

    if summary["files_processed"] == summary["files_attempted"]:
        status = "complete"
    elif summary["files_processed"] > 0:
        status = "complete_with_bad_files"
    else:
        status = "failed"
    summary["status"] = status
    summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["wall_time_s"] = round(time.time() - start, 3)
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": status,
        "year": 2024,
        "scope": "high-dM photon control region only",
        "probe_definitions": {
            "common": (
                "pT>220, fiducial eta, electronVeto, and medium pT/eta/HoverE/"
                "ECAL-iso/HCAL-iso VID components; exactly one photon passing "
                "this N-1 union is required by the trusted GCR worker"
            ),
            "target": "medium sieie and medium charged isolation",
            "measurement_pass": "failed medium sieie and medium charged isolation",
            "measurement_fail": "failed medium sieie and failed medium charged isolation",
            "application": "medium sieie and failed medium charged isolation",
            "vid_bitmap_encoding": "two bits per cut: fail=0, loose=1, medium=2, tight=3",
        },
        "transfer_factor": {
            "eta_labels": list(TRANSFER_ETA_LABELS),
            "pt_edges": list(TRANSFER_PT_EDGES),
            "labels": transfer_labels(),
        },
        "regions": list(GCR_REGIONS),
        "recoil_pt_bins": [float(x) for x in builder.RECOIL_PT_BINS],
        "highdm_distribution_variable_specs": (
            builder.HIGHDM_DISTRIBUTION_VARIABLE_SPECS
        ),
        "summary": summary,
        "datasets": datasets,
        "data_events": data_events,
    }
    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    write_json_gz(output, payload)
    metadata = {
        "schema_version": f"{OUTPUT_SCHEMA}_metadata",
        "status": status,
        "histogram_file": str(output),
        "histogram_size": output.stat().st_size,
        "histogram_sha256": sha256(output),
        "source_shard": str(shard_path),
        "source_record_digest": shard.get("record_digest"),
        "summary": summary,
        "dataset_count": len(datasets),
        "data_event_count": len(data_events),
    }
    write_json(metadata_output, metadata)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "metadata": str(metadata_output),
                "files_processed": summary["files_processed"],
                "files_attempted": summary["files_attempted"],
                "events_read": summary["events_read"],
                "selected_events": summary["selected_events"],
                "data_events": len(data_events),
                "dataset_count": len(datasets),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["files_processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

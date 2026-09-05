from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SIGNAL_PATTERN = re.compile(r"^GenModel_(?P<topology>[^_]+)_(?P<mstop>\d+)_(?P<mlsp>\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit every nominal flat-campaign shard and aggregate MC normalization."
    )
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--luminosity-pb", required=True, type=float)
    opts = parser.parse_args()
    if opts.year < 2022:
        raise ValueError(f"unsupported Run-3 year: {opts.year}")
    if opts.luminosity_pb <= 0.0:
        raise ValueError("--luminosity-pb must be positive")

    source = opts.source_directory
    root_paths = sorted(source.glob("*.root"))
    sidecar_paths = sorted(source.glob("*.json"))
    roots_by_stem = {path.stem: path for path in root_paths}
    sidecars_by_stem = {path.stem: path for path in sidecar_paths}
    missing_sidecars = sorted(set(roots_by_stem) - set(sidecars_by_stem))
    missing_roots = sorted(set(sidecars_by_stem) - set(roots_by_stem))

    physical: dict[str, dict[str, object]] = {}
    signal_sumw: defaultdict[str, float] = defaultdict(float)
    signal_event_sumw: defaultdict[str, float] = defaultdict(float)
    shards: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    source_bad_files: list[dict[str, object]] = []
    process_shards: Counter[str] = Counter()
    process_written: Counter[str] = Counter()
    kind_shards: Counter[str] = Counter()
    kind_bytes: Counter[str] = Counter()
    kind_written: Counter[str] = Counter()
    input_file_occurrences: Counter[str] = Counter()
    input_file_ids: dict[str, set[int]] = defaultdict(set)
    input_file_shards: defaultdict[str, list[str]] = defaultdict(list)

    for stem in sorted(set(roots_by_stem) & set(sidecars_by_stem)):
        root_path = roots_by_stem[stem]
        sidecar_path = sidecars_by_stem[stem]
        kind = stem.split("_shard_", 1)[0]
        try:
            payload = json.loads(sidecar_path.read_text())
        except Exception as error:
            invalid.append(
                {
                    "stem": stem,
                    "stage": "sidecar_read",
                    "exception_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        if payload.get("status") not in {"complete", "complete_with_bad_files"}:
            invalid.append(
                {"stem": stem, "stage": "sidecar_status", "status": payload.get("status")}
            )
            continue
        for bad_file in payload.get("bad_files") or []:
            source_bad_files.append({"stem": stem, **bad_file})
        datasets = payload.get("datasets") or {}
        for file_record in payload.get("files") or []:
            file_path = str(file_record.get("file_path") or "")
            if not file_path:
                invalid.append(
                    {"stem": stem, "stage": "input_file_audit", "error": "missing file_path"}
                )
                continue
            input_file_occurrences[file_path] += 1
            input_file_shards[file_path].append(stem)
            if file_record.get("file_id") is not None:
                input_file_ids[file_path].add(int(file_record["file_id"]))
        processes = sorted({str(record.get("process")) for record in datasets.values()})
        physical_ids = sorted(
            {int(record["physical_dataset_id"]) for record in datasets.values()}
        )
        for record in datasets.values():
            process = str(record.get("process"))
            process_shards[process] += 1
            process_written[process] += int(record.get("events_written", 0))
            physical_id = str(int(record["physical_dataset_id"]))
            if not record.get("is_data") and not record.get("is_signal"):
                current = physical.setdefault(
                    physical_id,
                    {
                        "physical_dataset_id": int(physical_id),
                        "physical_dataset": record["physical_dataset"],
                        "process": process,
                        "xsec_pb": float(record["xsec_pb"]),
                        "sumw": 0.0,
                        "files_processed": 0,
                        "split_dataset_ids": [],
                    },
                )
                if current["process"] != process or current["physical_dataset"] != record["physical_dataset"]:
                    raise RuntimeError(f"physical dataset identity conflict for {physical_id}")
                if abs(float(current["xsec_pb"]) - float(record["xsec_pb"])) > 1.0e-12:
                    raise RuntimeError(f"xsec conflict for {physical_id}")
                current["sumw"] = float(current["sumw"]) + float(record.get("sumw", 0.0))
                current["files_processed"] = int(current["files_processed"]) + int(record.get("files_processed", 0))
                current["split_dataset_ids"].append(int(record["dataset_id"]))
            if record.get("is_signal"):
                for name, value in (record.get("signal_sumw_by_genmodel") or {}).items():
                    signal_sumw[name] += float(value)
                for name, value in (record.get("signal_event_genweight_sum_by_genmodel") or {}).items():
                    signal_event_sumw[name] += float(value)

        events_written = int(payload.get("events_written", 0))
        size = int(root_path.stat().st_size)
        kind_shards[kind] += 1
        kind_bytes[kind] += size
        kind_written[kind] += events_written
        shards.append(
            {
                "stem": stem,
                "kind": kind,
                "root": str(root_path),
                "sidecar": str(sidecar_path),
                "root_size_bytes": size,
                "root_sha256_recorded": payload.get("root_sha256"),
                "events_read": int(payload.get("events_read", 0)),
                "events_written": events_written,
                "processes": processes,
                "physical_dataset_ids": physical_ids,
                "bad_files_recorded": len(payload.get("bad_files") or []),
            }
        )

    mass_points: list[dict[str, object]] = []
    for name in sorted(signal_sumw):
        match = SIGNAL_PATTERN.match(name)
        if match is None:
            invalid.append({"stage": "signal_name_parse", "genmodel": name})
            continue
        mass_points.append(
            {
                "genmodel": name,
                "topology": match.group("topology"),
                "mStop": int(match.group("mstop")),
                "mLSP": int(match.group("mlsp")),
                "deltaM": int(match.group("mstop")) - int(match.group("mlsp")),
                "sumw": signal_sumw[name],
                "event_genweight_sum": signal_event_sumw.get(name, 0.0),
            }
        )

    # A split dataset normally spans several output shards, so repeated
    # dataset_id values are expected.  A repeated *input file path*, however,
    # would double count events and generator sums and is a real campaign error.
    duplicate_input_files = {
        path: {
            "occurrences": input_file_occurrences[path],
            "file_ids": sorted(input_file_ids[path]),
            "shards": input_file_shards[path],
        }
        for path in sorted(input_file_occurrences)
        if input_file_occurrences[path] > 1
    }
    inconsistent_input_file_ids = {
        path: sorted(ids) for path, ids in input_file_ids.items() if len(ids) > 1
    }
    for record in physical.values():
        record["split_dataset_ids"] = sorted(set(record["split_dataset_ids"]))
        record["normalization_complete"] = bool(record["sumw"] != 0.0)

    status = "complete_with_permanent_skips" if source_bad_files else "complete"
    if (
        missing_sidecars
        or missing_roots
        or invalid
        or duplicate_input_files
        or inconsistent_input_file_ids
    ):
        status = "incomplete"
    output = {
        "schema_version": "gnn_lowdm_full_campaign_manifest_v1",
        "status": status,
        "year": int(opts.year),
        "source_directory": str(source),
        "policy": {
            "all_nominal_intermediate_roots_audited": True,
            "collision_data_supervised_training": "forbidden",
            "background_training_kinds": ["mc"],
            "signal_training_kinds": ["signal"],
            "split": {"train": 0.2, "validation": 0.1, "test": 0.7},
        },
        "inventory": {
            "roots": len(root_paths),
            "sidecars": len(sidecar_paths),
            "matched_pairs": len(shards),
            "by_kind": {
                kind: {
                    "shards": kind_shards[kind],
                    "root_bytes": kind_bytes[kind],
                    "events_written": kind_written[kind],
                }
                for kind in sorted(kind_shards)
            },
            "process_dataset_records": dict(sorted(process_shards.items())),
            "process_events_written": dict(sorted(process_written.items())),
            "unique_input_files": len(input_file_occurrences),
        },
        "normalization": {
            "luminosity_pb": float(opts.luminosity_pb),
            "by_physical_dataset_id": dict(sorted(physical.items(), key=lambda item: int(item[0]))),
            "signal_mass_points": mass_points,
        },
        "shards": shards,
        "audit": {
            "missing_sidecars": missing_sidecars,
            "missing_roots": missing_roots,
            "invalid": invalid,
            "duplicate_input_files": duplicate_input_files,
            "inconsistent_input_file_ids": inconsistent_input_file_ids,
            "source_bad_files": source_bad_files,
            "source_bad_file_count": len(source_bad_files),
            "source_bad_file_policy": (
                "Retain each readable intermediate ROOT and normalize MC to the "
                "sumw of successfully processed source files; skipped NanoAOD files "
                "remain explicitly recorded. Data luminosity coverage is incomplete "
                "until the skipped data file is recovered."
            ),
            "repeated_split_dataset_ids_note": (
                "Expected when one logical split dataset is processed across multiple output shards; "
                "input file paths, not dataset IDs, are the uniqueness unit."
            ),
        },
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "roots": len(root_paths),
                "matched_pairs": len(shards),
                "kinds": dict(kind_shards),
                "physical_datasets": len(physical),
                "signal_mass_points": len(mass_points),
                "output": str(opts.output),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare nominal and photon-fake GCR data event keys from one input snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import uproot


UT_BINS = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0],
    dtype=float,
)


def event_key(run: Any, lumi: Any, event: Any) -> tuple[int, int, int]:
    return int(run), int(lumi), int(event)


def physical_data_source(dataset: str) -> str:
    return str(dataset).split("____", 1)[0]


def data_source_rank(dataset: str) -> tuple[int, int, int, str]:
    text = physical_data_source(dataset)
    special_v2 = int("NANOv15_v2" in text)
    version = 0
    if "-v" in text:
        suffix = text.rsplit("-v", 1)[-1]
        if suffix.isdigit():
            version = int(suffix)
    stream_rank = 1 if text.startswith("EGamma0-") else 0
    return special_v2, version, stream_rank, text


def stable_id(text: str) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def dataset_map_from_campaign_manifest(path: Path) -> dict[int, str]:
    manifest = json.loads(path.read_text())
    bundle = Path(str(manifest["local_job_bundle"]["path"]))
    with gzip.open(bundle, "rt", encoding="utf-8") as handle:
        packed = json.load(handle)
    mapping: dict[int, str] = {}
    for job in packed.get("jobs") or []:
        shard = job.get("shard") or {}
        if str(shard.get("process_group") or "") != "EGamma":
            continue
        for record in shard.get("records") or []:
            dataset = str(record["dataset"])
            identifier = stable_id(dataset)
            previous = mapping.get(identifier)
            if previous is not None and previous != dataset:
                raise RuntimeError(
                    f"dataset-id collision: {identifier}: {previous} != {dataset}"
                )
            mapping[identifier] = dataset
    if not mapping:
        raise RuntimeError(f"no EGamma datasets found in {bundle}")
    return mapping


def deduplicate(
    records: list[tuple[tuple[int, int, int], str, float]],
) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    grouped: dict[
        tuple[int, int, int], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for key, dataset, ut in records:
        grouped[key][physical_data_source(dataset)].append(float(ut))

    selected: dict[tuple[int, int, int], float] = {}
    duplicate_keys = 0
    differing_ut_keys = 0
    source_choices: dict[str, int] = defaultdict(int)
    duplicate_source_sets: Counter[tuple[str, ...]] = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    for key, sources in grouped.items():
        chosen_source = max(sources, key=data_source_rank)
        chosen_values = sources[chosen_source]
        selected[key] = chosen_values[0]
        source_choices[chosen_source] += 1
        record_count = sum(len(values) for values in sources.values())
        if record_count > 1:
            duplicate_keys += 1
            source_set = tuple(sorted(sources))
            duplicate_source_sets[source_set] += 1
            reference = chosen_values[0]
            differing = any(
                not np.isclose(reference, value, rtol=0.0, atol=1.0e-6)
                for values in sources.values()
                for value in values
            )
            if differing:
                differing_ut_keys += 1
            if len(duplicate_examples) < 25:
                duplicate_examples.append(
                    {
                        "run": key[0],
                        "luminosityBlock": key[1],
                        "event": key[2],
                        "sources": {
                            source: values for source, values in sorted(sources.items())
                        },
                        "chosen_source": chosen_source,
                        "chosen_ut": chosen_values[0],
                        "differing_ut": differing,
                    }
                )
    return selected, {
        "input_records": len(records),
        "unique_event_keys": len(grouped),
        "selected_records": len(selected),
        "duplicate_event_keys": duplicate_keys,
        "discarded_records": len(records) - len(selected),
        "duplicate_keys_with_differing_ut": differing_ut_keys,
        "source_choice_event_counts": dict(sorted(source_choices.items())),
        "duplicate_source_sets": [
            {"sources": list(sources), "event_keys": count}
            for sources, count in duplicate_source_sets.most_common()
        ],
        "duplicate_examples": duplicate_examples,
        "policy": (
            "deduplicate by run-lumi-event; prefer NANOv15_v2, then the "
            "highest processing version, then EGamma0; retain the first "
            "target record from the chosen physical source"
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def read_nominal_events(
    input_dir: Path,
    global_datasets: dict[int, str] | None,
) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    # The merged data directory also contains JetMET.  The trusted nominal
    # histogram policy permits only the EGamma stream in GCR.
    files = sorted(input_dir.glob("EGamma*.root"))
    if not files:
        raise FileNotFoundError(f"no nominal ROOT files in {input_dir}")
    records: list[tuple[tuple[int, int, int], str, float]] = []
    rows_read = 0
    for path in files:
        if global_datasets is None:
            metadata = json.loads(path.with_suffix(".json").read_text())
            datasets = {
                int(key): str(record["dataset"])
                for key, record in (metadata.get("datasets") or {}).items()
            }
        else:
            datasets = global_datasets
        with uproot.open(path) as source:
            tree = source["Events"]
            arrays = tree.arrays(
                [
                    "run",
                    "luminosityBlock",
                    "event",
                    "dataset_id",
                    "feature_GCR",
                    "recoil_gcr",
                ],
                library="np",
            )
        rows_read += len(arrays["event"])
        mask = np.asarray(arrays["feature_GCR"], dtype=bool)
        for run, lumi, event, dataset_id, ut in zip(
            arrays["run"][mask],
            arrays["luminosityBlock"][mask],
            arrays["event"][mask],
            arrays["dataset_id"][mask],
            arrays["recoil_gcr"][mask],
        ):
            key = event_key(run, lumi, event)
            dataset = datasets.get(int(dataset_id))
            if dataset is None:
                raise RuntimeError(
                    f"{path}: dataset_id {int(dataset_id)} is absent from metadata"
                )
            records.append((key, dataset, float(ut)))
    selected, dedup_summary = deduplicate(records)
    return selected, {
        "files": len(files),
        "rows_read": rows_read,
        **dedup_summary,
    }


def read_fake_events(
    input_dir: Path,
) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    files = sorted(input_dir.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no photon-fake sidecars in {input_dir}")
    records: list[tuple[tuple[int, int, int], str, float]] = []
    records_read = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete photon-fake sidecar: {path}")
        for record in payload.get("data_events") or []:
            records_read += 1
            if record.get("probe") != "target":
                continue
            key = event_key(
                record["run"],
                record["luminosityBlock"],
                record["event"],
            )
            records.append(
                (
                    key,
                    str(record["source_dataset"]),
                    float(record["values"]["ut"]),
                )
            )
    selected, dedup_summary = deduplicate(records)
    return selected, {
        "files": len(files),
        "records_read": records_read,
        **dedup_summary,
    }


def compact_event(
    key: tuple[int, int, int],
    nominal: dict[tuple[int, int, int], float],
    fake: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    return {
        "run": key[0],
        "luminosityBlock": key[1],
        "event": key[2],
        "nominal_ut": nominal.get(key),
        "fake_ut": fake.get(key),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-data-dir", required=True, type=Path)
    parser.add_argument("--fake-data-dir", required=True, type=Path)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-equal", action="store_true")
    args = parser.parse_args()

    global_datasets = (
        None
        if args.campaign_manifest is None
        else dataset_map_from_campaign_manifest(
            args.campaign_manifest.absolute()
        )
    )
    nominal, nominal_summary = read_nominal_events(
        args.nominal_data_dir.absolute(),
        global_datasets,
    )
    fake, fake_summary = read_fake_events(args.fake_data_dir.absolute())
    nominal_keys = set(nominal)
    fake_keys = set(fake)
    nominal_only = sorted(nominal_keys - fake_keys)
    fake_only = sorted(fake_keys - nominal_keys)
    common = sorted(nominal_keys & fake_keys)
    differing_ut = [
        key
        for key in common
        if not np.isclose(nominal[key], fake[key], rtol=0.0, atol=1.0e-6)
    ]

    nominal_hist, _ = np.histogram(
        np.asarray(list(nominal.values()), dtype=float),
        bins=UT_BINS,
    )
    fake_hist, _ = np.histogram(
        np.asarray(list(fake.values()), dtype=float),
        bins=UT_BINS,
    )
    equal = not nominal_only and not fake_only and not differing_ut
    payload = {
        "schema_version": "photon_fake_gcr_event_key_audit_v1",
        "status": "equal" if equal else "different",
        "selection_source": "real_subset_worker.py",
        "nominal_data_dir": str(args.nominal_data_dir.absolute()),
        "fake_data_dir": str(args.fake_data_dir.absolute()),
        "campaign_manifest": (
            None
            if args.campaign_manifest is None
            else str(args.campaign_manifest.absolute())
        ),
        "nominal": nominal_summary,
        "fake": fake_summary,
        "comparison": {
            "common_event_keys": len(common),
            "nominal_only_event_keys": len(nominal_only),
            "fake_only_event_keys": len(fake_only),
            "differing_ut_event_keys": len(differing_ut),
            "nominal_only_examples": [
                compact_event(key, nominal, fake) for key in nominal_only[:100]
            ],
            "fake_only_examples": [
                compact_event(key, nominal, fake) for key in fake_only[:100]
            ],
            "differing_ut_examples": [
                compact_event(key, nominal, fake) for key in differing_ut[:100]
            ],
        },
        "ut_histogram": {
            "bin_edges": UT_BINS.tolist(),
            "nominal": nominal_hist.tolist(),
            "fake_target": fake_hist.tolist(),
            "difference_fake_minus_nominal": (
                fake_hist - nominal_hist
            ).tolist(),
        },
    }
    write_json(args.output.absolute(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if equal or not args.require_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

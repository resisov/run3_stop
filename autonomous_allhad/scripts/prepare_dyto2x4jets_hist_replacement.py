#!/usr/bin/env python3
"""Build an exclusive nominal-input union replacing every legacy DY input."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


NEW_DY = {
    "dyto2e": {
        "dataset_id": 1763632616,
        "dataset_prefix": "DYto2E-4Jets_Bin-MLL-50_",
    },
    "dyto2mu": {
        "dataset_id": 1972916737,
        "dataset_prefix": "DYto2Mu-4Jets_Bin-MLL-50_",
    },
    "dyto2tau": {
        "dataset_id": 2143375797,
        "dataset_prefix": "DYto2Tau-4Jets_Bin-MLL-50_",
    },
}


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sidecar_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def sidecar_datasets(
    path: str,
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    payload = sidecar_payload(path)
    if require_complete and not str(payload.get("status") or "").startswith(
        "complete"
    ):
        raise RuntimeError(f"incomplete sidecar: {path}")
    return list((payload.get("datasets") or {}).values())


def source_paths(path: str) -> set[str]:
    payload = json.loads(Path(path).read_text())
    return {
        str(record.get("file_path"))
        for record in payload.get("files") or []
        if record.get("file_path")
    }


def is_dy_dataset(record: dict[str, Any]) -> bool:
    dataset = str(record.get("dataset") or "")
    process = str(record.get("process") or "")
    return process == "DY" or dataset.startswith("DYto2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-roots", type=Path, required=True)
    parser.add_argument("--baseline-sidecars", type=Path, required=True)
    parser.add_argument("--new-dy-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_roots = read_lines(args.baseline_roots)
    baseline_sidecars = read_lines(args.baseline_sidecars)
    if len(baseline_roots) != len(baseline_sidecars):
        raise RuntimeError("baseline ROOT and sidecar counts differ")

    retained: list[tuple[str, str]] = []
    excluded: list[dict[str, Any]] = []
    excluded_source_paths: set[str] = set()
    for root, sidecar in zip(baseline_roots, baseline_sidecars):
        if Path(root).with_suffix(".json") != Path(sidecar):
            raise RuntimeError(f"ROOT/sidecar pairing mismatch: {root} {sidecar}")
        payload = sidecar_payload(sidecar)
        datasets = list((payload.get("datasets") or {}).values())
        if not str(payload.get("status") or "").startswith("complete"):
            sources = source_paths(sidecar)
            excluded_source_paths.update(sources)
            excluded.append(
                {
                    "root": root,
                    "sidecar": sidecar,
                    "reason": "baseline_sidecar_not_complete",
                    "sidecar_status": payload.get("status"),
                    "all_processes": sorted(
                        {str(record.get("process")) for record in datasets}
                    ),
                    "source_file_count": len(sources),
                }
            )
            continue
        dy_records = [record for record in datasets if is_dy_dataset(record)]
        if dy_records:
            sources = source_paths(sidecar)
            excluded_source_paths.update(sources)
            excluded.append(
                {
                    "root": root,
                    "sidecar": sidecar,
                    "reason": (
                        "contains_legacy_or_ptll_dy; whole ROOT excluded to "
                        "guarantee zero row-level overlap"
                    ),
                    "all_processes": sorted(
                        {str(record.get("process")) for record in datasets}
                    ),
                    "dy_dataset_ids": sorted(
                        {int(record.get("dataset_id")) for record in dy_records}
                    ),
                    "dy_datasets": sorted(
                        {str(record.get("dataset")) for record in dy_records}
                    ),
                    "source_file_count": len(sources),
                }
            )
        else:
            retained.append((root, sidecar))

    new_pairs: list[tuple[str, str]] = []
    new_source_paths: set[str] = set()
    new_dataset_ids: set[int] = set()
    new_dataset_names: set[str] = set()
    new_dataset_by_id: dict[int, str] = {}
    for process, policy in NEW_DY.items():
        directory = args.new_dy_campaign / "outputs" / process / "nominal"
        sidecars = sorted(directory.glob("*.json"))
        roots = sorted(directory.glob("*.root"))
        if len(sidecars) != len(roots):
            raise RuntimeError(f"{process}: ROOT/sidecar counts differ")
        roots_by_stem = {path.stem: path for path in roots}
        for sidecar in sidecars:
            root = roots_by_stem.get(sidecar.stem)
            if root is None:
                raise RuntimeError(f"{process}: missing ROOT for {sidecar}")
            datasets = sidecar_datasets(str(sidecar))
            if len(datasets) != 1:
                raise RuntimeError(f"{process}: expected one dataset in {sidecar}")
            record = datasets[0]
            dataset_id = int(record.get("dataset_id"))
            dataset = str(record.get("dataset") or "")
            if dataset_id != int(policy["dataset_id"]):
                raise RuntimeError(f"{process}: unexpected dataset ID {dataset_id}")
            if not dataset.startswith(str(policy["dataset_prefix"])):
                raise RuntimeError(f"{process}: unexpected dataset {dataset}")
            if "PTLL" in dataset:
                raise RuntimeError(f"{process}: PTLL dataset leaked into new DY")
            if str(record.get("process")) != "DY":
                raise RuntimeError(f"{process}: non-DY process label")
            new_dataset_ids.add(dataset_id)
            new_dataset_names.add(dataset)
            previous_dataset = new_dataset_by_id.setdefault(dataset_id, dataset)
            if previous_dataset != dataset:
                raise RuntimeError(
                    f"{process}: dataset ID {dataset_id} maps to multiple names"
                )
            paths = source_paths(str(sidecar))
            if new_source_paths.intersection(paths):
                raise RuntimeError(f"{process}: duplicate NanoAOD source coverage")
            new_source_paths.update(paths)
            new_pairs.append((str(root), str(sidecar)))

    expected_ids = {int(item["dataset_id"]) for item in NEW_DY.values()}
    if new_dataset_ids != expected_ids:
        raise RuntimeError(
            f"new dataset IDs {new_dataset_ids} != expected {expected_ids}"
        )
    source_overlap = sorted(excluded_source_paths.intersection(new_source_paths))
    if source_overlap:
        raise RuntimeError("legacy/new source overlap: " + ", ".join(source_overlap))

    final_pairs = retained + new_pairs
    final_roots = [root for root, _sidecar in final_pairs]
    final_sidecars = [sidecar for _root, sidecar in final_pairs]
    if len(final_roots) != len(set(final_roots)):
        raise RuntimeError("duplicate final ROOT paths")
    if len(final_sidecars) != len(set(final_sidecars)):
        raise RuntimeError("duplicate final sidecar paths")

    final_ptll_rows = 0
    final_dy_ids: set[int] = set()
    final_dy_names: set[str] = set()
    for sidecar in final_sidecars:
        for record in sidecar_datasets(sidecar):
            dataset = str(record.get("dataset") or "")
            if "PTLL" in dataset:
                final_ptll_rows += 1
            if is_dy_dataset(record):
                final_dy_ids.add(int(record.get("dataset_id")))
                final_dy_names.add(dataset)
    if final_ptll_rows:
        raise RuntimeError(f"{final_ptll_rows} PTLL dataset rows remain")
    if final_dy_ids != expected_ids:
        raise RuntimeError(
            f"final DY IDs {final_dy_ids} != new-only IDs {expected_ids}"
        )

    roots_path = output_dir / "input_roots.txt"
    sidecars_path = output_dir / "input_sidecars.txt"
    roots_path.write_text("\n".join(final_roots) + "\n")
    sidecars_path.write_text("\n".join(final_sidecars) + "\n")
    write_json(
        output_dir / "dyto2x4jets_xsec_overrides.json",
        {
            "schema": "background_xsec_overrides_v1",
            "status": "pending_xsdb",
            "datasets": {
                str(dataset_id): {
                    "dataset": dataset,
                    "xsec_pb": None,
                    "source": "XSDB official",
                }
                for dataset_id, dataset in sorted(new_dataset_by_id.items())
            },
        },
    )
    manifest = {
        "schema": "dyto2x4jets_histogram_replacement_union_v1",
        "status": "inputs_complete_xsec_pending",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": {
            "legacy_dy": (
                "exclude every baseline ROOT containing any DY dataset row; "
                "this includes mixed-process residuals"
            ),
            "adopted_dy": (
                "DYto2E-4Jets + DYto2Mu-4Jets + DYto2Tau-4Jets only"
            ),
            "normalization": (
                "histogram production is blocked until all three official "
                "XSDB cross sections are present"
            ),
        },
        "counts": {
            "baseline_roots": len(baseline_roots),
            "retained_non_dy_roots": len(retained),
            "excluded_roots_containing_dy": len(excluded),
            "new_dy_roots": len(new_pairs),
            "final_roots": len(final_roots),
            "final_sidecars": len(final_sidecars),
            "excluded_legacy_source_files": len(excluded_source_paths),
            "new_dy_source_files": len(new_source_paths),
            "legacy_new_source_overlap": len(source_overlap),
            "final_ptll_dataset_rows": final_ptll_rows,
        },
        "new_dy": {
            "dataset_ids": sorted(new_dataset_ids),
            "datasets": sorted(new_dataset_names),
            "source_files": len(new_source_paths),
        },
        "excluded": excluded,
        "lists": {
            "roots": str(roots_path),
            "sidecars": str(sidecars_path),
        },
        "sha256": {
            "roots": sha256_lines(final_roots),
            "sidecars": sha256_lines(final_sidecars),
        },
        "xsec_overrides": str(
            output_dir / "dyto2x4jets_xsec_overrides.json"
        ),
    }
    write_json(output_dir / "input_union_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "new_dy": manifest["new_dy"],
                "manifest": str(output_dir / "input_union_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

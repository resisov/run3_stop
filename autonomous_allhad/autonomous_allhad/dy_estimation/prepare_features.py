#!/usr/bin/env python3
"""Prepare the auditable HTCondor feature-stage campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..sidecar_store import read_root_metadata


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def zll_sample(dataset: str, process: str) -> bool:
    upper = dataset.upper()
    return process == "DY" or any(
        token in upper for token in ("TTZ", "WZ", "ZZ", "WWZ", "WZZ", "ZZZ", "WZG")
    )


def resolve_root(raw: str, run_directory: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else run_directory / path


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving EOS namespace symlinks.

    ``Path.resolve()`` rewrites ``/eos/user/t/...`` to ``/eos/home-t/...`` on
    lxplus.  The latter alias is not mounted on all batch workers.
    """

    return path if path.is_absolute() else Path.cwd() / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=("DY2E", "DY2M"), required=True)
    parser.add_argument("--roots-per-shard", type=int, default=60)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=8000)
    parser.add_argument(
        "--prevalidated-inputs",
        action="store_true",
        help=(
            "Partition a previously validated data/background ROOT list "
            "without repeating per-ROOT sidecar discovery. Dataset-level "
            "stream and DY-family filtering still runs in feature_stage."
        ),
    )
    parser.add_argument(
        "--input-validation",
        type=Path,
        help="Complete histogram_validation.json that certifies the input campaign.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python"),
    )
    args = parser.parse_args(argv)

    repo = lexical_absolute(args.repo)
    run_directory = repo / "autonomous_allhad"
    output_dir = lexical_absolute(args.output_dir)
    normalization = lexical_absolute(args.normalization)
    partitions_dir = output_dir / "partitions"
    outputs_dir = output_dir / "outputs"
    logs_dir = output_dir / "logs"
    for directory in (partitions_dir, outputs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stream = "EGamma" if args.channel == "DY2E" else "Muon"
    raw_roots = [
        root
        for input_path in args.input_roots
        for root in read_lines(input_path)
    ]
    selected: dict[str, list[str]] = {"data": [], "zll": [], "other": [], "mixed": []}
    exclusions: dict[str, int] = {
        "signal": 0,
        "wrong_data_stream": 0,
        "incomplete_sidecar": 0,
    }
    dataset_counts: dict[str, int] = {}

    prevalidation: dict[str, Any] | None = None
    if args.prevalidated_inputs:
        if args.input_validation is None:
            parser.error("--prevalidated-inputs requires --input-validation")
        prevalidation = json.loads(args.input_validation.read_text())
        if prevalidation.get("status") != "complete":
            raise RuntimeError(
                f"input validation is not complete: {prevalidation.get('status')}"
            )
        if len(raw_roots) != len(set(raw_roots)):
            raise RuntimeError("duplicate ROOT paths in prevalidated input list")
        for raw in raw_roots:
            root = resolve_root(raw, run_directory)
            stem = root.name
            if stem.startswith("data_shard_"):
                category = "data"
            elif stem.startswith("mc_shard_"):
                # A feature ROOT can contain several background datasets.
                # feature_stage performs the authoritative dataset-level
                # Z-like/other split and channel routing.
                category = "mixed"
            else:
                raise RuntimeError(
                    "prevalidated RZ input must contain only data_shard_ or "
                    f"mc_shard_ ROOTs, found {root}"
                )
            selected[category].append(str(root))
    else:
        for raw in raw_roots:
            root = resolve_root(raw, run_directory)
            try:
                metadata = read_root_metadata(root)
            except FileNotFoundError:
                exclusions["incomplete_sidecar"] += 1
                continue
            if not str(metadata.get("status") or "").startswith("complete"):
                exclusions["incomplete_sidecar"] += 1
                continue
            records = list((metadata.get("datasets") or {}).values())
            if any(bool(record.get("is_signal")) for record in records):
                exclusions["signal"] += 1
                continue
            is_data = any(bool(record.get("is_data")) for record in records)
            if is_data:
                processes = {str(record.get("process") or "") for record in records}
                if processes != {stream}:
                    exclusions["wrong_data_stream"] += 1
                    continue
                category = "data"
            else:
                components = {
                    "zll" if zll_sample(
                        str(record.get("dataset") or ""),
                        str(record.get("process") or ""),
                    ) else "other"
                    for record in records
                }
                category = next(iter(components)) if len(components) == 1 else "mixed"
            selected[category].append(str(root))
            for record in records:
                dataset = str(record.get("dataset") or "unknown")
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    all_selected = [root for category in selected.values() for root in category]
    if len(all_selected) != len(set(all_selected)):
        raise RuntimeError("duplicate selected ROOT paths")

    queue_rows: list[tuple[str, str, str]] = []
    partition_counts: dict[str, int] = {}
    roots_per_shard = max(1, args.roots_per_shard)
    for category in ("data", "zll", "mixed", "other"):
        roots = sorted(selected[category])
        shard_count = math.ceil(len(roots) / roots_per_shard) if roots else 0
        partition_counts[category] = shard_count
        for index in range(shard_count):
            shard = roots[index * roots_per_shard : (index + 1) * roots_per_shard]
            stem = f"{category}_{index:03d}"
            input_path = partitions_dir / f"{stem}.txt"
            output_path = outputs_dir / f"{stem}.json"
            input_path.write_text("\n".join(shard) + "\n")
            queue_rows.append((str(input_path), str(output_path), stem))

    queue_path = output_dir / "queue.tsv"
    queue_path.write_text(
        "\n".join("\t".join(row) for row in queue_rows) + "\n"
    )
    wrapper_path = output_dir / "run_feature.sh"
    wrapper_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export PYTHONPATH={run_directory}",
                "export PYTHONUNBUFFERED=1",
                f"cd {run_directory}",
                f"exec {args.python} -m autonomous_allhad.dy_estimation build-features \"$@\"",
                "",
            ]
        )
    )
    wrapper_path.chmod(0o755)
    submit_path = output_dir / "submit.sub"
    submit_path.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {wrapper_path}",
                (
                    f"arguments = --repo {repo} --input-list $(input_list) "
                    f"--normalization {normalization} "
                    "--output $(output_json) "
                    f"--jobs {args.cpus} --step-size 100000 "
                    f"--channels {args.channel}"
                ),
                f"initialdir = {run_directory}",
                "should_transfer_files = NO",
                f"request_cpus = {args.cpus}",
                f"request_memory = {args.memory_mb}",
                '+JobFlavour = "tomorrow"',
                '+JobBatchName = "DY_RZ_%s"' % args.channel,
                f"output = {logs_dir}/$(stem).out",
                f"error = {logs_dir}/$(stem).err",
                f"log = {logs_dir}/$(stem).log",
                "on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)",
                f"queue input_list, output_json, stem from {queue_path}",
                "",
            ]
        )
    )

    manifest = {
        "schema_version": "dy_estimation_feature_campaign_2024_v1",
        "status": "prepared",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "channel": args.channel,
        "data_stream": stream,
        "method": {
            "on_z": "81 < mll < 101 GeV",
            "off_z": "50 < mll < 81 GeV or mll > 101 GeV",
            "fit": (
                "RZ and RT in Nb=1 and Nb>=2, with Poisson data and "
                "Gaussian MC-stat constraints"
            ),
            "normalization": str(normalization),
            "entrypoint": "python -m autonomous_allhad.dy_estimation build-features",
            "code_sha256": {
                path.name: sha256_file(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
        "input": {
            "source": [str(lexical_absolute(path)) for path in args.input_roots],
            "roots": len(raw_roots),
            "sha256": sha256_lines(raw_roots),
            "prevalidated": bool(args.prevalidated_inputs),
            "validation": (
                {
                    "path": str(lexical_absolute(args.input_validation)),
                    "sha256": sha256_file(args.input_validation),
                    "status": prevalidation.get("status"),
                    "validated_roots": prevalidation.get("unique_input_root_count"),
                    "strict_warning_counts": prevalidation.get("strict_warning_counts"),
                }
                if prevalidation is not None
                else None
            ),
        },
        "selected": {
            "roots": len(all_selected),
            "sha256": sha256_lines(sorted(all_selected)),
            "by_category": {key: len(value) for key, value in selected.items()},
        },
        "excluded": exclusions,
        "partitions": {
            "roots_per_shard": roots_per_shard,
            "counts": partition_counts,
            "total": len(queue_rows),
            "queue": str(queue_path),
            "submit": str(submit_path),
            "wrapper": str(wrapper_path),
        },
        "dataset_sidecar_counts": dict(sorted(dataset_counts.items())),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

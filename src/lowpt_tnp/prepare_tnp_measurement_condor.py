#!/usr/bin/env python3
"""Prepare an EOS-only low-pT tag-and-probe Condor campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _eos(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    text = str(resolved)
    if not text.startswith("/eos/") or "/afs/" in text.lower():
        raise ValueError(f"{label} must be an EOS path and must not reference AFS: {text}")
    return resolved


def prepare(
    *,
    records_path: Path,
    workdir: Path,
    python_archive: Path,
    runtime_archive: Path,
    proxy: Path,
    config: Path,
    kind: str,
    files_per_shard: int,
) -> dict[str, Any]:
    paths = {
        name: _eos(path, name)
        for name, path in {
            "records": records_path,
            "workdir": workdir,
            "python_archive": python_archive,
            "runtime_archive": runtime_archive,
            "proxy": proxy,
            "config": config,
        }.items()
    }
    if kind not in {"electron", "muon"}:
        raise ValueError(f"unsupported TnP kind: {kind}")
    if files_per_shard != 20:
        raise ValueError("TnP production requires exactly 20 ROOT files per full shard")
    for name in ("records", "python_archive", "runtime_archive", "proxy", "config"):
        if not paths[name].exists():
            raise FileNotFoundError(f"{name} does not exist: {paths[name]}")
    payload = json.loads(paths["records"].read_text())
    measurement_config = json.loads(paths["config"].read_text())
    expected_year = str(measurement_config.get("year") or "")
    if not expected_year:
        raise ValueError("TnP campaign config must define year")
    if str(payload.get("year") or "") != expected_year:
        raise ValueError(
            f"records/config year mismatch: {payload.get('year')!r} != {expected_year!r}"
        )
    expected_mc_datasets = {
        str(dataset)
        for dataset in measurement_config.get("campaign_inputs", {}).get("mc_datasets", [])
    }
    if not expected_mc_datasets:
        raise ValueError("TnP campaign config must define at least one exact MC dataset")
    expected_definition = {
        "electron": "veto_id_only",
        "muon": "loose_id_only",
    }[kind]
    strategy = str(
        measurement_config.get("campaign_strategy")
        or ("parking_external_electron" if kind == "electron" else "isomu24_tag")
    )
    parking_paths = [
        "HLT_Mu9_Barrel_L1HP10_IP6",
        "HLT_Mu10_Barrel_L1HP11_IP6",
    ]
    if kind == "electron":
        expected_reference_paths = parking_paths
        expected_filter_bits = None
        expected_trigger_match = False
        expected_apply_reference_to_mc = False
        minimum_tag_pt = 5.0
        expected_data_prefixes = ("/ParkingSingleMuon",)
        expected_external = {
            "enabled": True,
            "pt_min_gev": 12.0,
            "abseta_max": 1.5,
            "require_tight_id": True,
            "miniiso_max": None,
        }
        configured_external = measurement_config.get("external_reference_muon")
        if configured_external is not None and configured_external != expected_external:
            raise ValueError(
                "parking-external electron TnP has an inconsistent offline muon topology: "
                f"{configured_external!r}"
            )
    elif strategy == "parking_external_muon":
        expected_reference_paths = parking_paths
        expected_filter_bits = None
        expected_trigger_match = False
        expected_apply_reference_to_mc = False
        minimum_tag_pt = 5.0
        expected_data_prefixes = ("/ParkingSingleMuon",)
        expected_external = {
            "enabled": True,
            "pt_min_gev": 12.0,
            "abseta_max": 1.5,
            "require_tight_id": True,
            "miniiso_max": None,
        }
        if measurement_config.get("external_reference_muon") != expected_external:
            raise ValueError(
                "parking-external muon TnP requires the audited third-muon topology: "
                f"{expected_external!r}"
            )
        if measurement_config.get("tag_miniiso_max", 0.1) is not None:
            raise ValueError(
                "parking-external muon TnP must not impose mini-isolation on the measured J/psi tag"
            )
    elif strategy == "isomu24_tag":
        expected_reference_paths = ["HLT_IsoMu24"]
        expected_filter_bits = 2
        expected_trigger_match = True
        expected_apply_reference_to_mc = True
        minimum_tag_pt = 26.0
        expected_data_prefixes = ("/Muon0/", "/Muon1/")
    else:
        raise ValueError(f"unsupported muon campaign_strategy: {strategy!r}")
    expected_probe_pt_edges = {
        "electron": [5.0, 7.0, 10.0],
        "muon": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }[kind]
    expected_reference_kind = "muon"
    if payload.get("measurement") != measurement_config.get("measurement"):
        raise ValueError(
            f"records/config measurement mismatch: {payload.get('measurement')!r} != "
            f"{measurement_config.get('measurement')!r}"
        )
    if payload.get("probe_definition") != expected_definition:
        raise ValueError(
            f"ID-only {kind} records require probe_definition={expected_definition!r}; "
            f"got {payload.get('probe_definition')!r}"
        )
    if measurement_config.get("probe_definition") != expected_definition:
        raise ValueError(
            f"ID-only {kind} campaign requires probe_definition={expected_definition!r}; "
            f"got {measurement_config.get('probe_definition')!r}"
        )
    if list(measurement_config.get("reference_paths") or []) != expected_reference_paths:
        raise ValueError(
            f"ID-only {kind} campaign requires reference_paths={expected_reference_paths!r}; "
            f"got {measurement_config.get('reference_paths')!r}"
        )
    if list(payload.get("reference_paths") or []) != expected_reference_paths:
        raise ValueError(
            f"ID-only {kind} records require reference_paths={expected_reference_paths!r}; "
            f"got {payload.get('reference_paths')!r}"
        )
    actual_filter_bits = measurement_config.get("tag_trigger_object_filter_bits")
    if expected_filter_bits is None:
        filter_bits_match = actual_filter_bits is None
    else:
        filter_bits_match = int(actual_filter_bits or -1) == expected_filter_bits
    if not filter_bits_match:
        raise ValueError(
            f"ID-only {kind} campaign requires tag trigger-object bits={expected_filter_bits}; "
            f"got {actual_filter_bits!r}"
        )
    if bool(measurement_config.get("tag_trigger_match_required", True)) != expected_trigger_match:
        raise ValueError(
            f"ID-only {kind} campaign requires tag_trigger_match_required="
            f"{expected_trigger_match!r}"
        )
    if bool(measurement_config.get("apply_reference_trigger_to_mc", True)) != expected_apply_reference_to_mc:
        raise ValueError(
            f"ID-only {kind} campaign requires apply_reference_trigger_to_mc="
            f"{expected_apply_reference_to_mc!r}"
        )
    if measurement_config.get("reference_trigger_object_kind", kind) != expected_reference_kind:
        raise ValueError(
            f"ID-only {kind} campaign requires reference_trigger_object_kind="
            f"{expected_reference_kind!r}"
        )
    probe_pt_edges = [float(value) for value in measurement_config.get("probe_pt_edges_gev") or []]
    if probe_pt_edges != expected_probe_pt_edges:
        raise ValueError(
            f"ID-only {kind} campaign requires probe pT edges={expected_probe_pt_edges!r}; "
            f"got {probe_pt_edges!r}"
        )
    tag_pt_min = float(measurement_config.get("tag_pt_min_gev", 0.0))
    if tag_pt_min < minimum_tag_pt:
        raise ValueError(
            f"ID-only {kind} campaign requires an explicit offline tag plateau of at least "
            f"{minimum_tag_pt:g} GeV; "
            f"got {tag_pt_min:g} GeV"
        )
    records = list(payload.get("records") or [])
    if not records:
        raise ValueError("empty TnP campaign records")
    for record in records:
        if str(record.get("sample")) not in {"data", "mc"}:
            raise ValueError(f"invalid TnP sample in record: {record}")
        if not str(record.get("file_path") or ""):
            raise ValueError(f"missing TnP file_path in record: {record}")
        if str(record.get("sample")) == "data":
            dataset = str(record.get("dataset") or "")
            if not dataset.startswith(expected_data_prefixes):
                raise ValueError(
                    f"{kind} {strategy} data must start with {expected_data_prefixes!r}; "
                    f"got {dataset!r}"
                )
        else:
            dataset = str(record.get("dataset") or "")
            if dataset not in expected_mc_datasets:
                raise ValueError(
                    f"{kind} {strategy} MC is outside the frozen config dataset list: "
                    f"{dataset!r} not in {sorted(expected_mc_datasets)!r}"
                )

    workdir = paths["workdir"]
    manifests = workdir / "manifests"
    outputs = workdir / "shard_outputs"
    logs = workdir / "logs"
    for directory in (workdir, manifests, outputs, logs):
        directory.mkdir(parents=True, exist_ok=True)

    queue_rows = []
    shard_paths = []
    for index, start in enumerate(range(0, len(records), files_per_shard)):
        subset = records[start : start + files_per_shard]
        name = f"tnp_{kind}_{index:05d}"
        shard = manifests / f"{name}.json"
        shard.write_text(json.dumps({
            "schema_version": 1,
            "shard_id": name,
            "kind": kind,
            "record_digest": hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest(),
            "files_per_shard": files_per_shard,
            "records": subset,
        }, indent=2, sort_keys=True) + "\n")
        queue_rows.append((name, shard, shard.name, outputs / f"shard_{index:05d}.json"))
        shard_paths.append(str(shard))

    wrapper = workdir / "run_tnp_measurement.sh"
    wrapper_text = f"""#!/bin/bash
set -euo pipefail
SHARD_NAME="$1"
RESULT_DEST="$2"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
cd "$WORKDIR"
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="$WORKDIR/runtime_cache/pycache"
export LOWPT_TNP_XRD_CACHE="$WORKDIR/runtime_xrd"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$LOWPT_TNP_XRD_CACHE"
export X509_USER_PROXY="$WORKDIR/{paths['proxy'].name}"
chmod 600 "$X509_USER_PROXY"
tar -xzf {paths['python_archive'].name}
tar -xzf {paths['runtime_archive'].name}
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORKDIR/src"
"$PY" -c 'import numpy, awkward, uproot, correctionlib; print(numpy.__file__); print(awkward.__file__); print(uproot.__file__)'
"$PY" -m lowpt_tnp count --kind {kind} --project-root "$WORKDIR" --shard "$WORKDIR/$SHARD_NAME" --config "$WORKDIR/{paths['config'].name}" --output "$WORKDIR/result.json"
test -s "$WORKDIR/result.json"
"$PY" -c 'import json; p=json.load(open("result.json")); assert p["status"] in ("success", "incomplete"); assert p["files_processed"] > 0'
case "$RESULT_DEST" in /eos/user/*) ;; *) echo "refusing non-EOS result destination: $RESULT_DEST" >&2; exit 64;; esac
xrdcp -f --nopbar "$WORKDIR/result.json" "root://eosuser.cern.ch/$RESULT_DEST"
xrdfs eosuser.cern.ch stat "$RESULT_DEST" >/dev/null
"""
    wrapper.write_text(wrapper_text)
    wrapper.chmod(0o700)

    queue = workdir / "queue.tsv"
    queue.write_text("".join(
        f"{name}\t{shard}\t{shard_name}\t{output}\n"
        for name, shard, shard_name, output in queue_rows
    ))
    submit = workdir / "submit.sub"
    submit_text = f"""universe = vanilla
executable = {wrapper}
arguments = $(shard_name) $(result)
initialdir = {workdir}
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/$(name).log
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {paths['python_archive']}, {paths['runtime_archive']}, {paths['proxy']}, {paths['config']}, $(shard)
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {paths['proxy']}
request_cpus = 1
request_memory = 4000MB
request_disk = 6000MB
+JobFlavour = "tomorrow"
queue name,shard,shard_name,result from {queue}
"""
    rendered = "\n".join((wrapper_text, submit_text, queue.read_text(), *shard_paths))
    if "/afs/" in rendered.lower():
        raise RuntimeError("AFS reference detected; refusing TnP submission")
    submit.write_text(submit_text)
    summary = {
        "schema_version": 1,
        "measurement": payload["measurement"],
        "probe_definition": measurement_config["probe_definition"],
        "campaign_strategy": strategy,
        "tag_pt_min_gev": tag_pt_min,
        "reference_paths": expected_reference_paths,
        "tag_trigger_object_filter_bits": expected_filter_bits,
        "tag_trigger_match_required": expected_trigger_match,
        "reference_trigger_object_kind": expected_reference_kind,
        "apply_reference_trigger_to_mc": expected_apply_reference_to_mc,
        "kind": kind,
        "records": len(records),
        "files_per_shard": files_per_shard,
        "shards": len(queue_rows),
        "workdir": str(workdir),
        "python_archive": str(paths["python_archive"]),
        "runtime_archive": str(paths["runtime_archive"]),
        "proxy": str(paths["proxy"]),
        "submit_file": str(submit),
        "queue_file": str(queue),
        "submission_backend": "EOS schedd selected by module load lxbatch/eossubmit",
        "submit_command": f"module load lxbatch/eossubmit && condor_submit {submit}",
        "afs_reference_check": "passed",
    }
    (workdir / "campaign.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def cli(argv: list[str] | None = None, *, default_kind: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--python-archive", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("electron", "muon"), default=default_kind, required=default_kind is None)
    parser.add_argument("--files-per-shard", type=int, default=20)
    args = parser.parse_args(argv)
    summary = prepare(
        records_path=args.records,
        workdir=args.workdir,
        python_archive=args.python_archive,
        runtime_archive=args.runtime_archive,
        proxy=args.proxy,
        config=args.config,
        kind=args.kind,
        files_per_shard=args.files_per_shard,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(cli())

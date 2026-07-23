from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


DATA_YEAR = "2025"
CORRECTION_YEAR = "2024"
GOODRUN_RELATIVE_PATH = Path(
    "analysis/data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json"
)
DEFAULT_DATASET_LIST = Path("analysis/datasets/datasets_2025_data.txt")
DEFAULT_METADATA = Path("analysis/metadata/KNU_2025_v4.json.gz")
DEFAULT_MC_INPUTS = Path(
    "autonomous_allhad/workflow/highlowdm_full_20260705_root_inputs.txt"
)
DEFAULT_CAMPAIGN = Path(
    "autonomous_allhad/workflow/flat_ntuple_2025_data_2024corr_20260714"
)
DEFAULT_PYTHON_ENV = Path("condor/py38.tgz")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        if executable:
            path.chmod(0o755)
        return
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text)
        if executable:
            tmp.chmod(0o755)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_dataset_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate dataset keys in {path}")
    return sorted(names)


def process_group(dataset: str) -> str:
    stream = dataset.split("-", 1)[0]
    if stream.startswith("JetMET"):
        return "JetMET"
    if stream.startswith("EGamma"):
        return "EGamma"
    if stream.startswith("Muon"):
        return "Muon"
    raise ValueError(f"Unsupported 2025 data stream in dataset key: {dataset}")


def metadata_files(entry: Any, dataset: str) -> list[str]:
    if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
        raise TypeError(f"Metadata entry {dataset} does not contain a files list")
    files = [str(item).strip() for item in entry["files"] if str(item).strip()]
    if not files:
        raise RuntimeError(f"Metadata entry {dataset} has no ROOT files")
    return sorted(files)


def build_records(
    dataset_names: list[str], metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for dataset in dataset_names:
        if dataset not in metadata:
            raise KeyError(f"Dataset key missing from metadata: {dataset}")
        group = process_group(dataset)
        for file_index, file_path in enumerate(metadata_files(metadata[dataset], dataset)):
            if not file_path.startswith("root://"):
                raise ValueError(f"Non-XRootD input path in {dataset}: {file_path}")
            if file_path in seen:
                duplicates.append(
                    {
                        "file_path": file_path,
                        "first_dataset": seen[file_path],
                        "duplicate_dataset": dataset,
                    }
                )
                continue
            seen[file_path] = dataset
            records.append(
                {
                    "sample_name": dataset,
                    "dataset": dataset,
                    "process_group": group,
                    "year": DATA_YEAR,
                    "correction_year": CORRECTION_YEAR,
                    "file_index": file_index,
                    "file_path": file_path,
                    "xsec_pb": -1.0,
                    "sumw_source": "data_unweighted",
                    "processing_status": "not_submitted",
                    "is_data": True,
                    "is_background": False,
                    "is_signal": False,
                }
            )
    return records, duplicates


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def render_wrapper(repo: Path, python_env: Path, proxy: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

echo "CONDOR_SCRATCH=$PWD"
name="$1"
shard="$2"
root_out="$3"
meta_out="$4"

repo="{repo}"
job_runtime="$PWD/runtime"
python_env="$PWD/{python_env.name}"
proxy="$PWD/{proxy.name}"

cleanup() {{
    rm -rf "$job_runtime"
}}
trap cleanup EXIT

mkdir -p "$job_runtime/home" "$job_runtime/cache" "$job_runtime/xrd" "$job_runtime/fragments"
export HOME="$job_runtime/home"
export TMPDIR="$job_runtime/cache"
export TMP="$job_runtime/cache"
export TEMP="$job_runtime/cache"
export XDG_CACHE_HOME="$job_runtime/cache"
export MPLCONFIGDIR="$job_runtime/cache/matplotlib"
export NUMBA_CACHE_DIR="$job_runtime/cache/numba"
export AUTONOMOUS_ALLHAD_ANALYSIS_CACHE_DIR="$job_runtime/cache/analysis"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$job_runtime/xrd"
export AUTONOMOUS_ALLHAD_FRAGMENT_DIR="$job_runtime/fragments"
export AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA=0
export AUTONOMOUS_ALLHAD_FLAT_CHUNK=30000
export AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT=1800
export AUTONOMOUS_ALLHAD_XRD_STREAMS=4
export AUTONOMOUS_ALLHAD_RECORD_WORKERS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo/autonomous_allhad"
export XRD_NETWORKSTACK=IPv4
export X509_USER_PROXY="$proxy"

chmod 600 "$X509_USER_PROXY"
test -s "$python_env"
test -f "$repo/{GOODRUN_RELATIVE_PATH}"

tar -xzf "$python_env"
pyroot="$PWD/py38"
if [ ! -x "$pyroot/bin/python" ] && [ -x "$PWD/bin/python" ]; then
    pyroot="$PWD"
fi
if [ ! -x "$pyroot/bin/python" ]; then
    pybin=$(find "$PWD" -maxdepth 3 \\( -type f -o -type l \\) -path "*/bin/python" -print -quit)
    pyroot=$(dirname "$(dirname "$pybin")")
fi
python="$pyroot/bin/python"
if [ ! -x "$python" ]; then
    echo "local python not found after unpacking {python_env.name}" >&2
    exit 66
fi
export PATH="$pyroot/bin:$PATH"
export LD_LIBRARY_PATH="$pyroot/lib:${{LD_LIBRARY_PATH:-}}"

"$python" - <<'CHECK'
import sys
import awkward
import numba
import numpy
import uproot

print("sys.executable", sys.executable)
print("numpy.__file__", numpy.__file__)
print("awkward.__file__", awkward.__file__)
print("numba.__file__", numba.__file__)
print("uproot.__file__", uproot.__file__)
CHECK

"$python" -m autonomous_allhad.flat_ntuple_worker \
    --repo "$repo" \
    --shard "$shard" \
    --output "$root_out" \
    --metadata-output "$meta_out" \
    --shift nominal \
    --skim-flag feature_flat_preselection \
    --record-workers 1

"$python" -c 'import json,sys; from pathlib import Path; d=json.load(open(sys.argv[1])); ok=d.get("status")=="complete" and Path(sys.argv[2]).is_file() and d.get("files_processed")==d.get("files_attempted"); raise SystemExit(0 if ok else 70)' "$meta_out" "$root_out"
"""


def render_submit(campaign: Path, python_env: Path, proxy: Path) -> str:
    return f"""universe = vanilla
executable = {campaign / "condor" / "run_flat_2025_data.sh"}
initialdir = {campaign}
arguments = $(name) $(shard) $(root_out) $(meta_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {python_env}, {proxy}
transfer_output_files = ""
output = {campaign / "logs"}/$(name).out
error = {campaign / "logs"}/$(name).err
log = {campaign / "logs"}/campaign.log
request_cpus = 4
request_memory = 6000MB
request_disk = 10000MB
+JobFlavour = "workday"
queue name,shard,root_out,meta_out from {campaign / "condor" / "arguments.txt"}
"""


def read_2024_mc_inputs(path: Path) -> tuple[list[str], int]:
    all_inputs = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    data_inputs = [
        item for item in all_inputs if Path(item).name.startswith("data_shard_")
    ]
    mc_inputs = [
        item for item in all_inputs if not Path(item).name.startswith("data_shard_")
    ]
    if not mc_inputs:
        raise RuntimeError(f"No reusable 2024 MC inputs found in {path}")
    return mc_inputs, len(data_inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an EOS-only 2025 DATA flat-ntuple campaign using 2024 object corrections."
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--dataset-list", type=Path, default=DEFAULT_DATASET_LIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--existing-2024-input-list", type=Path, default=DEFAULT_MC_INPUTS)
    parser.add_argument("--python-env", type=Path, default=DEFAULT_PYTHON_ENV)
    parser.add_argument("--proxy", type=Path, default=None)
    parser.add_argument("--files-per-shard", type=int, default=5)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else repo / path

    campaign = absolute(args.campaign_dir)
    dataset_list = absolute(args.dataset_list)
    metadata_path = absolute(args.metadata)
    old_input_list = absolute(args.existing_2024_input_list)
    python_env = absolute(args.python_env)
    goodrun = repo / GOODRUN_RELATIVE_PATH
    proxy = absolute(args.proxy) if args.proxy else campaign / "credentials" / "x509up"

    if args.files_per_shard < 1:
        raise ValueError("--files-per-shard must be positive")
    for path in (dataset_list, metadata_path, old_input_list, goodrun, python_env):
        if not path.exists():
            raise FileNotFoundError(path)
    if not str(campaign).startswith("/eos/"):
        raise ValueError(f"Campaign directory must be on EOS: {campaign}")
    for label, path in (("Python environment", python_env), ("proxy", proxy)):
        if not str(path).startswith("/eos/"):
            raise ValueError(f"{label} must be on EOS: {path}")

    dataset_names = read_dataset_names(dataset_list)
    with gzip.open(metadata_path, "rt") as handle:
        metadata = json.load(handle)
    records, duplicates = build_records(dataset_names, metadata)
    if duplicates:
        raise RuntimeError(
            f"Found {len(duplicates)} duplicate ROOT files across 2025 data metadata"
        )

    reused_mc, excluded_2024_data = read_2024_mc_inputs(old_input_list)
    for directory in ("shards", "outputs", "logs", "runtime", "condor", "credentials"):
        (campaign / directory).mkdir(parents=True, exist_ok=True)

    args_lines: list[str] = []
    shard_digests: list[str] = []
    for shard_index, shard_records in enumerate(chunks(records, args.files_per_shard)):
        name = f"data_shard_{shard_index:05d}"
        record_digest = sha256_bytes(stable_json(shard_records).encode())
        shard_payload = {
            "schema_version": "flat_ntuple_2025_data_shard_v1",
            "campaign": campaign.name,
            "shard_id": name,
            "record_digest": record_digest,
            "records": shard_records,
            "data_taking_year": DATA_YEAR,
            "correction_year": CORRECTION_YEAR,
            "goodrun_json": str(goodrun),
        }
        shard_path = campaign / "shards" / f"{name}.json"
        root_out = campaign / "outputs" / f"{name}.root"
        meta_out = campaign / "outputs" / f"{name}.json"
        write_json(shard_path, shard_payload)
        args_lines.append(f"{name} {shard_path} {root_out} {meta_out}")
        shard_digests.append(record_digest)

    write_text(campaign / "condor" / "arguments.txt", "\n".join(args_lines) + "\n")
    write_text(
        campaign / "condor" / "run_flat_2025_data.sh",
        render_wrapper(repo, python_env, proxy),
        executable=True,
    )
    write_text(
        campaign / "condor" / "flat_2025_data.sub",
        render_submit(campaign, python_env, proxy),
    )
    write_text(campaign / "reused_2024_mc_inputs.txt", "\n".join(reused_mc) + "\n")

    manifest = {
        "schema_version": "flat_ntuple_2025_data_campaign_v1",
        "campaign": campaign.name,
        "created_at": utc_now(),
        "status": "prepared_not_submitted",
        "repo": str(repo),
        "policy": {
            "data_taking_year": DATA_YEAR,
            "goodrun_year": DATA_YEAR,
            "correction_year": CORRECTION_YEAR,
            "mc_intermediate_production_jobs": 0,
            "mc_policy": "reuse_existing_2024_background_and_signal_MC_flat_ROOT_inputs",
            "temporary_storage_policy": "EOS_only_no_AFS_no_system_tmp",
            "worker_execution_environment": "EOS_tgz_transferred_and_unpacked_in_Condor_scratch",
        },
        "inputs": {
            "dataset_list": str(dataset_list),
            "dataset_list_sha256": sha256_file(dataset_list),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "goodrun_json": str(goodrun),
            "goodrun_json_sha256": sha256_file(goodrun),
            "reused_2024_input_list": str(old_input_list),
            "reused_2024_input_list_sha256": sha256_file(old_input_list),
            "python_env": str(python_env),
            "python_env_sha256": sha256_file(python_env),
        },
        "counts": {
            "data_dataset_keys": len(dataset_names),
            "data_root_files": len(records),
            "data_shards": len(args_lines),
            "files_per_shard": args.files_per_shard,
            "duplicate_data_files": len(duplicates),
            "mc_jobs": 0,
            "reused_2024_mc_root_files": len(reused_mc),
            "excluded_2024_data_root_files": excluded_2024_data,
        },
        "record_digest": sha256_bytes("".join(shard_digests).encode()),
        "paths": {
            "campaign": str(campaign),
            "shards": str(campaign / "shards"),
            "outputs": str(campaign / "outputs"),
            "logs": str(campaign / "logs"),
            "runtime": str(campaign / "runtime"),
            "arguments": str(campaign / "condor" / "arguments.txt"),
            "submit": str(campaign / "condor" / "flat_2025_data.sub"),
            "wrapper": str(campaign / "condor" / "run_flat_2025_data.sh"),
            "proxy": str(proxy),
            "python_env": str(python_env),
            "reused_2024_mc_inputs": str(campaign / "reused_2024_mc_inputs.txt"),
        },
        "submission": {
            "ready": proxy.is_file(),
            "proxy_exists": proxy.is_file(),
            "command": f"condor_submit {campaign / 'condor' / 'flat_2025_data.sub'}",
        },
    }
    write_json(campaign / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

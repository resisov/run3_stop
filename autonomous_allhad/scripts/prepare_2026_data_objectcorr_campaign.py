#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autonomous_allhad.object_corrections_2025 import (
    PAYLOADS,
    manifest as correction_manifest_2025,
    sha256,
    validate_payloads,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_CAMPAIGN = DEFAULT_REPO / "autonomous_allhad/workflow/flat2026"
DATA_YEAR = "2026"
CORRECTION_YEAR = "2025"
BLOCKED_REASON = (
    "2026 Condor preparation is blocked: the 2025 data JEC run-binned payload does not "
    "accept 2026 run numbers, and substituting a 2025 proxy run is explicitly forbidden."
)
EXPECTED_INPUTS = 14295
EXPECTED_INPUT_SHA256 = "2f171733ba3b4684139fd870db212124f90a4ff569fe161475b38cefa005fdae"
EXPECTED_DATASETS = 32
SELECTED_INPUTS = 7018
SELECTED_DATASETS = 16
LUMINOSITY_FB = 25.31
PRIMARY_DATASETS = (
    "JetMET0", "JetMET1", "EGamma0", "EGamma1", "EGamma2", "EGamma3", "Muon0", "Muon1",
)
ERAS = ("A", "B", "C", "D")
SELECTED_ERAS = ("B", "D")
LUMIMASK_NAME = "Cert_Collisions2026_401624_403937_golden.json"
LUMIMASK_SHA256 = "5c16911a0a03735d21c99f470afa737da0e20c6f213e3beebc79f29e02dcbd6f"
LUMIMASK_SOURCE = (
    Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions26/latest")
    / LUMIMASK_NAME
)
DATASET_RE = re.compile(
    r"^/(?P<primary>JetMET[01]|EGamma[0-3]|Muon[01])/Run2026(?P<era>[A-D])-PromptReco-v1/NANOAOD$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == rendered:
        return
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(rendered)
    os.replace(temporary, path)


def require_eos(path: Path, label: str) -> None:
    if not str(path).startswith("/eos/user/"):
        raise RuntimeError(f"{label} must be an EOS path: {path}")


def process_group(primary: str) -> str:
    if primary.startswith("JetMET"):
        return "JetMET"
    if primary.startswith("EGamma"):
        return "EGamma"
    if primary.startswith("Muon"):
        return "Muon"
    raise RuntimeError(f"unrecognized 2026 data primary dataset: {primary}")


def normalize_lfn(path: str) -> str:
    if path.startswith("root://"):
        return path
    if path.startswith("/store/"):
        return "root://cms-xrd-global.cern.ch/" + path
    raise RuntimeError(f"2026 frozen input is not an LFN or XRootD URL: {path}")


def parse_inputs(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = sha256(path)
    if digest != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"2026 input-list checksum drift: expected {EXPECTED_INPUT_SHA256}, got {digest}"
        )
    records: list[dict[str, Any]] = []
    seen_lfns: set[str] = set()
    dataset_indices: Counter[str] = Counter()
    stream_counts: Counter[str] = Counter()
    era_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    full_stream_counts: Counter[str] = Counter()
    full_era_counts: Counter[str] = Counter()
    full_dataset_counts: Counter[str] = Counter()
    excluded_era_counts: Counter[str] = Counter()
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        pieces = raw.split("\t")
        if len(pieces) != 2:
            raise RuntimeError(f"input line {line_number} is not dataset<TAB>LFN")
        dataset, lfn = (item.strip() for item in pieces)
        match = DATASET_RE.fullmatch(dataset)
        if match is None:
            raise RuntimeError(f"input line {line_number} has an unexpected dataset: {dataset}")
        primary = match.group("primary")
        era = match.group("era")
        expected_fragment = f"/store/data/Run2026{era}/{primary}/NANOAOD/"
        if not lfn.startswith(expected_fragment) or not lfn.endswith(".root"):
            raise RuntimeError(
                f"input line {line_number} LFN does not match its dataset: {lfn}"
            )
        if lfn in seen_lfns:
            raise RuntimeError(f"duplicate 2026 LFN at line {line_number}: {lfn}")
        seen_lfns.add(lfn)
        full_stream_counts[primary] += 1
        full_era_counts[f"Run2026{era}"] += 1
        full_dataset_counts[dataset] += 1
        if era not in SELECTED_ERAS:
            excluded_era_counts[f"Run2026{era}"] += 1
            continue
        file_index = dataset_indices[dataset]
        dataset_indices[dataset] += 1
        stream_counts[primary] += 1
        era_counts[f"Run2026{era}"] += 1
        dataset_counts[dataset] += 1
        records.append({
            "sample_name": dataset,
            "dataset": dataset,
            "process_group": process_group(primary),
            "year": DATA_YEAR,
            "file_index": file_index,
            "file_path": normalize_lfn(lfn),
            "lfn": lfn,
            "primary_dataset": primary,
            "run_era": f"Run2026{era}",
            "xsec_pb": None,
            "is_data": True,
            "is_background": False,
            "is_signal": False,
            "sumw_source": "data_unweighted",
        })
    expected_datasets = {
        f"/{primary}/Run2026{era}-PromptReco-v1/NANOAOD"
        for primary in PRIMARY_DATASETS
        for era in ERAS
    }
    actual_full_datasets = set(full_dataset_counts)
    selected_datasets = {
        f"/{primary}/Run2026{era}-PromptReco-v1/NANOAOD"
        for primary in PRIMARY_DATASETS
        for era in SELECTED_ERAS
    }
    actual_selected_datasets = set(dataset_counts)
    if len(seen_lfns) != EXPECTED_INPUTS:
        raise RuntimeError(f"expected {EXPECTED_INPUTS} frozen 2026 inputs, got {len(seen_lfns)}")
    if len(records) != SELECTED_INPUTS:
        raise RuntimeError(f"expected {SELECTED_INPUTS} Run2026B/D inputs, got {len(records)}")
    if len(actual_full_datasets) != EXPECTED_DATASETS or actual_full_datasets != expected_datasets:
        raise RuntimeError(
            "2026 dataset coverage mismatch: "
            f"missing={sorted(expected_datasets - actual_full_datasets)}, "
            f"unexpected={sorted(actual_full_datasets - expected_datasets)}"
        )
    if len(actual_selected_datasets) != SELECTED_DATASETS or actual_selected_datasets != selected_datasets:
        raise RuntimeError(
            "selected Run2026B/D dataset coverage mismatch: "
            f"missing={sorted(selected_datasets - actual_selected_datasets)}, "
            f"unexpected={sorted(actual_selected_datasets - selected_datasets)}"
        )
    return records, {
        "source": "frozen_dataset_tab_lfn_list",
        "path": str(path),
        "format": "dataset<TAB>LFN",
        "sha256": digest,
        "frozen_records": len(seen_lfns),
        "selected_records": len(records),
        "selected_eras": [f"Run2026{era}" for era in SELECTED_ERAS],
        "excluded_eras": [f"Run2026{era}" for era in ERAS if era not in SELECTED_ERAS],
        "excluded_era_file_counts": dict(sorted(excluded_era_counts.items())),
        "unique_lfns": len(seen_lfns),
        "duplicate_lfns": 0,
        "frozen_datasets": len(actual_full_datasets),
        "selected_datasets": len(actual_selected_datasets),
        "files_by_stream": dict(sorted(stream_counts.items())),
        "files_by_era": dict(sorted(era_counts.items())),
        "files_by_dataset": dict(sorted(dataset_counts.items())),
        "frozen_files_by_stream": dict(sorted(full_stream_counts.items())),
        "frozen_files_by_era": dict(sorted(full_era_counts.items())),
        "role": "data_only",
    }


def record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]


def make_shards(
    campaign: Path, records: list[dict[str, Any]], files_per_shard: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle_path = campaign / "bundles/shards_2026_data.tgz"
    shards: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="flat2026_data_shards_") as temporary:
        shard_dir = Path(temporary) / "shards"
        shard_dir.mkdir(parents=True)
        for index, start in enumerate(range(0, len(records), files_per_shard)):
            selected = records[start : start + files_per_shard]
            name = f"data_2026_shard_{index:05d}"
            path = shard_dir / f"{name}.json"
            digest = record_digest(selected)
            atomic_json(path, {
                "schema_version": "full_production_shard_spec_v8_2026_data_2025corr",
                "shard_id": name,
                "record_digest": digest,
                "record_group": "data",
                "data_year": DATA_YEAR,
                "correction_year": CORRECTION_YEAR,
                "records_per_shard": files_per_shard,
                "records": selected,
            })
            shards.append({
                "name": name,
                "basename": path.name,
                "records": len(selected),
                "record_digest": digest,
            })
        with tarfile.open(bundle_path, "w:gz") as archive:
            archive.add(shard_dir, arcname="shards", recursive=True)
    return shards, {
        "path": str(bundle_path),
        "size": bundle_path.stat().st_size,
        "sha256": sha256(bundle_path),
        "files": len(shards),
        "records": sum(item["records"] for item in shards),
    }


def build_worker_bundle(repo: Path, destination: Path) -> dict[str, Any]:
    package = repo / "autonomous_allhad/autonomous_allhad"
    files = sorted(path for path in package.glob("*.py") if path.is_file())
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"autonomous_allhad/{path.name}", recursive=False)
    return {
        "path": str(destination),
        "files": [path.name for path in files],
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def build_payload_bundle(destination: Path) -> dict[str, Any]:
    if sha256(LUMIMASK_SOURCE) != LUMIMASK_SHA256:
        raise RuntimeError("official 2026 golden JSON checksum drift")
    with tarfile.open(destination, "w:gz") as archive:
        for payload in PAYLOADS:
            archive.add(payload.source, arcname=str(payload.relative), recursive=False)
            if payload.filename in {"jetid.json.gz", "jetvetomaps.json.gz"}:
                archive.add(
                    payload.source,
                    arcname=f"analysis/data/JMESF/{CORRECTION_YEAR}/{payload.filename}",
                    recursive=False,
                )
        archive.add(
            LUMIMASK_SOURCE,
            arcname=f"analysis/data/lumiMask/{LUMIMASK_NAME}",
            recursive=False,
        )
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "correction_payload_year": CORRECTION_YEAR,
        "lumimask_year": DATA_YEAR,
        "selected_run_eras": [f"Run2026{era}" for era in SELECTED_ERAS],
        "excluded_run_eras": {
            "Run2026A": "not selected by analysis policy",
            "Run2026C": "low-pileup data; excluded from the standard 2026 analysis campaign",
        },
        "luminosity_fb": LUMINOSITY_FB,
        "lumimask_name": LUMIMASK_NAME,
        "lumimask_sha256": LUMIMASK_SHA256,
    }


def proxy_status(proxy: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["voms-proxy-info", "-file", str(proxy), "-timeleft"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    timeleft = int(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip().isdigit() else 0
    return {
        "path": str(proxy),
        "exists": proxy.is_file(),
        "size": proxy.stat().st_size if proxy.is_file() else 0,
        "sha256": sha256(proxy) if proxy.is_file() else None,
        "timeleft_seconds_at_preparation": timeleft,
        "status": "valid" if timeleft >= 3600 else "expired_or_too_short",
    }


def wrapper_text(proxy_name: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
NAME="${{1:?missing name}}"
SHARD="${{2:?missing shard basename}}"
DEST="${{3:?missing EOS destination}}"
WORKDIR="${{_CONDOR_SCRATCH_DIR:-$PWD}}"
cd "$WORKDIR"
case "$DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS destination: $DEST" >&2; exit 64 ;;
esac
mkdir -p runtime_home runtime_tmp runtime_cache runtime_xrd fragments
export HOME="$WORKDIR/runtime_home"
export TMPDIR="$WORKDIR/runtime_tmp"
export XDG_CACHE_HOME="$WORKDIR/runtime_cache"
export NUMBA_CACHE_DIR="$WORKDIR/runtime_cache/numba"
export AUTONOMOUS_ALLHAD_XRD_CACHE="$WORKDIR/runtime_xrd"
export AUTONOMOUS_ALLHAD_FRAGMENT_DIR="$WORKDIR/fragments"
export AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4 XRD_REQUESTTIMEOUT=180 XRD_REDIRECTLIMIT=10
export X509_USER_PROXY="$WORKDIR/{proxy_name}"
chmod 600 "$X509_USER_PROXY"
tar -xzf py38.tgz
tar -xzf objectcorr_2026_data_worker.tgz
tar -xzf objectcorr_2026_data_payloads.tgz
tar -xzf shards_2026_data.tgz
PY="$WORKDIR/bin/python3"
[ -x "$PY" ] || PY="$WORKDIR/bin/python"
[ -x "$PY" ] || PY="$WORKDIR/py38/bin/python"
test -x "$PY"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$WORKDIR/lib:$WORKDIR/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$WORKDIR"
"$PY" -u -m autonomous_allhad.intermediate_2026_data_worker \
  --repo "$WORKDIR" --shard "$WORKDIR/shards/$SHARD" \
  --output "$WORKDIR/out.root" --metadata-output "$WORKDIR/out.json" \
  --shift nominal --record-workers 4
test -s out.root
test -s out.json
export DEST
"$PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); assert d.get("status")=="complete", d.get("status"); assert d.get("schema_version")=="flat_ntuple_shard_v8_float32_fullselection_2026_data_2025corr"; assert not d.get("bad_files"), len(d.get("bad_files",[])); d["root_file"]=os.environ["DEST"]; p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
ROOT_URL="root://eosuser.cern.ch/$DEST"
META_DEST="${{DEST%.root}}.json"
META_URL="root://eosuser.cern.ch/$META_DEST"
XRDCOPY="$(command -v xrdcp)"
XRDFS="$(command -v xrdfs)"
staged=0
for attempt in 1 2 3 4 5; do
  if "$XRDCOPY" -f --nopbar --streams 4 out.root "$ROOT_URL" &&
     "$XRDCOPY" -f --nopbar out.json "$META_URL" &&
     "$XRDFS" eosuser.cern.ch stat "$DEST" >/dev/null &&
     "$XRDFS" eosuser.cern.ch stat "$META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "$((attempt * 10))"
done
test "$staged" -eq 1
echo "completed $NAME nominal"
'''


def submit_text(
    wrapper: Path,
    arguments: Path,
    logs: Path,
    py38: Path,
    worker_bundle: Path,
    payload_bundle: Path,
    shard_bundle: Path,
    proxy: Path,
) -> str:
    return f'''universe = vanilla
executable = {wrapper}
arguments = $(name) $(shard_name) $(root_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker_bundle}, {payload_bundle}, {shard_bundle}, {proxy}
transfer_output_files = ""
output = {logs}/$(name).out
error = {logs}/$(name).err
log = {logs}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 14000MB
+JobFlavour = "workday"
queue name,shard_name,root_out from {arguments}
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, but do not submit, the 2026 data campaign using 2025 corrections."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--input-list", type=Path)
    parser.add_argument("--files-per-shard", type=int, default=10)
    args = parser.parse_args(argv)
    raise RuntimeError(BLOCKED_REASON)
    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    input_list = (args.input_list or campaign / "inputs.txt").absolute()
    for path, label in ((repo, "repo"), (campaign, "campaign"), (input_list, "input list")):
        require_eos(path, label)
    if args.files_per_shard < 1:
        raise RuntimeError("files per shard must be positive")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    for path, label in ((py38, "python bundle"), (proxy, "proxy"), (input_list, "input list")):
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")
    proxy_info = proxy_status(proxy)
    if proxy_info["status"] != "valid":
        raise RuntimeError("proxy is expired or has less than one hour remaining")
    if not LUMIMASK_SOURCE.is_file():
        raise RuntimeError(f"missing official 2026 golden JSON: {LUMIMASK_SOURCE}")

    payload_validation = validate_payloads(repo)
    if payload_validation["status"] != "valid":
        raise RuntimeError("2025 payload validation failed: " + "; ".join(payload_validation["errors"]))
    records, input_summary = parse_inputs(input_list)
    if any(not item["is_data"] or item["is_background"] or item["is_signal"] for item in records):
        raise RuntimeError("2026 campaign preparation received a non-data record")

    campaign.mkdir(parents=True, exist_ok=True)
    for name in ("bundles", "condor", "logs", "outputs/nominal", "reports"):
        (campaign / name).mkdir(parents=True, exist_ok=True)
    atomic_json(campaign / "inputs_2026_data.json", {
        "schema_version": "flat2026_frozen_data_inputs_v1",
        "created_at": utc_now(),
        "summary": input_summary,
        "records": records,
    })
    shards, shard_bundle = make_shards(campaign, records, args.files_per_shard)
    worker_bundle_path = campaign / "bundles/objectcorr_2026_data_worker.tgz"
    payload_bundle_path = campaign / "bundles/objectcorr_2026_data_payloads.tgz"
    worker_bundle = build_worker_bundle(repo, worker_bundle_path)
    if "intermediate_2026_data_worker.py" not in worker_bundle["files"]:
        raise RuntimeError("2026 worker is missing from the worker bundle")
    payload_bundle = build_payload_bundle(payload_bundle_path)

    wrapper = campaign / "condor/run_intermediate_2026_data.sh"
    wrapper.write_text(wrapper_text(proxy.name))
    wrapper.chmod(0o755)
    arguments = campaign / "condor/arguments_2026_data.txt"
    arguments.write_text("\n".join(
        " ".join((
            shard["name"],
            shard["basename"],
            str(campaign / "outputs/nominal" / f"{shard['name']}.root"),
        ))
        for shard in shards
    ) + "\n")
    submit = campaign / "condor/submit_2026_data.sub"
    submit.write_text(submit_text(
        wrapper,
        arguments,
        campaign / "logs",
        py38,
        worker_bundle_path,
        payload_bundle_path,
        Path(shard_bundle["path"]),
        proxy,
    ))

    source_discovery = campaign / "manifest.json"
    source_discovery_sha = sha256(source_discovery) if source_discovery.is_file() else None
    manifest = {
        "schema_version": "intermediate_2026_data_campaign_v8_2025corr",
        "status": "prepared_not_submitted",
        "created_at": utc_now(),
        "data_year": DATA_YEAR,
        "selection_year": CORRECTION_YEAR,
        "correction_year": CORRECTION_YEAR,
        "scale_factor_year": CORRECTION_YEAR,
        "lumimask_year": DATA_YEAR,
        "campaign_scope": "data_only",
        "repo": str(repo),
        "campaign": str(campaign),
        "source_discovery_manifest": {
            "path": str(source_discovery),
            "sha256": source_discovery_sha,
            "preserved": True,
        },
        "policy": {
            "decision": "Treat 2026 as an extension of the 2025 analysis correction/SF era.",
            "data_identity_preserved": True,
            "golden_json_alias": False,
            "correction_and_sf_alias": {DATA_YEAR: CORRECTION_YEAR},
            "mc_in_campaign": False,
            "mc_scale_factors_applied_to_data": False,
            "luminosity_value_required_for_skim": False,
            "luminosity_value_status": "provided_for_selected_Run2026B_and_Run2026D",
        },
        "inputs": input_summary,
        "shards": {
            "total": len(shards),
            "records": sum(item["records"] for item in shards),
            "files_per_shard": args.files_per_shard,
            "last_shard_records": shards[-1]["records"],
        },
        "jobs": len(shards),
        "requested_shifts": ["nominal"],
        "payload_validation_2025": payload_validation,
        "correction_manifest_2025": correction_manifest_2025(),
        "lumimask": {
            "source": str(LUMIMASK_SOURCE),
            "filename": LUMIMASK_NAME,
            "sha256": LUMIMASK_SHA256,
            "status": "validated_2026_golden_json",
        },
        "worker_bundle": worker_bundle,
        "payload_bundle": payload_bundle,
        "shard_bundle": shard_bundle,
        "python_bundle": {"path": str(py38), "size": py38.stat().st_size, "sha256": sha256(py38)},
        "proxy": proxy_info,
        "submission": {
            "performed": False,
            "submit_file": str(submit),
            "arguments": str(arguments),
            "command": f"condor_submit {submit}",
        },
        "validation": {
            "data_only": True,
            "unique_inputs": len(records) == len({item["lfn"] for item in records}),
            "expected_frozen_input_count": input_summary["frozen_records"] == EXPECTED_INPUTS,
            "expected_selected_input_count": len(records) == SELECTED_INPUTS,
            "expected_input_checksum": input_summary["sha256"] == EXPECTED_INPUT_SHA256,
            "expected_frozen_dataset_count": input_summary["frozen_datasets"] == EXPECTED_DATASETS,
            "expected_selected_dataset_count": input_summary["selected_datasets"] == SELECTED_DATASETS,
            "shard_record_conservation": sum(item["records"] for item in shards) == len(records),
            "job_shard_bijection": len(shards) == len(arguments.read_text().splitlines()),
            "2025_payloads_valid": payload_validation["status"] == "valid",
            "2026_lumimask_valid": sha256(LUMIMASK_SOURCE) == LUMIMASK_SHA256,
            "proxy_valid_at_preparation": proxy_info["status"] == "valid",
        },
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"campaign validation failed: {manifest['validation']}")
    atomic_json(campaign / "campaign_manifest.json", manifest)
    (campaign / "reports/readiness_2026_data.md").write_text(
        "# 2026 Data Condor Readiness\n\n"
        "- Status: prepared, not submitted\n"
        f"- Frozen inputs audited: {input_summary['frozen_records']:,} unique ROOT files in {input_summary['frozen_datasets']} datasets\n"
        f"- Selected inputs: {len(records):,} Run2026B/D ROOT files in {input_summary['selected_datasets']} datasets\n"
        f"- Data shards/jobs: {len(shards):,} ({args.files_per_shard} files maximum)\n"
        f"- Certified luminosity (Run2026B+D): {LUMINOSITY_FB:.2f} fb^-1\n"
        "- Data/selection/correction years: 2026 / 2025 / 2025\n"
        f"- Golden JSON: {LUMIMASK_NAME} ({LUMIMASK_SHA256})\n"
        "- MC in campaign: no; MC scale factors on data: no\n"
        "- Submission performed: no\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coffea.util import load as coffea_load

from autonomous_allhad.object_corrections_2024 import (
    EXTERNAL_FINAL_WEIGHT_DEPENDENCIES,
    PAYLOADS,
    SHAPE_VARIATIONS,
    manifest as correction_manifest,
    sha256,
    validate_payloads,
    validate_shift,
)


DEFAULT_REPO = Path("/eos/user/t/taiwoo/run3_stop/decaf")
DEFAULT_CAMPAIGN = DEFAULT_REPO / "autonomous_allhad/workflow/intermediate_2024_fullselection_v2_20260723"
DATA_GROUPS = {"JetMET", "EGamma", "Muon"}


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
    raw = str(path)
    if raw.startswith("/afs/") or raw.startswith("/tmp/"):
        raise RuntimeError(f"{label} must not use AFS or /tmp: {path}")
    if not raw.startswith(("/eos/user/",)):
        raise RuntimeError(f"{label} must be an EOS path: {path}")


def process_group(dataset: str) -> str:
    patterns = (
        ("JetMET", "JetMET"), ("EGamma", "EGamma"), ("Muon", "Muon"),
        ("TT", "TT"), ("Zto2Nu", "Zto2Nu"), ("Wto", "WtoLNu"),
        ("QCD", "QCD"), ("GJets", "GJ"), ("GJ", "GJ"), ("DY", "DY"),
        ("TW", "ST"), ("TbarW", "ST"), ("TBbar", "ST"), ("TbarB", "ST"),
        ("WW", "VV"), ("WZ", "VV"), ("ZZ", "VV"), ("SMS", "SMS"),
    )
    for token, group in patterns:
        if token in dataset:
            return group
    return "other"


def dataset_files(metadata: Any) -> list[str]:
    if isinstance(metadata, dict):
        files = metadata.get("files", [])
    elif isinstance(metadata, list):
        files = metadata
    else:
        files = []
    if isinstance(files, dict):
        files = list(files.values())
    return [str(item) for item in files] if isinstance(files, list) else []


def normalize_lfn(path: str) -> str:
    if path.startswith("root://"):
        return path
    if path.startswith("/store/"):
        return "root://cms-xrd-global.cern.ch/" + path
    return path


def bad_file_paths(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text())
    entries = payload if isinstance(payload, list) else payload.get("bad_files", payload.get("files", []))
    out = set()
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict) or item.get("permanently_skipped") is False:
            continue
        raw = item.get("file_path") or item.get("physical_file_path")
        if raw:
            out.add(normalize_lfn(str(raw)))
    return out


def input_records(repo: Path, bad_paths: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = repo / "analysis/metadata/KNU_2024_v4.json.gz"
    datasets_path = repo / "analysis/datasets/datasets_2024.txt"
    with gzip.open(metadata_path, "rt") as source:
        metadata = json.load(source)
    datasets = [
        line.strip() for line in datasets_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    records = []
    excluded_bad = []
    excluded_signal = 0
    excluded_other = []
    missing_metadata = []
    for dataset in datasets:
        if dataset not in metadata:
            missing_metadata.append(dataset)
            continue
        group = process_group(dataset)
        files = dataset_files(metadata[dataset])
        if group == "SMS":
            excluded_signal += len(files)
            continue
        if group == "other":
            excluded_other.append({"dataset": dataset, "files": len(files)})
            continue
        xsec = metadata[dataset].get("xs") if isinstance(metadata[dataset], dict) else None
        for index, raw_path in enumerate(files):
            file_path = normalize_lfn(raw_path)
            if file_path in bad_paths:
                excluded_bad.append({"dataset": dataset, "file_path": file_path})
                continue
            is_data = group in DATA_GROUPS
            records.append({
                "sample_name": dataset,
                "dataset": dataset,
                "process_group": group,
                "year": "2024",
                "file_index": index,
                "file_path": file_path,
                "xsec_pb": xsec,
                "is_data": is_data,
                "is_background": not is_data,
                "is_signal": False,
                "sumw_source": "data_unweighted" if is_data else "Runs.genEventSumw preferred; Events.genWeight fallback",
            })
    signal_manifest_path = repo / "autonomous_allhad/workflow/full_signal_manifest.json"
    signal_manifest = json.loads(signal_manifest_path.read_text())
    signal_records = []
    skipped_fullsim = 0
    for item in signal_manifest.get("records", []):
        if str(item.get("simulation_type") or "") != "FastSim signal dataset":
            skipped_fullsim += 1
            continue
        file_path = normalize_lfn(str(item.get("file_path") or ""))
        if not file_path:
            continue
        signal_records.append({
            **item,
            "sample_name": str(item.get("sample_name") or item.get("dataset") or "SMS"),
            "dataset": str(item.get("dataset") or item.get("sample_name") or "SMS"),
            "process_group": "SMS",
            "year": "2024",
            "file_path": file_path,
            "xsec_pb": None,
            "is_data": False,
            "is_background": False,
            "is_signal": True,
            "simulation_type": "FastSim signal dataset",
            "sumw_source": "Runs.genEventSumw_T2tt_<mStop>_<mLSP>",
        })
    records.extend(signal_records)
    return records, {
        "metadata": str(metadata_path),
        "datasets": str(datasets_path),
        "configured_datasets": len(datasets),
        "records": len(records),
        "data_records": sum(bool(item["is_data"]) for item in records),
        "background_records": sum(bool(item["is_background"]) for item in records),
        "fastsim_signal_records": sum(bool(item["is_signal"]) for item in records),
        "signal_manifest": str(signal_manifest_path),
        "fullsim_signal_records_recorded_not_processed": skipped_fullsim,
        "excluded_bad_files": excluded_bad,
        "excluded_fastsim_signal_records": excluded_signal,
        "excluded_unclassified_datasets": excluded_other,
        "missing_metadata": missing_metadata,
    }


def record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]


def make_shards(
    campaign: Path,
    records: list[dict[str, Any]],
    data_size: int,
    mc_size: int,
    signal_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shards = []
    bundle_path = campaign / "bundles/fullselection_shards.tgz"
    groups = (
        ("data", [item for item in records if item["is_data"]], data_size),
        ("mc", [item for item in records if item["is_background"]], mc_size),
        ("signal", [item for item in records if item["is_signal"]], signal_size),
    )
    with tempfile.TemporaryDirectory(prefix="fullselection_shards_") as temporary:
        shard_dir = Path(temporary) / "shards"
        shard_dir.mkdir(parents=True)
        for group_name, selected, size in groups:
            for index, start in enumerate(range(0, len(selected), size)):
                chunk = selected[start : start + size]
                name = f"{group_name}_shard_{index:05d}"
                path = shard_dir / f"{name}.json"
                digest = record_digest(chunk)
                atomic_json(path, {
                    "schema_version": "full_production_shard_spec_v5_fullselection_2024",
                    "shard_id": name,
                    "record_digest": digest,
                    "record_group": group_name,
                    "records_per_shard": size,
                    "records": chunk,
                })
                shards.append({
                    "name": name,
                    "group": group_name,
                    "basename": path.name,
                    "records": len(chunk),
                    "record_digest": digest,
                })
        with tarfile.open(bundle_path, "w:gz") as archive:
            archive.add(shard_dir, arcname="shards", recursive=True)
    return shards, {
        "path": str(bundle_path),
        "size": bundle_path.stat().st_size,
        "sha256": sha256(bundle_path),
        "files": len(shards),
        "layout": "shards/<group>_shard_<index>.json",
    }


def validate_btag_efficiency(path: Path) -> dict[str, Any]:
    item = {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        item["status"] = "missing"
        return item
    item["sha256"] = sha256(path)
    try:
        payload = coffea_load(str(path))
        taggers = sorted(str(key) for key in payload.keys())
        upart = payload.get("UParTAK4")
        processes = sorted(str(key) for key in upart.keys()) if hasattr(upart, "keys") else []
    except Exception as exc:
        item["status"] = "unreadable"
        item["error"] = f"{type(exc).__name__}: {exc}"
    else:
        item["taggers"] = taggers
        item["upart_processes"] = processes
        item["status"] = "valid" if "UParTAK4" in taggers else "missing_UParTAK4"
    return item


def btag_efficiency_coverage(records: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    payload_keys = set(validation.get("upart_processes", []))
    record_counts = Counter(
        str(item["dataset"]).split("____", 1)[0]
        for item in records
        if item.get("is_background")
    )
    missing = sorted(set(record_counts) - payload_keys)
    return {
        "status": "complete" if not missing else "incomplete",
        "physical_dataset_policy": "strip metadata split suffix beginning with ____",
        "background_records": sum(record_counts.values()),
        "unique_physical_background_datasets": len(record_counts),
        "matched_unique_physical_datasets": len(set(record_counts) & payload_keys),
        "missing_unique_physical_datasets": missing,
        "missing_records": sum(record_counts[name] for name in missing),
        "payload_only_keys": sorted(payload_keys - set(record_counts)),
    }


def build_payload_bundle(repo: Path, destination: Path, btag_path: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for payload in PAYLOADS:
            archive.add(payload.source, arcname=str(payload.relative), recursive=False)
            if payload.filename in {"jetid.json.gz", "jetvetomaps.json.gz"}:
                archive.add(payload.source, arcname=f"analysis/data/JMESF/2024/{payload.filename}", recursive=False)
        lumimask = repo / "analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json"
        archive.add(lumimask, arcname="analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json", recursive=False)
        archive.add(btag_path, arcname="payloads/BTV/btageff2024.merged", recursive=False)
        archive.add(btag_path, arcname="analysis/hists/btageff2024.merged", recursive=False)
    return {"path": str(destination), "size": destination.stat().st_size, "sha256": sha256(destination)}


def build_worker_bundle(repo: Path, destination: Path) -> dict[str, Any]:
    package = repo / "autonomous_allhad/autonomous_allhad"
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def wrapper_text(proxy_name: str) -> str:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
NAME="@D@{{1:?missing name}}"
SHIFT="@D@{{2:?missing shift}}"
SHARD="@D@{{3:?missing shard basename}}"
DEST="@D@{{4:?missing EOS destination}}"
WORKDIR="@D@{{_CONDOR_SCRATCH_DIR:-@D@PWD}}"
cd "@D@WORKDIR"
case "@D@DEST" in
  /eos/user/t/taiwoo/*) ;;
  *) echo "refusing non-EOS destination: @D@DEST" >&2; exit 64 ;;
esac
mkdir -p runtime_home runtime_tmp runtime_cache runtime_mplconfig runtime_xrd fragments
export HOME="@D@WORKDIR/runtime_home"
export TMPDIR="@D@WORKDIR/runtime_tmp"
export TMP="@D@TMPDIR"
export TEMP="@D@TMPDIR"
export XDG_CACHE_HOME="@D@WORKDIR/runtime_cache"
export MPLCONFIGDIR="@D@WORKDIR/runtime_mplconfig"
export NUMBA_CACHE_DIR="@D@WORKDIR/runtime_cache/numba"
export PYTHONPYCACHEPREFIX="@D@WORKDIR/runtime_cache/pycache"
export AUTONOMOUS_ALLHAD_XRD_CACHE="@D@WORKDIR/runtime_xrd"
export AUTONOMOUS_ALLHAD_FRAGMENT_DIR="@D@WORKDIR/fragments"
export AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XRD_NETWORKSTACK=IPv4
export XRD_REQUESTTIMEOUT=180
export XRD_REDIRECTLIMIT=10
export X509_USER_PROXY="@D@WORKDIR/{proxy_name}"
chmod 600 "@D@X509_USER_PROXY"
tar -xzf py38.tgz
tar -xzf objectcorr_2024_worker.tgz
tar -xzf objectcorr_2024_payloads.tgz
tar -xzf fullselection_shards.tgz
PY="@D@WORKDIR/bin/python3"
[ -x "@D@PY" ] || PY="@D@WORKDIR/bin/python"
[ -x "@D@PY" ] || PY="@D@WORKDIR/py38/bin/python"
test -x "@D@PY"
export PATH="@D@(dirname "@D@PY"):@D@PATH"
export LD_LIBRARY_PATH="@D@WORKDIR/lib:@D@WORKDIR/py38/lib:@D@{{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="@D@WORKDIR"
"@D@PY" -u -m autonomous_allhad.intermediate_2024_worker \
  --repo "@D@WORKDIR" --shard "@D@WORKDIR/shards/@D@SHARD" \
  --output "@D@WORKDIR/out.root" --metadata-output "@D@WORKDIR/out.json" \
  --shift "@D@SHIFT" --record-workers 4
test -s out.root
test -s out.json
ROOT_URL="root://eosuser.cern.ch/@D@DEST"
META_DEST="@D@{{DEST%.root}}.json"
META_URL="root://eosuser.cern.ch/@D@META_DEST"
XRDCOPY="@D@(command -v xrdcp)"
XRDFS="@D@(command -v xrdfs)"
export DEST
"@D@PY" -c 'import json,os,pathlib; p=pathlib.Path("out.json"); d=json.loads(p.read_text()); d["root_file"]=os.environ["DEST"]; p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\\n")'
staged=0
for attempt in 1 2 3 4 5; do
  if "@D@XRDCOPY" -f --nopbar --streams 4 out.root "@D@ROOT_URL" &&
     "@D@XRDCOPY" -f --nopbar out.json "@D@META_URL" &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@DEST" >/dev/null &&
     "@D@XRDFS" eosuser.cern.ch stat "@D@META_DEST" >/dev/null; then
    staged=1
    break
  fi
  sleep "@D@((attempt * 10))"
done
test "@D@staged" -eq 1
echo "completed @D@NAME @D@SHIFT"
"""
    return text.replace("@D@", "$")


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
    return f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shift) $(shard_name) $(root_out)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = {py38}, {worker_bundle}, {payload_bundle}, {shard_bundle}, {proxy}
transfer_output_files = ""
output = {logs}/$(name)_$(shift).out
error = {logs}/$(name)_$(shift).err
log = {logs}/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 14000MB
+JobFlavour = "workday"
queue name,shift,shard_name,root_out from {arguments}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the corrected 2024 full-selection DATA, background, and FastSim intermediate ROOT campaign.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--data-shard-size", type=int, default=2)
    parser.add_argument("--mc-shard-size", type=int, default=8)
    parser.add_argument("--signal-shard-size", type=int, default=2)
    parser.add_argument("--shifts", nargs="+", default=["nominal"])
    args = parser.parse_args(argv)
    repo = args.repo.absolute()
    campaign = args.campaign.absolute()
    require_eos(repo, "repo")
    require_eos(campaign, "campaign")
    if args.data_shard_size < 1 or args.mc_shard_size < 1 or args.signal_shard_size < 1:
        raise RuntimeError("shard sizes must be positive")
    shifts = [validate_shift(item) for item in args.shifts]
    if len(shifts) != len(set(shifts)):
        raise RuntimeError("duplicate shifts requested")

    py38 = repo / "condor/py38.tgz"
    proxy = repo / f"analysis/proxy/x509up_u{os.getuid()}"
    btag_path = repo / "analysis/hists/btageff2024.merged"
    bad_path = repo / "autonomous_allhad/workflow/bad_files.json"
    for path, label in ((py38, "python bundle"), (proxy, "proxy"), (btag_path, "b-tag efficiency")):
        require_eos(path, label)
        if not path.is_file():
            raise RuntimeError(f"missing {label}: {path}")

    payload_validation = validate_payloads(repo)
    if payload_validation["status"] != "valid":
        raise RuntimeError("official payload validation failed: " + "; ".join(payload_validation["errors"]))
    btag_validation = validate_btag_efficiency(btag_path)
    if btag_validation["status"] != "valid":
        raise RuntimeError(f"b-tag efficiency validation failed: {btag_validation}")
    records, input_summary = input_records(repo, bad_file_paths(bad_path))
    if input_summary["missing_metadata"]:
        raise RuntimeError(f"configured datasets missing metadata: {len(input_summary['missing_metadata'])}")
    if input_summary["excluded_unclassified_datasets"]:
        raise RuntimeError(f"unclassified configured datasets: {len(input_summary['excluded_unclassified_datasets'])}")
    btag_coverage = btag_efficiency_coverage(records, btag_validation)
    btag_validation["coverage"] = btag_coverage
    if btag_coverage["status"] != "complete":
        raise RuntimeError(
            "b-tag efficiency physical-dataset coverage incomplete: "
            f"{len(btag_coverage['missing_unique_physical_datasets'])} missing datasets"
        )

    campaign.mkdir(parents=True, exist_ok=True)
    for name in ("bundles", "condor", "logs", "outputs", "reports"):
        (campaign / name).mkdir(parents=True, exist_ok=True)
    shards, shard_bundle = make_shards(
        campaign,
        records,
        args.data_shard_size,
        args.mc_shard_size,
        args.signal_shard_size,
    )
    worker_bundle_path = campaign / "bundles/objectcorr_2024_worker.tgz"
    payload_bundle_path = campaign / "bundles/objectcorr_2024_payloads.tgz"
    worker_bundle = build_worker_bundle(repo, worker_bundle_path)
    payload_bundle = build_payload_bundle(repo, payload_bundle_path, btag_path)

    wrapper = campaign / "condor/run_intermediate_2024.sh"
    wrapper.write_text(wrapper_text(proxy.name))
    wrapper.chmod(0o755)
    rows = []
    for shift in shifts:
        output_dir = campaign / "outputs" / shift
        output_dir.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            if shift != "nominal" and shard["group"] == "data":
                continue
            rows.append(" ".join((shard["name"], shift, shard["basename"], str(output_dir / f"{shard['name']}.root"))))
    arguments = campaign / "condor/arguments.txt"
    arguments.write_text("\n".join(rows) + "\n")
    submit = campaign / "condor/submit.sub"
    submit.write_text(
        submit_text(
            wrapper,
            arguments,
            campaign / "logs",
            py38,
            worker_bundle_path,
            payload_bundle_path,
            Path(shard_bundle["path"]),
            proxy,
        )
    )

    campaign_manifest = {
        "schema_version": "intermediate_2024_fullselection_campaign_v6",
        "status": "prepared_not_submitted",
        "created_at": utc_now(),
        "year": 2024,
        "luminosity_fb": 109.82,
        "repo": str(repo),
        "campaign": str(campaign),
        "inputs": input_summary,
        "bad_file_manifest": str(bad_path),
        "shards": {
            "total": len(shards),
            "data": sum(item["group"] == "data" for item in shards),
            "background": sum(item["group"] == "mc" for item in shards),
            "fastsim_signal": sum(item["group"] == "signal" for item in shards),
            "data_files_per_shard": args.data_shard_size,
            "mc_files_per_shard": args.mc_shard_size,
            "signal_files_per_shard": args.signal_shard_size,
        },
        "jobs": len(rows),
        "requested_shifts": shifts,
        "supported_shifts": list(SHAPE_VARIATIONS),
        "payload_validation": payload_validation,
        "btag_efficiency": btag_validation,
        "worker_bundle": worker_bundle,
        "payload_bundle": payload_bundle,
        "shard_bundle": shard_bundle,
        "python_bundle": {"path": str(py38), "size": py38.stat().st_size, "sha256": sha256(py38)},
        "correction_manifest": correction_manifest(),
        "dy_recoil_selection": {
            "DY2E": "OS medium electrons, pT(ll)>200, on-Z, electron-cleaned AK4/AK8, opening angle against uT phi, uT>250 GeV",
            "DY2M": "OS medium muons, pT(ll)>200, on-Z, muon-cleaned AK4/AK8, opening angle against uT phi, uT>250 GeV",
        },
        "photon_selection": (
            "pT>220 GeV fiducial photon with NanoAOD medium cutBased ID (>=2), "
            "explicit Photon_electronVeto, and photon-cleaned AK4/AK8 recoil selection"
        ),
        "lowdm_selection": {
            "categories": 42,
            "nsv_policy": "inclusive; Nsv is not used in category assignment",
            "isr_subjet_bveto_policy": "diagnostic only; not applied",
            "mtb_policy": "diagnostic only; mTb<175 is not applied",
            "upstream_region_flags": [f"feature_lowdm_{name}" for name in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")],
            "upstream_region_bins": [f"lowdm_search_bin_{name}" for name in ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")],
            "bjet_pt_threshold_gev": 30.0,
            "resolved_top_veto": "not fabricated; unavailable validated Run-3 resolved-top tagger is recorded in sidecars",
        },
        "execution_policy": {
            "worker": "transferred tgz unpacked in Condor scratch",
            "eos_fuse_worker_execution": False,
            "afs_worker_execution": False,
            "literal_tmp_paths": False,
            "input_access": "NanoAOD through XRootD",
            "stageout": "xrdcp to EOS with xrdfs verification",
            "analysis_directory_modified": False,
        },
        "submission": {
            "submit_file": str(submit),
            "arguments": str(arguments),
            "command": f"condor_submit {submit}",
        },
        "intermediate_root_ready": True,
        "nominal_kinematic_corrections": correction_manifest()["nominal_kinematic_corrections"],
        "final_histogram_weight_gate": {
            "ready": False,
            "reason": "Analysis-specific external calibration payloads were not fabricated.",
            "missing": EXTERNAL_FINAL_WEIGHT_DEPENDENCIES,
        },
    }
    atomic_json(campaign / "manifest.json", campaign_manifest)
    atomic_json(campaign / "corrections_2024.json", correction_manifest())
    report = (
        "# 2024 Full-selection Intermediate ROOT Campaign\n\n"
        f"- Status: {campaign_manifest['status']}\n"
        f"- Input records: {input_summary['records']}\n"
        f"- FastSim signal records: {input_summary['fastsim_signal_records']}\n"
        f"- Shards: {len(shards)}\n"
        f"- Jobs prepared: {len(rows)}\n"
        f"- Requested shifts: {', '.join(shifts)}\n"
        f"- JEC/JER: {correction_manifest()['jec_tag']} / {correction_manifest()['jer_tag']}\n"
        "- EGM: data scale and deterministic MC scale/smearing for electrons and photons\n"
        "- Muon: official MuonScaRe scale and MC resolution correction\n"
        "- Tau: DeepTau2018v2p5 VSjet Medium TES with object-to-MET propagation\n"
        f"- B-tag efficiency: valid, {btag_coverage['matched_unique_physical_datasets']}/"
        f"{btag_coverage['unique_physical_background_datasets']} physical MC datasets covered, "
        f"SHA256 {btag_validation['sha256']}\n"
        "- GCR: explicit Photon_electronVeto, photon-cleaned AK4/AK8, and photon-uT opening angles\n"
        "- DY2E/DY2M: OS/on-Z, pT(ll)>200, channel-specific cleaned AK4/AK8 and uT direction\n"
        "- Low-dM: 42 Nsv-inclusive category branches; ISR-subjet b veto and mTb<175 are diagnostic-only; b-jet pT threshold fixed at 30 GeV\n"
        "- FastSim: all 61 recorded Run-3 signal NanoAOD files included; FullSim anchors remain excluded\n"
        "- Worker runtime: transferred tgz in Condor scratch; XRootD input and EOS xrdcp stage-out\n"
        "- analysis/ modified: no\n"
        "- Submission performed: no\n"
    )
    (campaign / "reports/readiness.md").write_text(report)
    print(json.dumps(campaign_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

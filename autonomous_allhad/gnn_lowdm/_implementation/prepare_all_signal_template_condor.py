#!/usr/bin/env python3
"""Prepare an EOS-only Condor campaign for all-signal frozen-GNN templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def require_eos(path: str) -> str:
    if not path.startswith("/eos/") or "/afs/" in path or "/tmp/" in path:
        raise ValueError(f"not an EOS-only path: {path}")
    return path.rstrip("/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker_text(campaign: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
REQUEST="${{1:?missing EOS request}}"
case "$REQUEST" in /eos/*) ;; *) exit 64 ;; esac
CAMPAIGN="{campaign}"
JOB_ID="${{ClusterId:-manual}}_${{ProcId:-0}}"
JOB_ROOT="$CAMPAIGN/runtime_jobs/$JOB_ID"
mkdir -p "$JOB_ROOT/tmp" "$JOB_ROOT/cache"
export TMPDIR="$JOB_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$JOB_ROOT/cache"
export PYTHONPYCACHEPREFIX="$JOB_ROOT/cache/pycache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH="$CAMPAIGN/runtime/py38/bin:$PATH"
export LD_LIBRARY_PATH="$CAMPAIGN/runtime/py38/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$CAMPAIGN/vendor/mt2:$CAMPAIGN/payload"
PY="$CAMPAIGN/runtime/py38/bin/python3"
test -x "$PY"
test -s "$CAMPAIGN/vendor/mt2/mt2/__init__.py"
"$PY" -c 'import mt2,numpy,awkward,uproot; assert mt2.__version__ == "1.2.0"'
"$PY" -u "$CAMPAIGN/payload/evaluate_all_signal_template_partial.py" --request "$REQUEST"
OUTPUT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$REQUEST")
test -s "$OUTPUT"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] in {"complete","complete_with_exclusions"}; assert d["test_events"]>0' "$OUTPUT"
"""


def submit_text(campaign: str, queue: str, flavour: str) -> str:
    return f"""universe = vanilla
initialdir = {campaign}/condor
executable = {campaign}/condor/run_all_signal_template.sh
arguments = $(request)
getenv = False
should_transfer_files = NO
output = {campaign}/logs/stdout.$(ClusterId).$(ProcId)
error = {campaign}/logs/stderr.$(ClusterId).$(ProcId)
log = {campaign}/logs/campaign.log
request_cpus = 4
request_memory = 12000MB
request_disk = 2000MB
+JobFlavour = "{flavour}"
+CampaignName = "gnn_lowdm_all_signal_template_20260901"
+TestPartition = "70pct"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue request from {campaign}/condor/{queue}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--eos-campaign", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-eos", required=True)
    args = parser.parse_args()
    campaign = require_eos(args.eos_campaign)
    source_eos = require_eos(args.source_eos)
    staging = args.staging
    for name in ("requests", "partials", "payload", "model", "config", "condor", "logs", "vendor", "runtime"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    payloads = (
        source_dir / "evaluate_all_signal_template_partial.py",
        source_dir.parent / "data.py",
        source_dir / "rank005_numpy.py",
    )
    for path in payloads:
        shutil.copy2(path, staging / "payload" / path.name)
    roots = sorted(args.cache.glob("signal_cache_*.root"))
    if len(roots) != 11:
        raise RuntimeError(f"expected 11 complete signal cache shards, found {len(roots)}")
    requests = []
    for local in roots:
        sidecar = json.loads(local.with_suffix(".json").read_text())
        if sidecar.get("status") != "complete" or int(sidecar.get("events_selected", 0)) <= 0:
            raise RuntimeError(f"invalid signal cache sidecar: {local}")
        name = local.stem + ".json"
        request = {
            "schema_version": "gnn_lowdm_all_signal_template_request_v1",
            "input_root": f"{source_eos}/outputs/{local.name}",
            "input_sidecar": f"{source_eos}/outputs/{local.with_suffix('.json').name}",
            "campaign_manifest": f"{campaign}/config/full_campaign_manifest_2024.json",
            "xsec": f"{campaign}/config/stop_xsec_13p6TeV.json",
            "model": f"{campaign}/model/diagonal_v3_numpy.npz",
            "configuration": f"{campaign}/config/config.json",
            "batch_size": 256,
            "output": f"{campaign}/partials/{local.stem}.json",
        }
        write_json(staging / "requests" / name, request)
        requests.append(f"{campaign}/requests/{name}")
    pilot = [requests[-1]]
    remaining = requests[:-1]
    (staging / "condor" / "pilot_queue.txt").write_text("\n".join(pilot) + "\n")
    (staging / "condor" / "remaining_queue.txt").write_text("\n".join(remaining) + "\n")
    (staging / "condor" / "all_queue.txt").write_text("\n".join(requests) + "\n")
    worker = staging / "condor" / "run_all_signal_template.sh"
    worker.write_text(worker_text(campaign))
    worker.chmod(0o755)
    (staging / "condor" / "pilot.sub").write_text(submit_text(campaign, "pilot_queue.txt", "longlunch"))
    (staging / "condor" / "remaining.sub").write_text(submit_text(campaign, "remaining_queue.txt", "workday"))
    manifest = {
        "schema_version": "gnn_lowdm_all_signal_template_condor_v1",
        "status": "prepared",
        "eos_campaign": campaign,
        "source_cache": source_eos,
        "signal_cache_shards": len(roots),
        "input_selected_events": sum(
            int(json.loads(path.with_suffix(".json").read_text())["events_selected"])
            for path in roots
        ),
        "selection": "all generated signal mass points; diagonal-v3 SR; frozen GNN; 70% test",
        "payload_sha256": {path.name: sha256(path) for path in payloads},
        "pilot": pilot,
        "remaining": remaining,
        "schedd": "bigbird24.cern.ch",
        "worker_path_policy": "all mutable/runtime/input/output paths are absolute /eos paths",
    }
    write_json(staging / "campaign_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

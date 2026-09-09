"""Single-job NanoAOD -> Events -> TROTA -> TopWTruth production."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from . import flat_ntuple_worker as flat


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _work_path(path: Path) -> Path:
    path = path.absolute()
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    in_scratch = scratch and Path(scratch).absolute() in (path, *path.parents)
    if not str(path).startswith("/eos/") and not in_scratch:
        raise ValueError(f"production files must remain on EOS or batch scratch: {path}")
    if any(part in ("tmp", "afs") for part in path.parts):
        raise ValueError(f"disallowed production path: {path}")
    return path


def _unpack_payload(bundle: Path, destination: Path) -> None:
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(f"unsupported payload archive member: {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _check_events(metadata: dict[str, Any], shard: dict[str, Any],
                  previous: dict[str, Any] | None) -> None:
    records = shard["records"]
    if (metadata.get("status") != "complete" or metadata.get("bad_files")
            or metadata.get("files_processed") != len(records)
            or metadata.get("record_digest") != shard["record_digest"]):
        raise RuntimeError("incomplete or mismatched NanoAOD production")
    actual = {record["file_path"] for record in metadata["files"]}
    if actual != {record["file_path"] for record in records}:
        raise RuntimeError("NanoAOD input coverage changed")
    if previous:
        for key in ("events_read", "events_written", "record_digest", "skim_flag", "shape_shift"):
            if metadata.get(key) != previous.get(key):
                raise RuntimeError(f"regeneration differs from original {key}: "
                                   f"{metadata.get(key)!r} != {previous.get(key)!r}")
        old = previous.get("physical_datasets", {})
        new = metadata.get("physical_datasets", {})
        if set(old) != set(new):
            raise RuntimeError("physical dataset membership changed")
        for name in old:
            for key in ("sumw", "sumw2", "xsec_pb", "files_processed"):
                if old[name].get(key) != new[name].get(key):
                    raise RuntimeError(f"normalization changed: {name}: {key}")


def _trota_command(setup: Path, repo: Path, root: Path, model: Path,
                   report: Path, year: int) -> list[str]:
    # The environment is sourced in a child shell, never in the py38 parent.
    script = ('set -e; set +u; source "$1"; shift; '
              'export PYTHONPATH="$1/autonomous_allhad:$1${PYTHONPATH:+:$PYTHONPATH}"; '
              'shift; exec python3 -u -m autonomous_allhad.trota_resolved_2024_inplace "$@"')
    return ["/bin/bash", "-c", script, "trota", str(setup), str(repo),
            "--input", str(root), "--model", str(model), "--metadata-output", str(report),
            "--target-year", str(year), "--chunk-events", "20000", "--batch-size", "8192",
            "--allow-hadd-repair"]


def produce(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.absolute()
    campaign = _work_path(args.campaign)
    destination = _work_path(args.output)
    work_parent = _work_path(args.work_dir)
    report_path = _work_path(args.report)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.shard_id):
        raise ValueError("invalid shard id")
    if not args.python.is_file() or not args.trota_setup.is_file():
        raise FileNotFoundError("py38 executable or TROTA setup is unavailable")
    if not shutil.which("hadd"):
        raise FileNotFoundError("hadd is required by the existing intermediate producer")
    manifest = json.loads((campaign / "manifest.json").read_text())
    if str(manifest["year"]) != str(args.year):
        raise ValueError("campaign year mismatch")
    model_info = manifest[f"trota_topresolved_{args.year}"]
    model = Path(model_info["model_bundle"])
    for item in (manifest["shard_bundle"], manifest["payload_bundle"],
                 {"path": str(model), "sha256": model_info["model_sha256"]}):
        if flat._sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"campaign input checksum mismatch: {item['path']}")
    with tarfile.open(manifest["shard_bundle"]["path"]) as archive:
        shard = json.load(archive.extractfile(f"shards/{args.shard_id}.json"))
    if shard["shard_id"] != args.shard_id or not shard["records"]:
        raise ValueError("invalid or empty shard")
    if any(str(record["year"]) != str(args.year) or record.get("is_data")
           for record in shard["records"]):
        raise ValueError("integrated TopW production requires same-year MC records")
    if destination.stem != args.shard_id:
        raise ValueError("output name must match the shard id")
    old_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    state = old_report.get("regeneration")
    if state and state["output"] != str(destination):
        raise RuntimeError("existing regeneration belongs to another output")
    if state and state.get("status") == "complete":
        if flat._sha256(destination) != state["sha256"]:
            raise RuntimeError("completed output checksum changed")
        return state
    if state and state.get("status") == "running" and state.get("host") == socket.getfqdn():
        if Path(f"/proc/{state.get('pid')}").exists():
            raise RuntimeError("this regeneration is already running")
    if destination.exists() and not state and not args.replace_existing:
        raise FileExistsError("use --replace-existing for an explicitly authorized regeneration")
    meta_destination = destination.with_suffix(".json")
    previous = json.loads(meta_destination.read_text()) if meta_destination.exists() else None
    if not state:
        if destination.exists() and (not previous or flat._sha256(destination) != previous["root_sha256"]):
            raise RuntimeError("original ROOT/metadata checksum mismatch")
        work_parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=args.shard_id + ".", dir=work_parent))
        state = {"status": "running", "output": str(destination), "work_dir": str(work),
                 "year": args.year, "shard_id": args.shard_id, "started": _now(), "stages": {},
                 "original_sha256": previous["root_sha256"] if previous else None}
    else:
        work = _work_path(Path(state["work_dir"]))
        if (work.parent.resolve() != work_parent.resolve()
                or not work.name.startswith(args.shard_id + ".") or work.is_symlink()):
            raise ValueError("saved work directory is outside this shard's workspace")
        if not work.is_dir():
            raise FileNotFoundError(work)
    state.update(status="running", pid=os.getpid(), host=socket.getfqdn())
    old_report["regeneration"] = state

    def save() -> None:
        flat.write_json(report_path, old_report)

    env = dict(os.environ)
    env.update(PYTHONPATH=str(repo / "autonomous_allhad") + ":" + str(repo),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0", AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="0",
               AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"), AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE="0",
               AUTONOMOUS_ALLHAD_FRAGMENT_DIR=str(work / "fragments"),
               TMPDIR=str(work), TMP=str(work), TEMP=str(work), XDG_CACHE_HOME=str(work / "cache"),
               MPLCONFIGDIR=str(work / "mpl"), NUMBA_CACHE_DIR=str(work / "numba"))
    env["PATH"] = str(args.python.parent) + os.pathsep + env.get("PATH", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        env[name] = "1"
    root = work / (args.shard_id + ".root")
    metadata_path = root.with_suffix(".json")
    try:
        save()
        # Resume an interrupted ROOT/JSON publication without rerunning NanoAOD.
        published = state["stages"].get("topw_truth", {})
        if (state.get("stage") == "publish" and not root.exists()
                and published.get("sha256") == flat._sha256(destination)):
            metadata = json.loads(metadata_path.read_text())
            metadata.update(root_file=str(destination), integrated_production="Events+TROTA+TopWTruth")
            flat.write_json(meta_destination, metadata)
            state.update(status="complete", stage="complete", finished=_now(),
                         sha256=published["sha256"], events=metadata["events_written"])
            old_report.update(status="complete", root=str(destination), sha256=published["sha256"],
                              bytes=destination.stat().st_size, marker=metadata["topw_truth"])
            save()
            shutil.rmtree(work)
            return state
        if not state["stages"].get("events"):
            _unpack_payload(Path(manifest["payload_bundle"]["path"]), work)
            shard_path = work / "shard.json"
            flat.write_json(shard_path, shard)
            module = ("autonomous_allhad.intermediate_2024_worker" if args.year == 2024
                      else "autonomous_allhad.intermediate_2025_data_worker")
            command = [str(args.python), "-u", "-m", module, "--repo", str(work),
                       "--shard", str(shard_path), "--output", str(root), "--metadata-output",
                       str(metadata_path), "--shift", "nominal", "--record-workers", str(args.record_workers)]
            state["stage"] = "events"; save()
            subprocess.run(command, cwd=work, env=env, check=True)
            metadata = json.loads(metadata_path.read_text())
            _check_events(metadata, shard, previous)
            metadata["root_sha256"] = flat._sha256(root)
            flat.write_json(metadata_path, metadata)
            state["stages"]["events"] = {"status": "complete", "events": metadata["events_written"],
                                           "sha256": metadata["root_sha256"]}; save()
        if not state["stages"].get("trota"):
            state["stage"] = "trota"; save()
            trota_report = work / "trota.json"
            trota_env = dict(env)
            for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
                trota_env.pop(key, None)
            subprocess.run(_trota_command(args.trota_setup, repo, root, model, trota_report, args.year),
                           cwd=work, env=trota_env, check=True)
            trota = json.loads(trota_report.read_text())
            if trota.get("status") not in ("complete", "already_complete"):
                raise RuntimeError("TROTA did not complete")
            metadata = json.loads(metadata_path.read_text())
            metadata.update(root_sha256=flat._sha256(root), root_trees=["Events", "TROTA"])
            metadata[f"trota_topresolved_{args.year}"] = trota
            flat.write_json(metadata_path, metadata)
            state["stages"]["trota"] = {"status": "complete", "sha256": metadata["root_sha256"]}; save()
        state["stage"] = "topw_truth"; save()
        truth = flat.append_topw_truth(root, root, repo, work, args.year)
        metadata = json.loads(metadata_path.read_text())
        _check_events(metadata, shard, previous)
        if truth["sha256"] != metadata["root_sha256"]:
            raise RuntimeError("integrated ROOT/metadata checksum mismatch")
        state["stages"]["topw_truth"] = {"status": "complete", "sha256": truth["sha256"]}; save()
        if state["original_sha256"] is not None:
            if flat._sha256(destination) != state["original_sha256"]:
                raise RuntimeError("original output changed; refusing replacement")
        elif destination.exists():
            raise FileExistsError(destination)
        metadata.update(root_file=str(destination), integrated_production="Events+TROTA+TopWTruth")
        state["stage"] = "publish"; save()
        os.replace(root, destination)
        flat.write_json(meta_destination, metadata)
        if flat._sha256(destination) != truth["sha256"]:
            raise RuntimeError("published ROOT checksum mismatch")
        state.update(status="complete", stage="complete", finished=_now(), sha256=truth["sha256"],
                     events=metadata["events_written"])
        old_report.update(status="complete", root=str(destination), sha256=truth["sha256"],
                          bytes=destination.stat().st_size, marker=truth["marker"])
        save()
        shutil.rmtree(work)
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", stopped=_now())
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "campaign", "output", "work-dir", "report", "trota-setup"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--record-workers", type=int, choices=range(2, 11), default=2)
    parser.add_argument("--replace-existing", action="store_true")
    result = produce(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

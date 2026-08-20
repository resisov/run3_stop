#!/usr/bin/env python3
"""Run the full 2024 TROTA histogram campaign locally with bounded memory."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_SCHEMA = "trota_highdm_exclusive_2024_chunk_v1"
PRIORITY_ROOTS = (
    "data_shard_00700.root",
    "mc_shard_01500.root",
    "signal_shard_00007.root",
    "signal_shard_00047.root",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


TOP_LEVEL_SCALAR = re.compile(
    r'^  "(?P<key>schema_version|status|files_expected|files_completed)": '
    r'(?P<value>"(?:[^"\\]|\\.)*"|-?[0-9]+),?$'
)


def valid_chunk(path: Path) -> bool:
    """Check completion metadata without loading the histogram JSON into memory."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 4096))
            if not stream.read().rstrip().endswith(b"}"):
                return False
        metadata: dict[str, Any] = {}
        with path.open() as stream:
            for raw in stream:
                match = TOP_LEVEL_SCALAR.match(raw.rstrip("\n"))
                if not match:
                    continue
                metadata[match.group("key")] = json.loads(match.group("value"))
                if len(metadata) == 4:
                    break
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == CHUNK_SCHEMA
        and metadata.get("status") == "complete"
        and int(metadata.get("files_expected") or -1)
        == int(metadata.get("files_completed") or -2)
        and int(metadata.get("files_completed") or 0) > 0
    )


def cgroup_memory() -> tuple[int | None, int | None]:
    try:
        relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if "::" in line
        )
        cgroup_root = Path("/sys/fs/cgroup")
        current_group = cgroup_root / relative.lstrip("/")
        while current_group == cgroup_root or cgroup_root in current_group.parents:
            maximum_text = (current_group / "memory.max").read_text().strip()
            if maximum_text != "max":
                return (
                    int((current_group / "memory.current").read_text().strip()),
                    int(maximum_text),
                )
            if current_group == cgroup_root:
                break
            current_group = current_group.parent
        return None, None
    except (OSError, StopIteration, ValueError):
        return None, None


def parse_tasks(arguments: Path) -> list[dict[str, Path | str]]:
    tasks = []
    for line_number, raw in enumerate(arguments.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid arguments row {line_number}: {raw!r}")
        name, input_list, output = fields
        tasks.append({
            "name": name,
            "input_list": Path(input_list),
            "output": Path(output),
        })
    if not tasks:
        raise ValueError(f"no tasks in {arguments}")
    names = [str(task["name"]) for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError("duplicate chunk names")
    outputs = [str(task["output"]) for task in tasks]
    if len(set(outputs)) != len(outputs):
        raise ValueError("duplicate chunk outputs")
    return tasks


def prioritize(tasks: list[dict[str, Path | str]]) -> list[dict[str, Path | str]]:
    available = {
        Path(line.strip()).name
        for task in tasks
        for line in Path(task["input_list"]).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    wanted = set(PRIORITY_ROOTS) & available
    priority = []
    regular = []
    covered: set[str] = set()
    for task in tasks:
        roots = {
            Path(line.strip()).name
            for line in Path(task["input_list"]).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        matches = roots & wanted
        if matches:
            priority.append(task)
            covered.update(matches)
        else:
            regular.append(task)
    if covered != wanted:
        raise RuntimeError(f"priority roots are not covered: {sorted(wanted - covered)}")
    return priority + regular


def state_payload(
    *,
    started_at: str,
    total: int,
    skipped: int,
    completed: int,
    running: dict[str, Any],
    pending: int,
    failures: list[dict[str, Any]],
    memory_current: int | None,
    memory_max: int | None,
    max_workers: int,
) -> dict[str, Any]:
    return {
        "schema_version": "trota_highdm_exclusive_2024_local_state_v1",
        "status": "failed" if failures else ("complete" if skipped + completed == total else "running"),
        "started_at": started_at,
        "updated_at": now(),
        "total_chunks": total,
        "skipped_valid_chunks": skipped,
        "completed_this_run": completed,
        "running": sorted(running),
        "pending": pending,
        "failures": failures,
        "max_workers": max_workers,
        "cgroup_memory_current_bytes": memory_current,
        "cgroup_memory_max_bytes": memory_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--memory-start-fraction", type=float, default=0.80)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 12:
        raise ValueError("max-workers must be between 1 and 12")
    if not 0.5 <= args.memory_start_fraction <= 0.95:
        raise ValueError("memory-start-fraction must be in [0.5, 0.95]")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise RuntimeError(f"Python runtime is not executable: {args.python}")

    campaign = args.campaign_dir
    manifest = json.loads((campaign / "manifest.json").read_text())
    tasks = parse_tasks(Path(manifest["arguments"]))
    if len(tasks) != int(manifest["chunk_count"]):
        raise RuntimeError("campaign chunk count drift")
    tasks = prioritize(tasks)
    skipped = sum(valid_chunk(Path(task["output"])) for task in tasks)
    pending = deque(task for task in tasks if not valid_chunk(Path(task["output"])))
    log_dir = campaign / "logs" / "local"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign / "local_run_state.json"
    started_at = now()
    completed = 0
    failures: list[dict[str, Any]] = []
    running: dict[str, dict[str, Any]] = {}
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{args.repo / 'autonomous_allhad'}:{args.repo / 'autonomous_allhad/workflow'}",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "XDG_CACHE_HOME": "/tmp/h24nres_local_cache",
        "MPLCONFIGDIR": "/tmp/h24nres_local_mpl",
    })
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    while pending or running:
        memory_current, memory_max = cgroup_memory()
        memory_allows_start = (
            memory_current is None
            or memory_max is None
            or memory_current < args.memory_start_fraction * memory_max
        )
        while (
            pending
            and len(running) < args.max_workers
            and not failures
            and memory_allows_start
        ):
            task = pending.popleft()
            name = str(task["name"])
            stdout_path = log_dir / f"{name}.out"
            stderr_path = log_dir / f"{name}.err"
            stdout = stdout_path.open("w")
            stderr = stderr_path.open("w")
            command = [
                str(args.python), "-u",
                str(args.repo / "autonomous_allhad/workflow/build_trota_highdm_exclusive_2024.py"),
                "--repo", str(args.repo),
                "--input-list", str(task["input_list"]),
                "--normalization", str(campaign / "normalization.json"),
                "--output", str(task["output"]),
            ]
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            running[name] = {
                "process": process,
                "task": task,
                "stdout": stdout,
                "stderr": stderr,
                "started_at": now(),
            }
            time.sleep(1.0)
            memory_current, memory_max = cgroup_memory()
            memory_allows_start = (
                memory_current is None
                or memory_max is None
                or memory_current < args.memory_start_fraction * memory_max
            )

        finished = []
        for name, record in running.items():
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["stdout"].close()
            record["stderr"].close()
            task = record["task"]
            if returncode == 0 and valid_chunk(Path(task["output"])):
                completed += 1
            else:
                failures.append({
                    "name": name,
                    "returncode": returncode,
                    "input_list": str(task["input_list"]),
                    "output": str(task["output"]),
                    "stderr": str(log_dir / f"{name}.err"),
                })
            finished.append(name)
        for name in finished:
            del running[name]

        memory_current, memory_max = cgroup_memory()
        atomic_json(state_path, state_payload(
            started_at=started_at,
            total=len(tasks),
            skipped=skipped,
            completed=completed,
            running=running,
            pending=len(pending),
            failures=failures,
            memory_current=memory_current,
            memory_max=memory_max,
            max_workers=args.max_workers,
        ))
        if failures and not running:
            return 1
        if pending or running:
            time.sleep(2.0)

    merge_script = campaign / "condor" / "merge_trota_highdm_exclusive_2024.py"
    merge_log = log_dir / "merge.out"
    merge_err = log_dir / "merge.err"
    with merge_log.open("w") as stdout, merge_err.open("w") as stderr:
        merge = subprocess.run([
            str(args.python), "-u", str(merge_script),
            "--campaign-dir", str(campaign),
            "--output", str(campaign / "hists.json"),
            "--summary", str(campaign / "summary.json"),
        ], env=env, stdout=stdout, stderr=stderr)
    if merge.returncode != 0:
        failures.append({"name": "merge", "returncode": merge.returncode, "stderr": str(merge_err)})
        memory_current, memory_max = cgroup_memory()
        atomic_json(state_path, state_payload(
            started_at=started_at,
            total=len(tasks),
            skipped=skipped,
            completed=completed,
            running={},
            pending=0,
            failures=failures,
            memory_current=memory_current,
            memory_max=memory_max,
            max_workers=args.max_workers,
        ))
        return 1
    print(json.dumps({
        "status": "complete",
        "chunks": len(tasks),
        "skipped": skipped,
        "completed_this_run": completed,
        "output": str(campaign / "hists.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

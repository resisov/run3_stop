#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def complete(state_path: Path, expected: int) -> bool:
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    return (
        state.get("status") == "complete"
        and int(state.get("completed_valid") or 0) == expected
        and int(state.get("failed_or_invalid") or 0) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--current-pid", type=int, required=True)
    parser.add_argument("--expected", type=int, default=75)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    state_path = args.campaign / "btageff_state.json"
    while process_exists(args.current_pid):
        time.sleep(30)
    if complete(state_path, args.expected):
        return 0

    runner = args.repo / "autonomous_allhad/scripts/run_t2tb_t2bw_btageff_local.py"
    for attempt in range(2, args.max_attempts + 1):
        if state_path.exists():
            shutil.copy2(
                state_path,
                args.campaign / f"btageff_state.attempt{attempt - 1}.json",
            )
        log_dir = args.campaign / "btageff_logs"
        if log_dir.exists():
            archived = args.campaign / f"btageff_logs_attempt{attempt - 1}"
            if not archived.exists():
                shutil.copytree(log_dir, archived)
        log = args.campaign / f"btageff_runner.attempt{attempt}.log"
        environment = dict(os.environ)
        environment["X509_USER_PROXY"] = str(
            args.repo / "autonomous_allhad/x509up_u147757"
        )
        with log.open("w") as handle:
            subprocess.run(
                [
                    str(args.python),
                    str(runner),
                    "--repo",
                    str(args.repo),
                    "--campaign",
                    str(args.campaign),
                    "--python",
                    str(args.python),
                    "--max-parallel",
                    "2",
                    "--workers-per-dataset",
                    "1",
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if complete(state_path, args.expected):
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

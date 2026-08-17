#!/usr/bin/env python3
"""Build J/psi tag-and-probe histograms for 5--10 GeV veto electrons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.tnp_histograms import build_histograms, read_file_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-files", type=Path, required=True)
    parser.add_argument("--mc-files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_2024.json"))
    parser.add_argument("--step-size", type=int, default=100_000)
    args = parser.parse_args()
    payload = build_histograms(kind="electron", data_files=read_file_list(args.data_files), mc_files=read_file_list(args.mc_files), config=json.loads(args.config.read_text()), repo=Path.cwd().resolve(), step_size=args.step_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0 if all(not item["files_failed"] and item["files_processed"] for item in payload["processing"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

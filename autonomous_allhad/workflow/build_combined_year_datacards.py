#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner


def cards_by_mass(directory: Path) -> dict[str, Path]:
    prefix = "datacard_"
    cards = {}
    for path in sorted(directory.glob(f"{prefix}*.txt")):
        mass = path.stem[len(prefix):]
        if mass:
            cards[mass] = path.absolute()
    return cards


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine two years of matching mass-point datacards.")
    parser.add_argument("--left-dir", required=True, type=Path)
    parser.add_argument("--left-label", default="y2024")
    parser.add_argument("--left-lumi-name", default="Lumi_2024")
    parser.add_argument("--right-dir", required=True, type=Path)
    parser.add_argument("--right-label", default="y2025")
    parser.add_argument("--right-lumi-name", default="Lumi_2025")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--combine-cards", default="combineCards.py")
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=600)
    args = parser.parse_args()

    left = cards_by_mass(args.left_dir)
    right = cards_by_mass(args.right_dir)
    if not left or not right:
        raise SystemExit("both source datacard directories must be non-empty")
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    if left_only or right_only:
        raise SystemExit(
            f"mass grids differ: left_only={len(left_only)}, right_only={len(right_only)}"
        )

    output_dir = args.output_dir.absolute()
    datacard_dir = output_dir / "datacards"
    limit_dir = output_dir / "limits"
    datacard_dir.mkdir(parents=True, exist_ok=True)
    combined_cards: dict[str, str] = {}
    warnings: list[dict[str, str]] = []
    for mass in sorted(left):
        output = datacard_dir / f"datacard_{mass}.txt"
        temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        try:
            with temporary.open("w") as handle:
                result = subprocess.run(
                    [
                        args.combine_cards,
                        f"{args.left_label}={left[mass]}",
                        f"{args.right_label}={right[mass]}",
                    ],
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"combineCards.py failed for {mass}: {result.stderr.strip()[:1000]}"
                )
            text = temporary.read_text()
            required = (args.left_lumi_name, args.right_lumi_name)
            missing_lumi = [name for name in required if name not in text]
            if missing_lumi:
                raise RuntimeError(
                    f"combined card {mass} is missing luminosity nuisances: {missing_lumi}"
                )
            if result.stderr.strip():
                warnings.append({"mass_point": mass, "stderr": result.stderr.strip()[:1000]})
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        combined_cards[mass] = str(output)

    runner = output_dir / "run_combine_expected.sh"
    write_parallel_runner(
        combined_cards,
        limit_dir,
        runner,
        args.runner_jobs,
        args.point_timeout,
    )
    manifest = {
        "status": "combine_inputs_ready",
        "method": "combineCards.py with year-labelled channels",
        "mass_point_count": len(combined_cards),
        "mass_points": sorted(combined_cards),
        "left": {
            "label": args.left_label,
            "datacard_dir": str(args.left_dir.absolute()),
            "luminosity_nuisance": args.left_lumi_name,
        },
        "right": {
            "label": args.right_label,
            "datacard_dir": str(args.right_dir.absolute()),
            "luminosity_nuisance": args.right_lumi_name,
        },
        "nuisance_correlation_policy": (
            "Identical nuisance names remain correlated across years; "
            "year-specific luminosity nuisance names remain uncorrelated."
        ),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "combine_cards_warnings": warnings,
    }
    write_json(output_dir / "combine_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "mass_points": len(combined_cards),
                "warnings": len(warnings),
                "datacard_dir": str(datacard_dir),
                "runner": str(runner),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

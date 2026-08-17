#!/usr/bin/env python3
"""Validate every AsymptoticLimits output in a tail-merged diagnostic grid."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import uproot


MASS_RE = re.compile(r"mStop(?P<stop>\d+)_mLSP(?P<lsp>\d+)")


def load_excluded_points(input_dir: Path) -> set[str]:
    excluded_points: set[str] = set()
    exclusion_dir = input_dir / "excluded_points"
    for path in sorted(exclusion_dir.glob("limit_point_exclusions_*.json")):
        payload = json.loads(path.read_text())
        for record in payload.get("exclusions", []):
            if record.get("status") not in {
                "excluded_by_user",
                "excluded_by_user_policy",
            }:
                continue
            if record.get("model") != input_dir.name:
                continue
            excluded_points.add(
                f"mStop{int(record['mStop_GeV'])}_mLSP{int(record['mLSP_GeV'])}"
            )
    return excluded_points


def validate_limit(path: Path) -> tuple[bool, str]:
    try:
        with uproot.open(path) as root_file:
            tree = root_file["limit"]
            limits = tree["limit"].array(library="np")
            quantiles = tree["quantileExpected"].array(library="np")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    expected_quantiles = (0.025, 0.16, 0.5, 0.84, 0.975)
    if len(limits) != 5 or len(quantiles) != 5:
        return False, f"expected 5 entries, found {len(limits)}/{len(quantiles)}"
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in limits):
        return False, "non-finite or negative limit"
    if not all(math.isfinite(float(value)) for value in quantiles):
        return False, "non-finite quantile"
    if not all(
        math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-5)
        for value, expected in zip(quantiles, expected_quantiles)
    ):
        return False, f"unexpected quantiles: {quantiles.tolist()}"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    original_expected = int(manifest["mass_point_count"])
    excluded_points = load_excluded_points(args.input_dir)
    expected = original_expected - len(excluded_points)
    files = sorted((args.input_dir / "limits").glob("higgsCombine*.root"))
    valid = []
    invalid = []
    for path in files:
        ok, reason = validate_limit(path)
        record = {"path": str(path), "reason": reason}
        (valid if ok else invalid).append(record)

    cards = sorted((args.input_dir / "datacards").glob("datacard_mStop*_mLSP*.txt"))
    card_points = {
        match.group(0)
        for path in cards
        if (match := MASS_RE.search(path.name)) is not None
    }
    valid_points = {
        match.group(0)
        for record in valid
        if (match := MASS_RE.search(Path(record["path"]).name)) is not None
    }
    result = {
        "status": (
            "complete"
            if len(cards) == expected
            and len(valid) == expected
            and valid_points == card_points
            and not invalid
            else "incomplete"
        ),
        "original_expected": original_expected,
        "expected": expected,
        "excluded": len(excluded_points),
        "excluded_points": sorted(excluded_points),
        "cards": len(cards),
        "outputs": len(files),
        "valid": len(valid),
        "invalid": len(invalid),
        "missing": len(card_points - valid_points),
        "invalid_examples": invalid[:10],
        "missing_examples": sorted(card_points - valid_points)[:10],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

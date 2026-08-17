#!/usr/bin/env python3
"""Fit veto-electron tag-and-probe pass/fail mass histograms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.reference_trigger_counts import json_safe
from workflow.tnp_fit import fit_histogram_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("histograms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_histogram_payload(json.loads(args.histograms.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    return 0 if all(item["valid"] for item in result["bins"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

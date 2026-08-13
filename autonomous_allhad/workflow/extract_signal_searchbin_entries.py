#!/usr/bin/env python3
"""Stream a large histogram JSON and extract nominal unweighted SR entries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCHEMES = {
    "highdm": "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR",
    "lowdm": "cat7_SR_lowDeltaM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_entries(lines, first_line: str) -> list[int]:
    text = first_line.split("[", 1)[1]
    pieces = [text]
    while "]" not in pieces[-1]:
        pieces.append(next(lines))
    payload = "".join(pieces).split("]", 1)[0]
    return [int(token) for token in re.findall(r"-?\d+", payload)]


def main() -> int:
    args = parse_args()
    requested = set(args.sample)
    result = {
        "status": "complete",
        "input": str(args.input),
        "samples": {sample: {} for sample in args.sample},
    }
    in_search_bins = False
    current_scheme = ""
    current_sample = ""
    current_variation = ""
    with args.input.open() as handle:
        lines = iter(handle)
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if indent == 2 and stripped.startswith('"search_bin_histograms":'):
                in_search_bins = True
                continue
            if not in_search_bins:
                continue
            if indent == 2 and stripped.startswith('"') and not stripped.startswith('"search_bin_histograms":'):
                break
            if indent == 4 and stripped.startswith('"'):
                current_scheme = stripped.split('"', 2)[1]
                current_sample = ""
                current_variation = ""
                continue
            if current_scheme not in SCHEMES.values():
                continue
            if indent == 6 and stripped.startswith('"'):
                current_sample = stripped.split('"', 2)[1]
                current_variation = ""
                continue
            if current_sample not in requested:
                continue
            if indent == 8 and stripped.startswith('"'):
                current_variation = stripped.split('"', 2)[1]
                continue
            if current_variation == "nominal" and indent == 10 and stripped.startswith('"entries":'):
                entries = parse_entries(lines, line)
                regime = next(key for key, value in SCHEMES.items() if value == current_scheme)
                result["samples"][current_sample][regime] = {
                    "bin_entries": entries,
                    "selected_entries": sum(entries),
                }

    missing = {
        sample: [regime for regime in SCHEMES if regime not in result["samples"][sample]]
        for sample in args.sample
    }
    missing = {sample: regimes for sample, regimes in missing.items() if regimes}
    if missing:
        result["status"] = "incomplete"
        result["missing"] = missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

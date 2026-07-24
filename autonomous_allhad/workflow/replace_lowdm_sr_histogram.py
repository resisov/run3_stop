#!/usr/bin/env python3
"""Replace the Low-dM SR scheme in a complete histogram payload."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SCHEME = "cat7_SR_lowDeltaM"
NBINS = 42


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def is_leaf(value: Any) -> bool:
    return isinstance(value, dict) and all(key in value for key in ("sumw", "sumw2", "entries"))


def validate_histograms(histograms: dict[str, Any]) -> None:
    if not histograms:
        raise ValueError(f"replacement has no {SCHEME} histograms")
    leaves = 0
    for sample, variations in histograms.items():
        for variation, leaf in variations.items():
            if not is_leaf(leaf):
                raise ValueError(f"invalid histogram leaf: {sample}/{variation}")
            lengths = {len(leaf[key]) for key in ("sumw", "sumw2", "entries")}
            if lengths != {NBINS}:
                raise ValueError(f"{sample}/{variation} is not {NBINS} bins: {sorted(lengths)}")
            leaves += 1
    if leaves == 0:
        raise ValueError("replacement contains no histogram leaves")


def nominal_totals(histograms: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sample, variations in histograms.items():
        leaf = variations.get("nominal")
        if is_leaf(leaf):
            out[sample] = float(sum(leaf["sumw"]))
    return out


def selected_summary(totals: dict[str, float]) -> dict[str, float]:
    return {
        "background": sum(value for sample, value in totals.items() if not sample.startswith(("T2tt", "data"))),
        "mStop600_mLSP400": totals.get("T2tt_mStop600_mLSP400", 0.0),
        "mStop900_mLSP700": totals.get("T2tt_mStop900_mLSP700", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--replacement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-roots", type=int, default=9484)
    args = parser.parse_args()

    base = read_json(args.base)
    replacement = read_json(args.replacement)
    if replacement.get("status") != "complete":
        raise ValueError(f"replacement status is {replacement.get('status')!r}, not 'complete'")
    processed_roots = len((replacement.get("summary") or {}).get("input_roots") or [])
    if processed_roots != args.expected_roots:
        raise ValueError(f"replacement processed {processed_roots} roots, expected {args.expected_roots}")

    old_histograms = (base.get("search_bin_histograms") or {}).get(SCHEME) or {}
    new_histograms = (replacement.get("search_bin_histograms") or {}).get(SCHEME) or {}
    validate_histograms(old_histograms)
    validate_histograms(new_histograms)

    old_totals = selected_summary(nominal_totals(old_histograms))
    new_totals = selected_summary(nominal_totals(new_histograms))
    comparison = {
        key: {
            "before": old_totals[key],
            "after": new_totals[key],
            "ratio": new_totals[key] / old_totals[key] if old_totals[key] else None,
        }
        for key in old_totals
    }

    base["search_bin_histograms"][SCHEME] = copy.deepcopy(new_histograms)
    base.setdefault("search_bin_schemes", {})[SCHEME] = copy.deepcopy(
        (replacement.get("search_bin_schemes") or {}).get(SCHEME) or {}
    )
    base["lowdm_selection_update"] = {
        "status": "complete",
        "classification": "physics proposal adopted by user on 2026-07-24",
        "rebuilt_region": "SR",
        "scheme": SCHEME,
        "bins": NBINS,
        "selection_change": [
            "ISR-subjet b veto removed",
            "mTb < 175 GeV requirement removed",
        ],
        "base": str(args.base),
        "replacement": str(args.replacement),
        "processed_nominal_roots": processed_roots,
        "control_region_policy": (
            "The five existing Low-dM 42-bin CR templates are retained because the "
            "broad intermediate stores SR primitives but not region-specific CR primitives."
        ),
        "nominal_yield_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(base["lowdm_selection_update"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

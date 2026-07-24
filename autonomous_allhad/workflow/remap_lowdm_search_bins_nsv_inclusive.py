#!/usr/bin/env python3
"""Project every Low-dM CR/SR payload into the adopted Nsv-inclusive bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_flat_boosted_recoil_hists import (
    LOWDM_42BIN_LABELS,
    LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES,
    LOWDM_REGION_MAP,
)


SCHEMES = list(LOWDM_REGION_MAP.values())


def is_hist_leaf(value: Any) -> bool:
    return isinstance(value, dict) and all(key in value for key in ("sumw", "sumw2", "entries"))


def add_lists(left: list[Any], right: list[Any]) -> list[Any]:
    return [a + b for a, b in zip(left, right)]


def remap_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
    if any(len(leaf.get(key) or []) != 53 for key in ("sumw", "sumw2", "entries")):
        raise ValueError("Low-dM source histogram is not 53 bins")
    out = {}
    for key in ("sumw", "sumw2", "entries"):
        old = leaf[key]
        out[key] = (
            add_lists(old[0:4], old[8:12])
            + add_lists(old[4:8], old[12:16])
            + list(old[16:32])
            + list(old[35:53])
        )
        if len(out[key]) != 42:
            raise AssertionError(f"{key} remap produced {len(out[key])} bins")
    return out


def merge_leaf(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sumw", "sumw2", "entries"):
        target[key] = add_lists(target[key], source[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repair", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    repair_payload = json.loads(args.repair.read_text()) if args.repair else {}
    projection_by_scheme: dict[str, Any] = {}
    for region, scheme in LOWDM_REGION_MAP.items():
        source = (payload.get("search_bin_histograms") or {}).get(scheme) or {}
        if not source:
            raise ValueError(f"missing {scheme}")

        projected: dict[str, Any] = {}
        dropped = {"sumw": 0.0, "sumw2": 0.0, "entries": 0}
        remapped_leaves = 0
        for sample, variations in source.items():
            for variation, leaf in variations.items():
                if not is_hist_leaf(leaf):
                    continue
                for key in dropped:
                    dropped[key] += sum(leaf[key][32:35])
                projected.setdefault(sample, {})[variation] = remap_leaf(leaf)
                remapped_leaves += 1

        repaired_leaves = 0
        repair = (repair_payload.get("search_bin_histograms") or {}).get(scheme) or {}
        for sample, variations in repair.items():
            for variation, leaf in variations.items():
                if not is_hist_leaf(leaf):
                    continue
                target = projected.setdefault(sample, {}).setdefault(
                    variation,
                    {"sumw": [0.0] * 42, "sumw2": [0.0] * 42, "entries": [0] * 42},
                )
                merge_leaf(target, leaf)
                repaired_leaves += 1

        payload["search_bin_histograms"][scheme] = projected
        payload.setdefault("search_bin_schemes", {})[scheme] = {
            "bin_labels": LOWDM_42BIN_LABELS,
            "selection": f"Low-dM {region} with Nsv excluded from category assignment",
            "delta_m": "low",
            "region": region,
            "nsv_policy": "inclusive; Nsv is not used in category assignment",
            "category_sizes": LOWDM_NSV_INCLUSIVE_CATEGORY_SIZES,
        }
        projection_by_scheme[scheme] = {
            "region": region,
            "remapped_histogram_leaves": remapped_leaves,
            "repair_histogram_leaves": repaired_leaves,
            "removed_old_nb1_nsv_nonzero_contribution_before_repair": dropped,
        }

    payload["lowdm_nsv_inclusive_projection"] = {
        "status": "complete",
        "source": str(args.input),
        "source_bins": 53,
        "output_bins": 42,
        "repair": str(args.repair) if args.repair else None,
        "schemes": projection_by_scheme,
        "policy": (
            "Old Nb=0 Nsv blocks are summed exactly; old Nb=1 Nsv=0 and all Nb>=2 blocks "
            "are preserved; old Nb=1,Nsv>=1 bins are replaced by event-level repair output."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(payload["lowdm_nsv_inclusive_projection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

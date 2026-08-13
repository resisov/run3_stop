#!/usr/bin/env python3
"""Preserve the exact datacard/Combine parameter names in impact plots."""

import argparse
import json
import re
from pathlib import Path


def impact_label(name: str) -> str:
    """Keep NPS nuisance names; shorten only generated autoMCStats names."""

    match = re.fullmatch(
        r"prop_bin([hl])([A-Za-z0-9]+)(?:_b(\d+)|_(.+?))_bin0(?:_(.+))?",
        name,
    )
    if not match:
        return name

    regime_code, region, sr_bin, category, process = match.groups()
    regime = "highdm" if regime_code == "h" else "lowdm"
    process = process or "combined"
    process = re.sub(r"_(?:Nb1|Nb2|Nb2plus|Nb3plus)$", "", process)
    location = f"bin{int(sr_bin)}" if sr_bin is not None else category
    location = re.sub(r"_u(\d+)$", r"_bin\1", location)
    return f"prop_{region}_{process}_{regime}_{location}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    impacts = json.loads(args.impacts.read_text())
    translations = {
        parameter["name"]: impact_label(parameter["name"])
        for parameter in impacts["params"]
    }
    args.output.write_text(
        json.dumps(translations, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "labels": len(translations),
                "unchanged": sum(
                    name == translated
                    for name, translated in translations.items()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

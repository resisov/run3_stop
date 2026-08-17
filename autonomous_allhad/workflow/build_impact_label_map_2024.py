#!/usr/bin/env python3
"""Preserve the exact datacard/Combine parameter names in impact plots."""

import argparse
from collections import Counter
import json
import re
from pathlib import Path


def impact_label(name: str) -> str:
    """Keep NPS nuisance names; shorten only generated autoMCStats names."""

    adopted = re.fullmatch(
        r"prop_bin(LLCR|QCDCR|GCR|SR)_(highdm|lowdm)_"
        r"(?:(Nb1|Nb2plus)_bin(\d+)|bin(\d+))_bin0(?:_(.+))?",
        name,
    )
    if adopted:
        region, regime, group, control_bin, signal_bin, process = adopted.groups()
        location = f"bin{int(signal_bin or control_bin)}"
        process_label = process or "combined"
        process_label = re.sub(
            r"_(?:Nb1|Nb2|Nb2plus|Nb3plus)_u\d+$", "", process_label
        )
        group_label = f"_{group}" if group else ""
        return f"prop_{region}_{process_label}_{regime}{group_label}_{location}"

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
    candidates = {
        parameter["name"]: impact_label(parameter["name"])
        for parameter in impacts["params"]
    }
    multiplicities = Counter(candidates.values())
    translations = {}
    for name, candidate in candidates.items():
        if multiplicities[candidate] == 1:
            translations[name] = candidate
            continue
        process_match = re.search(
            r"_(Nb1|Nb2|Nb2plus|Nb3plus)_u(\d+)$", name
        )
        if process_match:
            group, recoil_bin = process_match.groups()
            translations[name] = f"{candidate}_{group}_UTbin{int(recoil_bin)}"
        else:
            translations[name] = name
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

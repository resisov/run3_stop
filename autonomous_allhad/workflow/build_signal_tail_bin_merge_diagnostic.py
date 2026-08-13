#!/usr/bin/env python3
"""Build Combine cards with selected High-dM bins merged.

This is a diagnostic transformer for already validated one-bin-per-channel
Combine templates.  It preserves every unmodified channel, sums nominal
histograms and their Sumw2 for each requested adjacent pair, recomputes the
corresponding lnN factors from the template summary, and removes the second
channel in each pair.  It can transform either one requested signal mass
point or the complete signal grid into one shared ROOT template.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

from build_boosted_an17_combine_inputs import signal_process_name, write_json
from build_free_background_combine_inputs_2024 import (
    datacard_text,
    make_backgrounds_free,
)


def parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid merge pair {value!r}; expected FIRST:SECOND"
        ) from exc
    if right != left + 1:
        raise argparse.ArgumentTypeError(
            f"merge pair must be adjacent: {left}:{right}"
        )
    if not (1 <= left < right <= 60):
        raise argparse.ArgumentTypeError(
            f"High-dM bin pair outside 1..60: {left}:{right}"
        )
    if (left - 1) // 6 != (right - 1) // 6:
        raise argparse.ArgumentTypeError(
            f"pair crosses a category boundary: {left}:{right}"
        )
    return left, right


def weighted_factor(
    records: list[dict[str, Any] | None],
    nuisance: str,
    direction: str,
) -> float:
    total = sum(float(record["yield"]) for record in records if record)
    if total <= 0.0:
        return 1.0
    shifted = 0.0
    for record in records:
        if not record:
            continue
        factor = (
            (record.get("nuisance_factors") or {})
            .get(nuisance, {})
            .get(direction, 1.0)
        )
        shifted += float(record["yield"]) * float(factor)
    return shifted / total


def merge_process_records(
    records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    records = [record for record in records if record]
    if not records:
        return None
    nuisances = sorted(
        {
            nuisance
            for record in records
            for nuisance in (record.get("nuisance_factors") or {})
        }
    )
    return {
        "yield": sum(float(record["yield"]) for record in records),
        "weight_nuisances": nuisances,
        "nuisance_factors": {
            nuisance: {
                "down": weighted_factor(records, nuisance, "down"),
                "up": weighted_factor(records, nuisance, "up"),
            }
            for nuisance in nuisances
        },
    }


def merge_channel_summary(
    name: str,
    source_names: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    source_records = [summary["channels"][source] for source in source_names]
    processes = sorted(
        {
            process
            for record in source_records
            for process in record["backgrounds"]
        }
    )
    backgrounds = {}
    for process in processes:
        merged = merge_process_records(
            [record["backgrounds"].get(process) for record in source_records]
        )
        if merged:
            backgrounds[process] = merged
    result = copy.deepcopy(source_records[0])
    result["backgrounds"] = backgrounds
    result["background_yield"] = sum(
        float(record["background_yield"]) for record in source_records
    )
    result["diagnostic_merged_channels"] = source_names
    result["rate_params"] = {}
    return result


def merge_signal_summary(
    mass_key: str,
    channel_map: dict[str, list[str]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    original = summary["signals"][mass_key]
    result = copy.deepcopy(original)
    result["channels"] = {}
    result["nuisance_factors"] = {}
    for destination, sources in channel_map.items():
        yields = [
            float(original["channels"].get(source, 0.0))
            for source in sources
        ]
        result["channels"][destination] = sum(yields)
        nuisances = sorted(
            {
                nuisance
                for source in sources
                for nuisance in (
                    original["nuisance_factors"].get(source) or {}
                )
            }
        )
        factors = {}
        total = sum(yields)
        for nuisance in nuisances:
            pair = {}
            for direction in ("down", "up"):
                shifted = 0.0
                for source, nominal in zip(sources, yields):
                    factor = (
                        (
                            original["nuisance_factors"].get(source)
                            or {}
                        )
                        .get(nuisance, {})
                        .get(direction, 1.0)
                    )
                    shifted += nominal * float(factor)
                pair[direction] = shifted / total if total > 0.0 else 1.0
            factors[nuisance] = pair
        result["nuisance_factors"][destination] = factors
    return result


def channel_objects(
    source_names: list[str],
    summary: dict[str, Any],
    signals: list[str],
) -> list[str]:
    return sorted(
        {
            "data_obs",
            *signals,
            *(
                process
                for source in source_names
                for process in summary["channels"][source]["backgrounds"]
            ),
        }
    )


def write_merged_root(
    source_root: Path,
    output_root: Path,
    channel_map: dict[str, list[str]],
    summary: dict[str, Any],
    signals: list[str],
) -> None:
    import ROOT

    # The large majority of channels are unchanged.  Preserve the complete
    # validated ROOT contract by copying it once, then overwrite only the
    # destination directories of requested merge pairs.  The now-unreferenced
    # second directories remain in the file, but are omitted from the summary
    # and cards.  This is exactly equivalent to rewriting every retained
    # object and avoids O(channels * signal-points) PyROOT cloning.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_root, output_root)
    source = ROOT.TFile.Open(str(source_root), "READ")
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open source ROOT file: {source_root}")
    output = ROOT.TFile.Open(str(output_root), "UPDATE")
    if not output or output.IsZombie():
        raise RuntimeError(f"cannot create output ROOT file: {output_root}")
    try:
        for destination, sources in channel_map.items():
            if len(sources) == 1:
                continue
            directory = output.GetDirectory(destination)
            if not directory:
                raise RuntimeError(
                    f"destination directory missing: {destination}"
                )
            object_names = sorted(
                {
                    key.GetName()
                    for source_name in sources
                    for key in source.GetDirectory(
                        source_name
                    ).GetListOfKeys()
                }
            )
            for process in object_names:
                merged = None
                for source_name in sources:
                    hist = source.Get(f"{source_name}/{process}")
                    if not hist:
                        continue
                    if merged is None:
                        merged = hist.Clone(process)
                        merged.SetDirectory(0)
                    else:
                        merged.Add(hist)
                if merged is None:
                    continue
                directory.cd()
                merged.SetName(process)
                merged.Write(process, ROOT.TObject.kOverwrite)
                del merged
        output.Flush()
    finally:
        output.Close()
        source.Close()
    if not output_root.exists() or output_root.stat().st_size <= 0:
        raise RuntimeError(f"empty output ROOT file: {output_root}")


def channels_for_card(summary: dict[str, Any]) -> list[dict[str, Any]]:
    channels = []
    for name, record in summary["channels"].items():
        backgrounds = {
            process: {
                "variations": {
                    nuisance: {}
                    for nuisance in process_record.get(
                        "weight_nuisances", []
                    )
                }
            }
            for process, process_record in record["backgrounds"].items()
        }
        channels.append(
            {
                "name": name,
                "kind": record["kind"],
                "regime": record["regime"],
                "region": record["region"],
                "source_bin": record["source_bin"],
                "backgrounds": backgrounds,
                "rate_params": {},
                "signal_source": None,
            }
        )
    make_backgrounds_free(channels)
    for channel in channels:
        summary["channels"][channel["name"]]["rate_params"] = channel[
            "rate_params"
        ]
    return channels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--template-summary", type=Path, required=True)
    parser.add_argument("--topology", choices=("T2tt", "T2bW", "T2tb"), required=True)
    mass_group = parser.add_mutually_exclusive_group(required=True)
    mass_group.add_argument("--mass-key")
    mass_group.add_argument(
        "--all-mass-points",
        action="store_true",
        help="Transform every signal mass point in the template summary",
    )
    parser.add_argument(
        "--merge-pair",
        action="append",
        type=parse_pair,
        required=True,
        help="Adjacent one-based High-dM bin pair FIRST:SECOND",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    args = parser.parse_args()

    summary = json.loads(args.template_summary.read_text())
    if args.all_mass_points:
        mass_keys = sorted(summary.get("signals", {}))
    else:
        mass_keys = [args.mass_key]
    if not mass_keys:
        raise SystemExit("no signal mass points selected")
    missing_mass_keys = [
        mass_key
        for mass_key in mass_keys
        if mass_key not in summary.get("signals", {})
    ]
    if missing_mass_keys:
        raise SystemExit(
            "mass point absent from summary: "
            + ", ".join(missing_mass_keys)
        )
    pairs = list(args.merge_pair)
    used = [item for pair in pairs for item in pair]
    if len(set(used)) != len(used):
        raise SystemExit(f"overlapping merge pairs: {pairs}")

    source_to_destination = {}
    for left, right in pairs:
        destination = f"hSR_b{left - 1:02d}"
        source_to_destination[f"hSR_b{right - 1:02d}"] = destination

    channel_map: dict[str, list[str]] = {}
    for name in summary["channels"]:
        if name in source_to_destination:
            continue
        channel_map[name] = [name]
    for left, right in pairs:
        destination = f"hSR_b{left - 1:02d}"
        channel_map[destination] = [
            destination,
            f"hSR_b{right - 1:02d}",
        ]

    signals = [signal_process_name(mass_key) for mass_key in mass_keys]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_dir / "templates.root"
    output_summary_path = args.output_dir / "template_summary.json"
    card_dir = (
        args.output_dir / "datacards"
        if args.all_mass_points
        else args.output_dir
    )
    card_dir.mkdir(parents=True, exist_ok=True)

    write_merged_root(
        args.template_root,
        output_root,
        channel_map,
        summary,
        signals,
    )

    new_summary = {
        "channels": {},
        "signals": {
            mass_key: merge_signal_summary(
                mass_key, channel_map, summary
            )
            for mass_key in mass_keys
        },
        "background_grouping_contract": summary.get(
            "background_grouping_contract", {}
        ),
    }
    for destination, sources in channel_map.items():
        if len(sources) == 1:
            new_summary["channels"][destination] = copy.deepcopy(
                summary["channels"][destination]
            )
        else:
            new_summary["channels"][destination] = merge_channel_summary(
                destination, sources, summary
            )

    channels = channels_for_card(new_summary)
    write_json(output_summary_path, new_summary)
    output_cards = {}
    template_reference = (
        Path("..") / output_root.name
        if args.all_mass_points
        else Path(output_root.name)
    )
    for mass_key in mass_keys:
        output_card = card_dir / f"datacard_{mass_key}.txt"
        output_card.write_text(
            datacard_text(
                template_reference,
                channels,
                mass_key,
                new_summary,
                args.auto_mc_stats,
            )
        )
        output_cards[mass_key] = str(output_card)

    signal_yield_checks = {}
    for mass_key in mass_keys:
        source_signal_total = sum(
            float(value)
            for value in summary["signals"][mass_key]["channels"].values()
        )
        output_signal_total = sum(
            float(value)
            for value in new_summary["signals"][mass_key]["channels"].values()
        )
        if abs(source_signal_total - output_signal_total) > max(
            1.0e-10, abs(source_signal_total) * 1.0e-10
        ):
            raise RuntimeError(
                "signal yield changed during merge for "
                f"{mass_key}: {source_signal_total} -> {output_signal_total}"
            )
        signal_yield_checks[mass_key] = {
            "source": source_signal_total,
            "output": output_signal_total,
            "difference": output_signal_total - source_signal_total,
        }

    output_channel_names = set(new_summary["channels"])
    channel_counts = {
        "total": len(output_channel_names),
        "highdm_signal": sum(
            name.startswith("hSR_") for name in output_channel_names
        ),
        "highdm_control": sum(
            name.startswith("h") and not name.startswith("hSR_")
            for name in output_channel_names
        ),
        "lowdm_signal": sum(
            name.startswith("lSR_") for name in output_channel_names
        ),
        "lowdm_control": sum(
            name.startswith("l") and not name.startswith("lSR_")
            for name in output_channel_names
        ),
    }
    free_background_parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel.get("rate_params", {}).values()
        }
    )
    manifest = {
        "status": "complete",
        "schema_version": (
            f"2024_highdm{channel_counts['highdm_signal']}_"
            f"lowdm{channel_counts['lowdm_signal']}_"
            "free_background_sharedcr_v1"
        ),
        "transformation_schema": "signal_tail_bin_merge_diagnostic_v2",
        "model": "free_background_global_process_normalizations",
        "channels": channel_counts,
        "topology": args.topology,
        "mass_points": mass_keys,
        "mass_keys": mass_keys,
        "mass_point_count": len(mass_keys),
        "source_template_root": str(args.template_root),
        "source_template_summary": str(args.template_summary),
        "merge_pairs_1based": [list(pair) for pair in pairs],
        "output_template_root": str(output_root),
        "output_template_summary": str(output_summary_path),
        "output_datacards": output_cards,
        "source_channel_count": len(summary["channels"]),
        "output_channel_count": len(new_summary["channels"]),
        "root_storage_policy": (
            "source ROOT copied in full; merged destination directories "
            "overwritten; removed channels retained but unreferenced"
        ),
        "unreferenced_merged_source_channels": sorted(
            source_to_destination
        ),
        "signal_yield_checks": signal_yield_checks,
        "free_background_parameters": free_background_parameters,
        "external_background_constraints": [],
        "auto_mc_stats": args.auto_mc_stats,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

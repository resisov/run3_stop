#!/usr/bin/env python3
"""Plot the 73 retained High-dM SR bins directly from limit templates."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import uproot

import plot_control_search_bins_style as style


BACKGROUND_COMPONENTS = {
    "DY": ("DY",),
    "GJ": ("PhotonJet",),
    "QCD": ("QCD_",),
    "TT": ("Top_",),
    "VV": ("VV_VVV",),
    "WtoLNu": ("WtoLNu_",),
    "Zto2Nu": ("Zto2Nu_",),
}
SIGNALS = {
    "T2tt_mStop1000_mLSP1": "sig_mStop1000_mLSP1",
    "T2tt_mStop1200_mLSP1": "sig_mStop1200_mLSP1",
}


def nominal_names(directory: uproot.ReadOnlyDirectory) -> list[str]:
    return [
        str(name)
        for name in directory.keys(cycle=False)
        if "_CMS_" not in str(name)
        and not str(name).endswith("Up")
        and not str(name).endswith("Down")
    ]


def matching_names(names: list[str], patterns: tuple[str, ...]) -> list[str]:
    selected = []
    for name in names:
        for pattern in patterns:
            if name == pattern or name.startswith(pattern):
                selected.append(name)
                break
    return selected


def one_bin(directory: uproot.ReadOnlyDirectory, names: list[str]) -> tuple[float, float]:
    value = 0.0
    variance = 0.0
    for name in names:
        histogram = directory[name]
        value += float(np.asarray(histogram.values(), dtype=float)[0])
        current_variance = histogram.variances()
        if current_variance is not None:
            variance += float(np.asarray(current_variance, dtype=float)[0])
    return value, variance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--templates",
        type=Path,
        nargs="+",
        help="one or more same-binning yearly template ROOT files to sum",
    )
    inputs.add_argument("--payloads", type=Path, nargs="+",
                        help="canonical SR model JSON exports; no local ROOT input")
    parser.add_argument("--bin-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--luminosity-fb", type=float, default=style.LUMINOSITY_FB)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    exports = [json.loads(path.read_text()) for path in args.payloads or []]
    if exports:
        if any(item.get("schema_version") != "canonical_sr_plot_payload_v1"
               or item.get("sr_observations_included") is not False
               or item.get("prediction_stage") != "prefit"
               or item.get("rate_initials_applied") is not True
               or item.get("uncertainty") != "mc_statistical_only"
               or item.get("topology") != "T2tt" for item in exports):
            raise RuntimeError("invalid blinded T2tt SR plot payload")
        bin_map = exports[0]["bin_map"]
        if any(item["bin_map"]["highdm"] != bin_map["highdm"] for item in exports):
            raise RuntimeError("yearly High-dM bin maps differ")
    else:
        if args.bin_map is None:
            parser.error("--templates requires --bin-map")
        bin_map = json.loads(args.bin_map.read_text())
    highdm = list(bin_map["highdm"])
    expected_removed = [f"SR_highdm_bin{index}" for index in range(6)]
    if exports:
        if len(highdm) != 73 or any(item["highdm_projection"].get("dropped_final_bins_1based") != list(range(1, 7)) for item in exports):
            raise RuntimeError("SR export must contain the 73 retained High-dM bins")
    elif len(highdm) != 79 or [record["channel"] for record in highdm[:6]] != expected_removed:
        raise RuntimeError("limit bin map does not begin with SR_highdm_bin0--5")

    samples = {
        sample: {"sumw": [], "sumw2": []}
        for sample in (*BACKGROUND_COMPONENTS, "data_obs", *SIGNALS)
    }
    if exports:
        for record in highdm:
            directories = [item["channels"][record["channel"]] for item in exports]
            for sample, patterns in BACKGROUND_COMPONENTS.items():
                leaves = [directory[name] for directory in directories
                          for name in matching_names(list(directory), patterns)]
                samples[sample]["sumw"].append(sum(float(leaf["sumw"][0]) for leaf in leaves))
                samples[sample]["sumw2"].append(sum(float(leaf["sumw2"][0]) for leaf in leaves))
            samples["data_obs"]["sumw"].append(0.0)
            samples["data_obs"]["sumw2"].append(0.0)
            for sample, name in SIGNALS.items():
                for field in ("sumw", "sumw2"):
                    samples[sample][field].append(sum(float(directory[name][field][0]) for directory in directories))
    with ExitStack() as stack:
        root_files = [stack.enter_context(uproot.open(path)) for path in args.templates or []]
        for record in highdm if root_files else []:
            directories = [root_file[str(record["channel"])] for root_file in root_files]
            names_by_directory = [nominal_names(directory) for directory in directories]
            for sample, patterns in BACKGROUND_COMPONENTS.items():
                value = 0.0
                variance = 0.0
                for directory, names in zip(directories, names_by_directory):
                    current_value, current_variance = one_bin(
                        directory, matching_names(names, patterns)
                    )
                    value += current_value
                    variance += current_variance
                samples[sample]["sumw"].append(value)
                samples[sample]["sumw2"].append(variance)
            value = 0.0
            variance = 0.0
            for directory in directories:
                current_value, current_variance = one_bin(directory, ["data_obs"])
                value += current_value
                variance += current_variance
            samples["data_obs"]["sumw"].append(value)
            samples["data_obs"]["sumw2"].append(variance)
            for output_name, root_name in SIGNALS.items():
                value = 0.0
                variance = 0.0
                for directory in directories:
                    current_value, current_variance = one_bin(directory, [root_name])
                    value += current_value
                    variance += current_variance
                samples[output_name]["sumw"].append(value)
                samples[output_name]["sumw2"].append(variance)

    payload = {
        "search_bin_schemes": {
            "highdm_search_bins": {
                "bin_labels": [record["label"] for record in highdm],
                "dropped_final_bins_1based": list(range(1, 7)) if exports else [],
            }
        },
        "search_bin_histograms": {
            "highdm_search_bins": {
                sample: {"nominal": values}
                for sample, values in samples.items()
            }
        },
    }
    # The limit template already carries the adopted 79-bin projection.  The
    # block builder removes precisely channels 0--5 and returns 73 bins.
    style.LUMINOSITY_RELATIVE_UNCERTAINTY = 0.0
    blocks = style.selected_an17_recoil_blocks(payload, "highdm_search_bins")
    if sum(int(block["nbin"]) for block in blocks) != 73:
        raise RuntimeError("High-dM limit-template projection did not yield 73 bins")
    summary = style.draw_flat_blocks(
        blocks,
        args.output,
        xlabel="Search bin",
        uncertainty_label_override="MC stat. unc.",
        luminosity_fb=args.luminosity_fb,
    )
    summary.update(
        {
            "schema_version": "highdm73_limit_template_plot_v1",
            "templates": [str(path) for path in args.templates or []],
            "payloads": [str(path) for path in args.payloads or []],
            "bin_map": str(args.bin_map) if args.bin_map else None,
            "luminosity_fb": args.luminosity_fb,
            "source_highdm_bins": 79,
            "removed_channels": expected_removed,
            "retained_highdm_bins": 73,
            "significance_definition": "S/sqrt(B)",
            "main_ylabel": "Events",
            "prediction_stage": "prefit" if exports else "template_input",
            "uncertainty": "mc_statistical_only",
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

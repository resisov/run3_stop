#!/usr/bin/env python3
"""Build the adopted canonical likelihood with only Low-dM channels.

The statistical treatment is identical to ``build_combine_inputs.py`` for
Low-dM: LLCR, QCDCR, and GCR are simultaneous Poisson channels, RZ enters as
an external covariance constraint, and the measured Sgamma parameter is
shared by the matched GCR and Z(inv) SR bin.  High-dM control and signal
channels are removed before templates and cards are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_combine_inputs as canonical  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lowdm_channels(
    channels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [channel for channel in channels if channel.get("regime") == "lowdm"]
    if not selected:
        raise ValueError("canonical model contains no Low-dM channels")
    if any("highdm" in str(channel.get("name", "")).lower() for channel in selected):
        raise ValueError("High-dM channel leaked into Low-dM-only model")
    if not any(channel.get("region") == "SR" for channel in selected):
        raise ValueError("Low-dM-only model has no signal-region channels")
    if not any(channel.get("region") != "SR" for channel in selected):
        raise ValueError("Low-dM-only model has no control-region channels")
    return selected


def validate_rate_parameter_scopes(
    channels: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel.get("rate_params", {}).values()
        }
    )
    scopes = {
        parameter: sorted(
            {
                "sr" if channel.get("region") == "SR" else "cr"
                for channel in channels
                if parameter in channel.get("rate_params", {}).values()
            }
        )
        for parameter in parameters
    }
    invalid = {
        parameter: scope
        for parameter, scope in scopes.items()
        if set(scope) != {"cr", "sr"}
    }
    if invalid:
        raise ValueError(f"unmatched Low-dM rate parameters: {invalid}")
    return parameters, scopes


def lowdm_mass_points(
    hists: dict[str, Any],
    topology: str,
    only: list[str] | None,
    max_mstop: int,
) -> list[str]:
    candidates = canonical.mass_points(
        hists,
        only,
        max_mstop,
        topology=topology,
    )
    selected = []
    for mass_key in candidates:
        values, _ = canonical.signal_leaf(
            hists,
            "lowdm",
            mass_key,
            "nominal",
            topology,
        )
        if len(values) and float(np.sum(values)) > 0.0:
            selected.append(mass_key)
    if not selected:
        raise ValueError(f"no positive-yield Low-dM {topology} mass points")
    return selected


def write_lowdm_cards(
    channels: list[dict[str, Any]],
    masses: list[str],
    template_root: Path,
    summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
) -> dict[str, str]:
    cards = canonical.write_cards(
        channels,
        masses,
        template_root,
        summary,
        output_dir,
        auto_mc_stats,
    )
    for card_name in cards.values():
        path = Path(card_name)
        text = path.read_text()
        if any(token in text for token in ("LLCR_highdm", "SR_highdm", "QCDCR_highdm", "GCR_highdm")):
            raise ValueError(f"High-dM channel leaked into {path}")
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hists", type=Path, required=True)
    parser.add_argument("--hists-sha256")
    parser.add_argument("--campaign-year", choices=("2024", "2025"), required=True)
    parser.add_argument("--topology", choices=("T2tt", "T2bW", "T2tb"), required=True)
    parser.add_argument("--sgamma", type=Path, required=True)
    parser.add_argument("--rz-high", type=Path, required=True)
    parser.add_argument("--rz-low", type=Path, required=True)
    parser.add_argument("--zgamma-double-ratio", type=Path, required=True)
    parser.add_argument("--search-bin-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    canonical.CAMPAIGN_YEAR = str(args.campaign_year)
    canonical.NPS_LUMI_NAME = f"lumi_13p6TeV_{args.campaign_year}"
    canonical.enforce_downstream_input_boundary(args)

    input_paths = {
        "hists": args.hists,
        "sgamma": args.sgamma,
        "rz_high": args.rz_high,
        "rz_low": args.rz_low,
        "zgamma_double_ratio": args.zgamma_double_ratio,
        "search_bin_config": args.search_bin_config,
    }
    sgamma = canonical.read_json(args.sgamma)
    rz_high = canonical.read_json(args.rz_high)
    rz_low = canonical.read_json(args.rz_low)
    double_ratio = canonical.read_json(args.zgamma_double_ratio)
    search_bin_configuration = canonical.read_json(args.search_bin_config)
    for label, payload in (
        ("Sgamma", sgamma),
        ("RZ high", rz_high),
        ("Z/gamma double ratio", double_ratio),
    ):
        if payload.get("status") != "complete":
            raise SystemExit(f"{label} input incomplete: {payload.get('status')}")
    if rz_low.get("status") not in {"complete", "feature_stage_complete"}:
        raise SystemExit(f"RZ low input incomplete: {rz_low.get('status')}")
    if search_bin_configuration.get("schema_version") != "search_bin_scheme_v1":
        raise SystemExit("unsupported High-dM search-bin configuration")
    if str(search_bin_configuration.get("campaign_year")) != str(args.campaign_year):
        raise SystemExit("search-bin configuration year mismatch")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hists, exact, extracted_signals = canonical.extract_current_histogram_input(
        args.hists,
        args.topology,
    )
    rz_covariance = canonical.build_rz_covariance(rz_high, rz_low)
    write_json(output_dir / "rz_covariance.json", rz_covariance)
    all_channels, unmatched, bin_map, dropped = canonical.build_channels(
        exact,
        sgamma,
        rz_covariance,
        double_ratio,
    )
    channels = lowdm_channels(all_channels)
    parameters, parameter_scopes = validate_rate_parameter_scopes(channels)
    write_json(
        output_dir / "bin_map.json",
        {"highdm": [], "lowdm": bin_map["lowdm"]},
    )
    masses = lowdm_mass_points(
        hists,
        args.topology,
        args.only,
        args.max_mstop,
    )
    template_root = output_dir / "templates.root"
    card_dir = output_dir / "cards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_limits.sh"
    summary = canonical.build_root(
        channels,
        hists,
        masses,
        template_root,
        topology=args.topology,
    )
    canonical.overwrite_observations(template_root, channels, summary)
    cards = write_lowdm_cards(
        channels,
        masses,
        template_root,
        summary,
        card_dir,
        args.auto_mc_stats,
    )
    canonical.write_parallel_runner(
        cards,
        limit_dir,
        runner,
        args.runner_jobs,
        args.point_timeout,
    )

    channel_counts = {
        "total": len(channels),
        "highdm_control": 0,
        "highdm_signal": 0,
        "lowdm_control": sum(channel["region"] != "SR" for channel in channels),
        "lowdm_signal": sum(channel["region"] == "SR" for channel in channels),
    }
    manifest = {
        "schema_version": f"canonical_lowdm_only_{args.campaign_year}_v1",
        "status": "combine_inputs_ready",
        "model": {
            "highdm_included": False,
            "control_regions": list(canonical.LOW_CONTROL_REGIONS),
            "dilepton_poisson_channels": False,
            "zinv_free_normalization_rate_parameter": False,
            "highdm_bins": 0,
            "highdm_source_bins": len(exact["highdm"]["search_bin_labels"]),
            "lowdm_bins": len(exact["lowdm"]["search_bin_labels"]),
            "signal_topology": args.topology,
            "dropped_empty_control_channels": [
                name for name in dropped if "lowdm" in name.lower()
            ],
        },
        "inputs": {
            label: {
                "path": str(path),
                "sha256": (
                    args.hists_sha256
                    if label == "hists" and args.hists_sha256
                    else canonical.sha256(path)
                ),
            }
            for label, path in input_paths.items()
        },
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "canonical_builder": {
            "path": str(Path(canonical.__file__).resolve()),
            "sha256": sha256(Path(canonical.__file__)),
        },
        "template_root": str(template_root),
        "cards": cards,
        "runner": str(runner),
        "limit_validator": str(runner.with_name("validate_limit_outputs.py")),
        "limit_success_gate": "five finite nonnegative expected quantiles with canonical values",
        "mass_points": masses,
        "extracted_signal_samples": extracted_signals,
        "channels": channel_counts,
        "rate_parameter_count": len(parameters),
        "rate_parameters": parameters,
        "rate_parameter_scopes": parameter_scopes,
        "unmatched_rate_parameters_dropped_by_full_builder": unmatched,
        "auto_mc_stats": args.auto_mc_stats,
        "background_grouping": canonical.background_grouping_contract(),
        "root_summary": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "topology": args.topology,
                "channels": channel_counts,
                "rate_parameters": len(parameters),
                "mass_point_count": len(masses),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build 2024 High-dM 54/60-bin and corresponding High+Low-dM limit models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_boosted_an17_combine_inputs import (  # noqa: E402
    BACKGROUND_NAME,
    LUMI_LNN,
    LUMI_NAME,
    datacard_text,
    write_hist,
    write_json,
)
from build_combine_inputs_from_preview import collect_limits, plot_contour  # noqa: E402
from build_flat_boosted_an17_combine_inputs import CONTROL_REGION_MAP, cr_channel  # noqa: E402
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner  # noqa: E402
from build_flat_recoil_sr_combine_inputs import (  # noqa: E402
    SIGNAL_PREFIX,
    aggregate_variations,
    parse_mass_key,
    sample_to_mass_key,
    signal_process_name,
)
from build_flat_selected_recoil6_lowdm_combine_inputs import (  # noqa: E402
    selected_recoil6_channel,
    signal_array_from_selected,
    signal_variations_from_selected,
)
from build_lowdm42_combine_inputs import (  # noqa: E402
    LOWDM_REGION_MAP,
    lowdm_channel,
    signal_array as lowdm_signal_array,
)
from background_process_groups import (  # noqa: E402
    BACKGROUND_PROCESS_ORDER,
    background_grouping_contract,
)


HIGHDM_SCHEMES = {
    54: (
        "boosted_an17_selected_recoil6_with_nt0_wsplit_SR",
        "cat7_SR_selected_recoil54_nt0_wsplit",
    ),
    60: (
        "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR",
        "cat7_SR_selected_recoil60_nb2_nt2plus_w0",
    ),
}
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def nominal_signal_yield(flat: dict[str, Any], scheme: str, mass_key: str) -> float:
    leaf = (
        (((flat.get("search_bin_histograms") or {}).get(scheme) or {}).get(SIGNAL_PREFIX + mass_key) or {})
        .get("nominal")
        or {}
    )
    return sum(float(value) for value in (leaf.get("sumw") or []))


def mass_points(
    flat: dict[str, Any],
    highdm_scheme: str,
    include_lowdm: bool,
    only: list[str] | None,
    max_mstop: int | None,
) -> list[str]:
    schemes = [highdm_scheme]
    if include_lowdm:
        schemes.append(LOWDM_REGION_MAP["SR"])
    selected: set[str] = set()
    for scheme in schemes:
        by_sample = ((flat.get("search_bin_histograms") or {}).get(scheme) or {})
        for sample in by_sample:
            mass_key = sample_to_mass_key(sample)
            if not mass_key or (only and mass_key not in only):
                continue
            mstop, mlsp = parse_mass_key(mass_key)
            if mlsp >= mstop or (max_mstop is not None and mstop > max_mstop):
                continue
            if any(nominal_signal_yield(flat, source, mass_key) > 0.0 for source in schemes):
                selected.add(mass_key)
    return sorted(selected, key=parse_mass_key)


def lowdm_signal_variations(
    flat: dict[str, Any],
    scheme: str,
    mass_key: str,
    nominal: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(scheme) or {})
    return aggregate_variations(by_sample, [SIGNAL_PREFIX + mass_key], len(nominal), nominal)


def build_root(
    channels: list[dict[str, Any]],
    flat: dict[str, Any],
    selected_masses: list[str],
    output_root: Path,
    data_mode: str,
    highdm_scheme: str,
) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_shape_nuisances": sorted({name for channel in channels for name in channel["variations"]}),
        "background_grouping_contract": background_grouping_contract(),
    }
    try:
        for channel in channels:
            name = channel["name"]
            source_region = channel["source_region"]
            directory = root_file.mkdir(name)
            edges = np.asarray(channel["edges"], dtype=float)
            background = np.asarray(channel["background"], dtype=float)
            background_sumw2 = np.asarray(channel["background_sumw2"], dtype=float)
            data = background if data_mode == "asimov" else np.asarray(channel["data"], dtype=float)
            write_hist(directory, "data_obs", data, np.maximum(data, 0.0), edges)
            grouped = channel.get("background_processes") or {}
            if not grouped:
                raise ValueError(f"{name}: missing grouped background processes")
            grouped_sumw = np.zeros(len(background), dtype=float)
            grouped_sumw2 = np.zeros(len(background), dtype=float)
            process_summary: dict[str, Any] = {}
            for process in BACKGROUND_PROCESS_ORDER:
                record = grouped.get(process) or {}
                raw_values = record.get("sumw")
                raw_sumw2 = record.get("sumw2")
                values = np.asarray(
                    [] if raw_values is None else raw_values, dtype=float
                )
                sumw2 = np.asarray(
                    [] if raw_sumw2 is None else raw_sumw2, dtype=float
                )
                if len(values) != len(background) or len(sumw2) != len(background):
                    raise ValueError(f"{name}/{process}: grouped template has wrong bin count")
                grouped_sumw += values
                grouped_sumw2 += sumw2
                sources = list(record.get("source_samples") or [])
                if not sources:
                    continue
                write_hist(directory, process, np.maximum(values, MIN_BIN), sumw2, edges)
                nuisances = record.get("variations") or {}
                for nuisance, pair in nuisances.items():
                    write_hist(
                        directory,
                        f"{process}_{nuisance}Up",
                        np.maximum(np.asarray(pair["up"], dtype=float), MIN_BIN),
                        sumw2,
                        edges,
                    )
                    write_hist(
                        directory,
                        f"{process}_{nuisance}Down",
                        np.maximum(np.asarray(pair["down"], dtype=float), MIN_BIN),
                        sumw2,
                        edges,
                    )
                process_summary[process] = {
                    "display_label": record.get("display_label", process),
                    "yield": float(np.sum(values)),
                    "source_samples": sources,
                    "shape_nuisances": sorted(nuisances),
                }
            if not np.allclose(grouped_sumw, background, rtol=1.0e-10, atol=1.0e-8):
                raise ValueError(f"{name}: grouped yields do not reproduce the background total")
            if not np.allclose(grouped_sumw2, background_sumw2, rtol=1.0e-10, atol=1.0e-8):
                raise ValueError(f"{name}: grouped sumw2 does not reproduce the background total")
            summary["channels"][name] = {
                "source_region": source_region,
                "kind": channel["kind"],
                "bin_count": len(background),
                "background_yield": float(np.sum(background)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted(channel["variations"]),
                "background_processes": process_summary,
                "bin_labels": channel.get("bin_labels") or [],
            }

            for mass_key in selected_masses:
                process = signal_process_name(mass_key)
                signal_variations: dict[str, dict[str, np.ndarray]] = {}
                if source_region == highdm_scheme:
                    signal, signal_sumw2 = signal_array_from_selected(
                        flat,
                        highdm_scheme,
                        mass_key,
                        len(background),
                    )
                    high_variations = signal_variations_from_selected(
                        flat,
                        highdm_scheme,
                        mass_key,
                        len(background),
                    )
                    signal_variations = {
                        nuisance: {
                            "up": pair["up"][0],
                            "down": pair["down"][0],
                        }
                        for nuisance, pair in high_variations.items()
                    }
                elif source_region in LOWDM_REGION_MAP.values():
                    signal, signal_sumw2 = lowdm_signal_array(flat, source_region, mass_key)
                    signal_variations = lowdm_signal_variations(flat, source_region, mass_key, signal)
                else:
                    signal = np.zeros(len(background), dtype=float)
                    signal_sumw2 = np.zeros(len(background), dtype=float)

                write_hist(directory, process, np.maximum(signal, MIN_BIN), signal_sumw2, edges)
                signal_summary = summary["signals"].setdefault(
                    mass_key,
                    {"process": process, "channels": {}, "shape_nuisances": {}},
                )
                signal_summary["channels"][name] = float(np.sum(signal))
                for nuisance, pair in signal_variations.items():
                    write_hist(
                        directory,
                        f"{process}_{nuisance}Up",
                        np.maximum(np.asarray(pair["up"], dtype=float), MIN_BIN),
                        signal_sumw2,
                        edges,
                    )
                    write_hist(
                        directory,
                        f"{process}_{nuisance}Down",
                        np.maximum(np.asarray(pair["down"], dtype=float), MIN_BIN),
                        signal_sumw2,
                        edges,
                    )
                    signal_summary["shape_nuisances"].setdefault(nuisance, []).append(name)
    finally:
        root_file.Close()
    return summary


def write_datacards(
    channels: list[dict[str, Any]],
    selected_masses: list[str],
    template_root: Path,
    root_summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
    include_lowdm: bool,
    highdm_bins: int,
    lowdm_bins: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    model_note = (
        f"# 2024 model: five 6-bin High-dM CRs plus {highdm_bins}-bin High-dM SR"
        + (
            f"; five {lowdm_bins}-bin Low-dM CRs plus {lowdm_bins}-bin Low-dM SR are also included.\n"
            if include_lowdm
            else "; no Low-dM channel is included.\n"
        )
    )
    obsolete_note = (
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n"
    )
    for mass_key in selected_masses:
        card = output_dir / f"datacard_{mass_key}.txt"
        text = datacard_text(
            template_root,
            channels,
            mass_key,
            root_summary,
            auto_mc_stats,
            lumi_name=LUMI_NAME,
            lumi_lnn=LUMI_LNN,
        ).replace(obsolete_note, "")
        card.write_text(model_note + text)
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-lowdm", action="store_true")
    parser.add_argument("--highdm-bins", type=int, choices=(54, 60), default=54)
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1600)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    include_lowdm = not args.no_lowdm
    highdm_scheme, highdm_channel = HIGHDM_SCHEMES[args.highdm_bins]
    flat = read_json(args.hists)
    channels = [
        cr_channel(flat, source_region, channel_name)
        for source_region, channel_name in CONTROL_REGION_MAP.items()
    ]
    channels.append(selected_recoil6_channel(flat, highdm_scheme, highdm_channel))
    if include_lowdm:
        channels.extend(
            lowdm_channel(flat, region, scheme)
            for region, scheme in LOWDM_REGION_MAP.items()
        )
    lowdm_bins = (
        len(
            next(
                channel["background"]
                for channel in channels
                if channel["source_region"] == LOWDM_REGION_MAP["SR"]
            )
        )
        if include_lowdm
        else 0
    )
    if include_lowdm and any(
        len(channel["background"]) != lowdm_bins
        for channel in channels
        if channel["source_region"] in LOWDM_REGION_MAP.values()
    ):
        raise ValueError("Low-dM regions do not have a common bin count")
    selected_masses = mass_points(
        flat, highdm_scheme, include_lowdm, args.only, args.max_mstop
    )
    if args.max_points is not None:
        selected_masses = selected_masses[: args.max_points]
    if not selected_masses:
        raise SystemExit("no signal mass points selected")

    outdir = args.output_dir
    tag = (
        f"highdm{args.highdm_bins}_lowdm{lowdm_bins}"
        if include_lowdm
        else f"highdm{args.highdm_bins}_only"
    )
    template_root = outdir / f"templates_{tag}.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    if args.collect_only:
        root_summary: dict[str, Any] = {}
    else:
        root_summary = build_root(
            channels,
            flat,
            selected_masses,
            template_root,
            args.data_mode,
            highdm_scheme,
        )
        cards = write_datacards(
            channels,
            selected_masses,
            template_root,
            root_summary,
            datacard_dir,
            args.auto_mc_stats,
            include_lowdm,
            args.highdm_bins,
            lowdm_bins,
        )
        write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    limit_payload = collect_limits(limit_dir, selected_masses, outdir / "expected_limits.json")
    contour_png = outdir / f"expected_limit_contour_{tag}.png"
    analysis_label = (
        rf"High-$\Delta m$ {args.highdm_bins}-bin + Low-$\Delta m$ {lowdm_bins}-bin"
        if include_lowdm
        else rf"High-$\Delta m$ {args.highdm_bins}-bin only"
    )
    contour_written = plot_contour(
        limit_payload,
        contour_png,
        run2_contours=Path("/eos/user/t/taiwoo/run2_sus19010_contours.json"),
        luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
        analysis_label=analysis_label,
        x_max=float(args.max_mstop),
    )
    manifest = {
        "status": "combine_outputs_complete" if limit_payload["status"] == "complete" else "combine_inputs_ready",
        "schema": f"2024_{tag}_v1",
        "hists": str(args.hists),
        "channels": [channel["name"] for channel in channels],
        "channel_count": len(channels),
        "highdm": {
            "control_channels": 5,
            "control_bins_each": 6,
            "signal_channel": highdm_channel,
            "signal_bins": args.highdm_bins,
        },
        "lowdm": {
            "included": include_lowdm,
            "control_channels": 5 if include_lowdm else 0,
            "control_bins_each": lowdm_bins if include_lowdm else 0,
            "signal_channel": LOWDM_REGION_MAP["SR"] if include_lowdm else None,
            "signal_bins": lowdm_bins if include_lowdm else 0,
        },
        "total_analysis_bins": sum(len(channel["background"]) for channel in channels),
        "data_mode": args.data_mode,
        "max_mstop_inclusive": args.max_mstop,
        "mass_points": selected_masses,
        "mass_point_count": len(selected_masses),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "expected_limits": str(outdir / "expected_limits.json"),
        "contour_png": str(contour_png) if contour_written else None,
        "contour_pdf": str(contour_png.with_suffix(".pdf")) if contour_written else None,
        "systematics": {
            "weight_shapes": sorted({name for channel in channels for name in channel["variations"]}),
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
            "object_shape_variations": sorted(
                {
                    name
                    for channel in channels
                    for name in channel["variations"]
                    if name == "jesFlavorQCD"
                }
            ),
        },
        "root_summary": root_summary,
        "limit_collection": limit_payload,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "model": tag,
        "channels": manifest["channel_count"],
        "bins": manifest["total_analysis_bins"],
        "mass_points": len(selected_masses),
        "limits_collected": limit_payload["collected_point_count"],
        "contour": manifest["contour_png"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

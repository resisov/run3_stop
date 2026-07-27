#!/usr/bin/env python3
"""Build the 2024 High-dM60 + Low-dM34 bin-by-bin transfer-factor model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from background_process_groups import (  # noqa: E402
    BACKGROUND_PROCESS_ORDER,
    background_grouping_contract,
)
from build_boosted_an17_combine_inputs import (  # noqa: E402
    LUMI_LNN,
    LUMI_NAME,
    parse_mass_key,
    signal_process_name,
    stable_path,
    write_hist,
    write_json,
)
from build_combine_inputs_from_preview import collect_limits, plot_contour  # noqa: E402
from build_flat_recoil_ntop_split_combine_inputs import (  # noqa: E402
    write_parallel_runner,
)


HIGH_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"
LOW_SCHEME = "cat7_SR_lowDeltaM"
SIGNAL_PREFIX = "T2tt_"
MIN_BIN = 1.0e-9
MIN_VARIATION_RATIO = 1.0e-3
RAW_COMPONENTS = {
    "VV_VVV": ("VV",),
    "Top": ("ST", "TT"),
    "DY": ("DY",),
    "PhotonJet": ("GJ",),
    "WtoLNu": ("WtoLNu",),
    "Zto2Nu": ("Zto2Nu",),
    "QCD": ("QCD",),
    "Other": (),
}
CONTROLLED = ("Top", "WtoLNu", "QCD", "Zto2Nu")
HIGH_GROUPS = ("Nb1", "Nb2", "Nb3plus")
LOW_GROUPS = ("Nb1", "Nb2plus")
HIGH_REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
LOW_REGIONS = HIGH_REGIONS
DENOMINATOR_LINKS = {
    "LLCR": {"Top": "Top", "WtoLNu": "WtoLNu"},
    "QCDCR": {"QCD": "QCD"},
    "GCR": {"PhotonJet": "Zto2Nu"},
    "DY2E": {"DY": "Zto2Nu"},
    "DY2M": {"DY": "Zto2Nu"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def leaf_arrays(
    by_sample: dict[str, Any],
    process: str,
    variation: str,
    nbin: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(nbin, dtype=float)
    sumw2 = np.zeros(nbin, dtype=float)
    for sample in RAW_COMPONENTS[process]:
        variations = by_sample.get(sample) or {}
        record = variations.get(variation) or variations.get("nominal") or {}
        current = np.asarray(record.get("sumw") or [], dtype=float)
        current_sumw2 = np.asarray(record.get("sumw2") or [], dtype=float)
        if len(current) != nbin or len(current_sumw2) != nbin:
            if not len(current) and not len(current_sumw2):
                continue
            raise ValueError(
                f"{process}/{sample}/{variation}: expected {nbin} bins, "
                f"found {len(current)}/{len(current_sumw2)}"
            )
        values += current
        sumw2 += current_sumw2
    return values, sumw2


def variation_pairs(
    by_sample: dict[str, Any],
    process: str,
    nbin: int,
) -> dict[str, dict[str, np.ndarray]]:
    names = {
        name
        for sample in RAW_COMPONENTS[process]
        for name in (by_sample.get(sample) or {})
        if name != "nominal"
    }
    bases = sorted(
        {
            name[:-2]
            for name in names
            if name.endswith("Up") and name[:-2] + "Down" in names
        }
    )
    return {
        base: {
            "up": leaf_arrays(by_sample, process, base + "Up", nbin)[0],
            "down": leaf_arrays(by_sample, process, base + "Down", nbin)[0],
        }
        for base in bases
    }


def one_bin_background(
    by_sample: dict[str, Any],
    process: str,
    source_bin: int,
    nbin: int,
) -> dict[str, Any] | None:
    nominal, sumw2 = leaf_arrays(by_sample, process, "nominal", nbin)
    value = float(nominal[source_bin])
    variance = float(sumw2[source_bin])
    if value <= 0.0 and variance <= 0.0:
        return None
    variations = variation_pairs(by_sample, process, nbin)
    nominal_value = max(value, MIN_BIN)
    variation_floor = max(MIN_BIN, nominal_value * MIN_VARIATION_RATIO)
    retained_variations = {}
    for nuisance, pair in variations.items():
        up_value = max(float(pair["up"][source_bin]), variation_floor)
        down_value = max(float(pair["down"][source_bin]), variation_floor)
        if np.isclose(
            up_value,
            nominal_value,
            rtol=1.0e-12,
            atol=1.0e-15,
        ) and np.isclose(
            down_value,
            nominal_value,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            continue
        retained_variations[nuisance] = {
            "up": np.asarray([up_value], dtype=float),
            "down": np.asarray([down_value], dtype=float),
        }
    return {
        "nominal": np.asarray([nominal_value], dtype=float),
        "sumw2": np.asarray([max(variance, 0.0)], dtype=float),
        "variations": retained_variations,
    }


def sum_one_bin_backgrounds(
    records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    records = [record for record in records if record is not None]
    if not records:
        return None
    nominal = sum((record["nominal"] for record in records), np.zeros(1))
    sumw2 = sum((record["sumw2"] for record in records), np.zeros(1))
    nuisances = sorted(
        {
            nuisance
            for record in records
            for nuisance in record["variations"]
        }
    )
    variations = {}
    for nuisance in nuisances:
        up = np.zeros(1)
        down = np.zeros(1)
        for record in records:
            pair = record["variations"].get(nuisance)
            up += pair["up"] if pair else record["nominal"]
            down += pair["down"] if pair else record["nominal"]
        variations[nuisance] = {"up": up, "down": down}
    return {"nominal": nominal, "sumw2": sumw2, "variations": variations}


def high_param(process: str, group: str, recoil_bin: int) -> str:
    return f"tf_high_{process}_{group}_u{recoil_bin}"


def low_param(process: str, group: str, search_bin: int) -> str:
    return f"tf_low_{process}_{group}_b{search_bin:02d}"


def build_channels(exact: dict[str, Any]) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    high_source = exact["highdm"]
    low_source = exact["lowdm"]

    for region in HIGH_REGIONS:
        for group in HIGH_GROUPS:
            by_sample = high_source["recoil"][region][group]
            for recoil_bin in range(len(high_source["recoil_edges"]) - 1):
                backgrounds = {}
                rate_params = {}
                for process in BACKGROUND_PROCESS_ORDER:
                    record = one_bin_background(
                        by_sample,
                        process,
                        recoil_bin,
                        len(high_source["recoil_edges"]) - 1,
                    )
                    if record is not None:
                        backgrounds[process] = record
                for denominator_process, target_process in DENOMINATOR_LINKS[
                    region
                ].items():
                    if denominator_process in backgrounds:
                        rate_params[denominator_process] = high_param(
                            target_process, group, recoil_bin
                        )
                channels.append(
                    {
                        "name": f"h{region}_{group}_u{recoil_bin}",
                        "kind": "highdm_control_nb_recoil_onebin",
                        "regime": "highdm",
                        "region": region,
                        "nb_group": group,
                        "source_bin": recoil_bin,
                        "backgrounds": backgrounds,
                        "rate_params": rate_params,
                        "signal_source": None,
                    }
                )

    high_components = high_source["sr_components"]
    high_bins = len(
        next(
            iter(
                next(
                    iter(
                        next(iter(high_components.values())).values()
                    )
                ).values()
            )
        )["sumw"]
    )
    for source_bin in range(high_bins):
        backgrounds = {}
        rate_params = {}
        recoil_bin = source_bin % 6
        for process in CONTROLLED:
            for group in HIGH_GROUPS:
                by_sample = high_components[group]
                record = one_bin_background(
                    by_sample, process, source_bin, high_bins
                )
                if record is None:
                    continue
                component_name = f"{process}_{group}"
                backgrounds[component_name] = record
                rate_params[component_name] = high_param(
                    process, group, recoil_bin
                )
        for process in ("VV_VVV", "DY", "PhotonJet"):
            backgrounds[process] = sum_one_bin_backgrounds(
                [
                    one_bin_background(
                        high_components[group],
                        process,
                        source_bin,
                        high_bins,
                    )
                    for group in HIGH_GROUPS
                ]
            )
            if backgrounds[process] is None:
                backgrounds.pop(process)
        channels.append(
            {
                "name": f"hSR_b{source_bin:02d}",
                "kind": "highdm_signal_searchbin",
                "regime": "highdm",
                "region": "SR",
                "source_bin": source_bin,
                "backgrounds": backgrounds,
                "rate_params": rate_params,
                "signal_source": ("highdm", source_bin),
            }
        )

    low_search = low_source["search_components"]
    low_bins = len(low_source["search_bin_labels"])
    for region in LOW_REGIONS:
        for source_bin in range(low_bins):
            group = "Nb1" if source_bin < 16 else "Nb2plus"
            by_sample = low_search[region][group]
            backgrounds = {}
            rate_params = {}
            for process in BACKGROUND_PROCESS_ORDER:
                record = one_bin_background(
                    by_sample, process, source_bin, low_bins
                )
                if record is not None:
                    backgrounds[process] = record
            for denominator_process, target_process in DENOMINATOR_LINKS[
                region
            ].items():
                if denominator_process in backgrounds:
                    rate_params[denominator_process] = low_param(
                        target_process, group, source_bin
                    )
            channels.append(
                {
                    "name": f"l{region}_b{source_bin:02d}",
                    "kind": "lowdm_control_searchbin",
                    "regime": "lowdm",
                    "region": region,
                    "nb_group": group,
                    "source_bin": source_bin,
                    "bin_label": low_source["search_bin_labels"][source_bin],
                    "backgrounds": backgrounds,
                    "rate_params": rate_params,
                    "signal_source": None,
                }
            )
    for source_bin in range(low_bins):
        group = "Nb1" if source_bin < 16 else "Nb2plus"
        by_sample = low_search["SR"][group]
        backgrounds = {}
        rate_params = {}
        for process in BACKGROUND_PROCESS_ORDER:
            record = one_bin_background(
                by_sample, process, source_bin, low_bins
            )
            if record is not None:
                backgrounds[process] = record
            if process in CONTROLLED and process in backgrounds:
                rate_params[process] = low_param(process, group, source_bin)
        channels.append(
            {
                "name": f"lSR_b{source_bin:02d}",
                "kind": "lowdm_signal_searchbin",
                "regime": "lowdm",
                "region": "SR",
                "nb_group": group,
                "source_bin": source_bin,
                "bin_label": low_source["search_bin_labels"][source_bin],
                "backgrounds": backgrounds,
                "rate_params": rate_params,
                "signal_source": ("lowdm", source_bin),
            }
        )
    parameter_scope: dict[str, set[str]] = {}
    for channel in channels:
        scope = "sr" if channel["region"] == "SR" else "cr"
        for parameter in channel["rate_params"].values():
            parameter_scope.setdefault(parameter, set()).add(scope)
    valid_parameters = {
        parameter
        for parameter, scopes in parameter_scope.items()
        if scopes == {"cr", "sr"}
    }
    for channel in channels:
        channel["rate_params"] = {
            process: parameter
            for process, parameter in channel["rate_params"].items()
            if parameter in valid_parameters
        }
    return channels


def signal_histogram(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
) -> dict[str, Any]:
    scheme = HIGH_SCHEME if regime == "highdm" else LOW_SCHEME
    return (
        ((hists.get("search_bin_histograms") or {}).get(scheme) or {}).get(
            SIGNAL_PREFIX + mass_key
        )
        or {}
    )


def signal_leaf(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
    variation: str,
) -> tuple[np.ndarray, np.ndarray]:
    variations = signal_histogram(hists, regime, mass_key)
    record = variations.get(variation) or variations.get("nominal") or {}
    return (
        np.asarray(record.get("sumw") or [], dtype=float),
        np.asarray(record.get("sumw2") or [], dtype=float),
    )


def signal_variations(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
) -> list[str]:
    names = set(signal_histogram(hists, regime, mass_key))
    return sorted(
        {
            name[:-2]
            for name in names
            if name.endswith("Up") and name[:-2] + "Down" in names
        }
    )


def mass_points(
    hists: dict[str, Any],
    only: list[str] | None,
    max_mstop: int,
) -> list[str]:
    selected = set()
    for scheme in (HIGH_SCHEME, LOW_SCHEME):
        for sample, variations in (
            (hists.get("search_bin_histograms") or {}).get(scheme) or {}
        ).items():
            match = re.fullmatch(r"T2tt_(mStop\d+_mLSP\d+)", sample)
            if not match:
                continue
            mass_key = match.group(1)
            if only and mass_key not in only:
                continue
            mstop, mlsp = parse_mass_key(mass_key)
            nominal = (variations.get("nominal") or {}).get("sumw") or []
            if mstop <= max_mstop and mlsp < mstop and sum(nominal) > 0.0:
                selected.add(mass_key)
    return sorted(selected, key=parse_mass_key)


def build_root(
    channels: list[dict[str, Any]],
    hists: dict[str, Any],
    masses: list[str],
    output_root: Path,
) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_grouping_contract": background_grouping_contract(),
    }
    try:
        for channel in channels:
            directory = root_file.mkdir(channel["name"])
            background_total = np.zeros(1)
            background_sumw2 = np.zeros(1)
            background_summary = {}
            for process, record in sorted(channel["backgrounds"].items()):
                write_hist(
                    directory,
                    process,
                    record["nominal"],
                    record["sumw2"],
                    np.asarray([0.0, 1.0]),
                )
                background_total += record["nominal"]
                background_sumw2 += record["sumw2"]
                nuisance_factors = {
                    nuisance: {
                        "down": float(pair["down"][0] / record["nominal"][0]),
                        "up": float(pair["up"][0] / record["nominal"][0]),
                    }
                    for nuisance, pair in record["variations"].items()
                }
                background_summary[process] = {
                    "yield": float(record["nominal"][0]),
                    "weight_nuisances": sorted(record["variations"]),
                    "nuisance_factors": nuisance_factors,
                }
            write_hist(
                directory,
                "data_obs",
                background_total,
                background_sumw2,
                np.asarray([0.0, 1.0]),
            )
            summary["channels"][channel["name"]] = {
                "kind": channel["kind"],
                "regime": channel["regime"],
                "region": channel["region"],
                "source_bin": channel["source_bin"],
                "background_yield": float(background_total[0]),
                "backgrounds": background_summary,
                "rate_params": channel["rate_params"],
            }
            if channel.get("nb_group"):
                summary["channels"][channel["name"]]["nb_group"] = channel[
                    "nb_group"
                ]
            if channel.get("bin_label"):
                summary["channels"][channel["name"]]["bin_label"] = channel[
                    "bin_label"
                ]
            for mass_key in masses:
                process = signal_process_name(mass_key)
                signal = np.zeros(1)
                signal_sumw2 = np.zeros(1)
                signal_nuisance_factors: dict[str, dict[str, float]] = {}
                if channel["signal_source"]:
                    regime, source_bin = channel["signal_source"]
                    nominal, sumw2 = signal_leaf(
                        hists, regime, mass_key, "nominal"
                    )
                    if source_bin < len(nominal):
                        signal[0] = max(float(nominal[source_bin]), MIN_BIN)
                        signal_sumw2[0] = max(
                            float(sumw2[source_bin]), 0.0
                        )
                    for nuisance in signal_variations(
                        hists, regime, mass_key
                    ):
                        up = signal_leaf(
                            hists, regime, mass_key, nuisance + "Up"
                        )[0]
                        down = signal_leaf(
                            hists, regime, mass_key, nuisance + "Down"
                        )[0]
                        if source_bin >= len(up) or source_bin >= len(down):
                            continue
                        signal_variation_floor = max(
                            MIN_BIN,
                            signal[0] * MIN_VARIATION_RATIO,
                        )
                        up_value = max(
                            float(up[source_bin]),
                            signal_variation_floor,
                        )
                        down_value = max(
                            float(down[source_bin]),
                            signal_variation_floor,
                        )
                        if np.isclose(
                            up_value,
                            signal[0],
                            rtol=1.0e-12,
                            atol=1.0e-15,
                        ) and np.isclose(
                            down_value,
                            signal[0],
                            rtol=1.0e-12,
                            atol=1.0e-15,
                        ):
                            continue
                        signal_nuisance_factors[nuisance] = {
                            "down": float(down_value / signal[0]),
                            "up": float(up_value / signal[0]),
                        }
                write_hist(
                    directory,
                    process,
                    signal,
                    signal_sumw2,
                    np.asarray([0.0, 1.0]),
                )
                signal_summary = summary["signals"].setdefault(
                    mass_key,
                    {
                        "process": process,
                        "channels": {},
                        "weight_nuisances": {},
                        "nuisance_factors": {},
                    },
                )
                signal_summary["channels"][channel["name"]] = float(signal[0])
                signal_summary["nuisance_factors"][
                    channel["name"]
                ] = signal_nuisance_factors
                for nuisance in signal_nuisance_factors:
                    signal_summary["weight_nuisances"].setdefault(
                        nuisance, []
                    ).append(channel["name"])
    finally:
        root_file.Close()
    return summary


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    summary: dict[str, Any],
    auto_mc_stats: int,
) -> str:
    signal = signal_process_name(mass_key)
    channel_names = [channel["name"] for channel in channels]
    background_names = sorted(
        {
            process
            for channel in channels
            for process in channel["backgrounds"]
        }
    )
    background_ids = {
        process: index + 1
        for index, process in enumerate(background_names)
    }
    columns: list[tuple[str, str, int]] = []
    for channel in channels:
        channel_name = channel["name"]
        signal_yield = (
            summary["signals"][mass_key]["channels"].get(channel_name, 0.0)
        )
        if signal_yield > 0.0:
            columns.append((channel_name, signal, 0))
        for process in sorted(channel["backgrounds"]):
            columns.append(
                (channel_name, process, background_ids[process])
            )
    nuisances = sorted(
        {
            nuisance
            for channel in channels
            for record in channel["backgrounds"].values()
            for nuisance in record["variations"]
        }
        | set(summary["signals"][mass_key]["weight_nuisances"])
    )
    lines = [
        "# 2024 bin-by-bin transfer-factor model: High-dM 60-bin SR with "
        "five CRs split into Nb=1,2,>=3 and six recoil bins; Low-dM "
        "34-bin SR with five identically binned CRs split into Nb=1,>=2.",
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {stable_path(template_root)} $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
        "------------",
        "bin " + " ".join(channel_names),
        "observation " + " ".join(["-1"] * len(channel_names)),
        "------------",
        "bin " + " ".join(column[0] for column in columns),
        "process " + " ".join(column[1] for column in columns),
        "process " + " ".join(str(column[2]) for column in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
    ]
    signal_factors = summary["signals"][mass_key]["nuisance_factors"]
    for nuisance in nuisances:
        mask = []
        for channel_name, process, _process_id in columns:
            if process == signal:
                factors = (
                    signal_factors.get(channel_name) or {}
                ).get(nuisance)
            else:
                process_summary = (
                    summary["channels"][channel_name]["backgrounds"].get(
                        process
                    )
                    or {}
                )
                factors = (
                    process_summary.get("nuisance_factors") or {}
                ).get(nuisance)
            if not factors:
                mask.append("-")
                continue
            mask.append(
                f"{float(factors['down']):.8g}/"
                f"{float(factors['up']):.8g}"
            )
        lines.append(nuisance + " lnN " + " ".join(mask))
    lines.append(
        LUMI_NAME
        + " lnN "
        + " ".join(f"{LUMI_LNN:.3f}" for _ in columns)
    )
    rate_lines = []
    for channel in channels:
        for process, parameter in sorted(channel["rate_params"].items()):
            if process in channel["backgrounds"]:
                rate_lines.append(
                    f"{parameter} rateParam {channel['name']} {process} 1 [0,5]"
                )
    lines.extend(rate_lines)
    if auto_mc_stats >= 0:
        lines.append(f"* autoMCStats {auto_mc_stats}")
    lines.append(
        "# Background grouping: Top=TT+ST; VV_VVV is displayed as VV+VVV; "
        "PhotonJet is displayed as Photon+jet."
    )
    return "\n".join(lines) + "\n"


def write_cards(
    channels: list[dict[str, Any]],
    masses: list[str],
    template_root: Path,
    summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for mass_key in masses:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(
            datacard_text(
                template_root,
                channels,
                mass_key,
                summary,
                auto_mc_stats,
            )
        )
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", type=Path, required=True)
    parser.add_argument("--exact-transfer-inputs", type=Path, required=True)
    parser.add_argument("--transfer-factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    hists = read_json(args.hists)
    exact = read_json(args.exact_transfer_inputs)
    transfer = read_json(args.transfer_factors)
    if exact.get("status") != "complete":
        raise SystemExit(f"exact transfer inputs are not complete: {exact.get('status')}")
    if transfer.get("status") != "complete":
        raise SystemExit(f"transfer factors are not complete: {transfer.get('status')}")
    channels = build_channels(exact)
    masses = mass_points(hists, args.only, args.max_mstop)
    if not masses:
        raise SystemExit("no signal mass points selected")

    output_dir = args.output_dir
    template_root = output_dir / "templates_highdm60_lowdm34_nb_recoil_tf.root"
    card_dir = output_dir / "datacards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_combine_expected.sh"
    summary = build_root(channels, hists, masses, template_root)
    cards = write_cards(
        channels,
        masses,
        template_root,
        summary,
        card_dir,
        args.auto_mc_stats,
    )
    write_parallel_runner(
        cards,
        limit_dir,
        runner,
        args.runner_jobs,
        args.point_timeout,
    )
    limits = collect_limits(
        limit_dir,
        masses,
        output_dir / "expected_limits.json",
    )
    contour_png = output_dir / "expected_limit_contour_highdm60_lowdm34_nb_recoil_tf.png"
    contour_pdf = output_dir / "expected_limit_contour_highdm60_lowdm34_nb_recoil_tf.pdf"
    contour_complete = False
    if limits["status"] in {"complete", "partial"}:
        contour_complete = plot_contour(
            limits,
            contour_png,
            analysis_label=r"High-$\Delta m$ 60-bin + Low-$\Delta m$ 34-bin, $N_b$-recoil TF",
            x_max=float(args.max_mstop),
        )
        if contour_complete:
            import matplotlib.pyplot as plt

            plt.close("all")
            # plot_contour writes the PDF sibling itself.
            contour_pdf = contour_png.with_suffix(".pdf")
    rate_parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel["rate_params"].values()
        }
    )
    manifest = {
        "status": (
            "combine_outputs_complete"
            if limits["status"] == "complete" and contour_complete
            else "combine_inputs_ready"
        ),
        "schema_version": "highdm60_lowdm34_nb_recoil_tf_2024_v1",
        "hists": str(args.hists),
        "exact_transfer_inputs": str(args.exact_transfer_inputs),
        "transfer_factors": str(args.transfer_factors),
        "template_root": str(template_root),
        "datacard_dir": str(card_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "mass_points": masses,
        "mass_point_count": len(masses),
        "max_mstop": args.max_mstop,
        "channels": {
            "total": len(channels),
            "highdm_control": sum(
                channel["kind"] == "highdm_control_nb_recoil_onebin"
                for channel in channels
            ),
            "highdm_signal": sum(
                channel["kind"] == "highdm_signal_searchbin"
                for channel in channels
            ),
            "lowdm_control": sum(
                channel["kind"] == "lowdm_control_searchbin"
                for channel in channels
            ),
            "lowdm_signal": sum(
                channel["kind"] == "lowdm_signal_searchbin"
                for channel in channels
            ),
        },
        "transfer_parameter_count": len(rate_parameters),
        "transfer_parameters": rate_parameters,
        "auto_mc_stats": args.auto_mc_stats,
        "weight_nuisance_representation": {
            "type": "asymmetric_lnN",
            "scope": "exact Up/Down yield ratios per one-bin channel",
            "correlations": "shared nuisance name across all affected channels and processes",
            "reason": "mathematically equivalent to a shape template for a one-bin channel",
        },
        "shape_template_regularization": {
            "scope": "nonpositive or near-zero varied one-bin templates only",
            "minimum_ratio_to_nominal": MIN_VARIATION_RATIO,
            "nominal_rates_unchanged": True,
        },
        "background_grouping": background_grouping_contract(),
        "root_summary": summary,
        "limits": limits,
        "contour_complete": contour_complete,
    }
    write_json(output_dir / "combine_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "channels": manifest["channels"],
                "transfer_parameters": len(rate_parameters),
                "mass_points": len(masses),
                "limits": limits["status"],
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

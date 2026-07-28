#!/usr/bin/env python3
"""Build the AN-style 2024 High-dM60 + Low-dM34 Combine model.

Lost-lepton and QCD backgrounds are constrained by direct CR-to-SR
rateParams.  Z->nunu uses a separately measured R_Z normalization and a
photon-CR S_gamma shape rateParam, matching the likelihood structure in the
analysis note.  The dilepton samples therefore do not enter the simultaneous
fit as direct transfer-factor control channels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from background_process_groups import (
    BACKGROUND_PROCESS_ORDER,
    background_grouping_contract,
)
from build_boosted_an17_combine_inputs import (
    LUMI_LNN,
    LUMI_NAME,
    signal_process_name,
    stable_path,
    write_hist,
    write_json,
)
from build_combine_inputs_from_preview import collect_limits, plot_contour
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner
from build_nb_recoil_tf_combine_inputs_2024 import (
    HIGH_GROUPS,
    HIGH_SCHEME,
    LOW_SCHEME,
    MIN_BIN,
    build_root,
    mass_points,
    one_bin_background,
    sum_one_bin_backgrounds,
)


HIGH_CONTROL_REGIONS = ("LLCR", "QCDCR", "GCR")
LOW_CONTROL_REGIONS = HIGH_CONTROL_REGIONS
RARE_PROCESSES = ("VV_VVV", "DY", "PhotonJet")
CONTROLLED_PROCESSES = ("Top", "WtoLNu", "QCD", "Zto2Nu")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def scaled_record(
    record: dict[str, Any] | None, scale: float
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "nominal": record["nominal"] * scale,
        "sumw2": record["sumw2"] * scale * scale,
        "variations": {
            nuisance: {
                "up": pair["up"] * scale,
                "down": pair["down"] * scale,
            }
            for nuisance, pair in record["variations"].items()
        },
    }


def low_family(label: str) -> str:
    return label.rsplit("_recoil_", 1)[0]


def require_factor(record: dict[str, Any], label: str) -> float:
    if record.get("status") != "complete":
        raise ValueError(f"{label} is not complete: {record}")
    value = float(record["value"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} is nonpositive/nonfinite: {value}")
    return value


def initial_factor(record: dict[str, Any]) -> float:
    """Return a safe fit seed without converting a sparse bin into a prior."""

    if record.get("status") == "complete":
        value = float(record["value"])
        if np.isfinite(value) and value > 0.0:
            return value
    return 1.0


def rz_payload(
    factors: dict[str, Any], regime: str, group: str
) -> tuple[float, float, str]:
    rz_group = "Nb1" if group == "Nb1" else "Nb2plus"
    record = factors["RZ"][regime]["combined"][rz_group]
    if record.get("status") != "complete":
        raise ValueError(f"RZ {regime}/{rz_group} unavailable: {record}")
    value = float(record["RZ"])
    stat = float(record["RZ_stat"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"RZ {regime}/{rz_group} invalid: {value}")
    return value, stat, f"RZ_{regime}_{rz_group}"


def high_parameter(kind: str, group: str, recoil_bin: int) -> str:
    return f"{kind}_high_{group}_u{recoil_bin}"


def low_parameter(kind: str, group: str, source_bin: int) -> str:
    return f"{kind}_low_{group}_b{source_bin:02d}"


def low_shared_sgamma_payload(
    factors: dict[str, Any],
    label: str,
) -> tuple[str, float, str, int]:
    match = re.fullmatch(
        r"(Nb1|Nb2plus)_(PISR300to500|PISR500plus)_.+_recoil_(\d+)",
        label,
    )
    if not match:
        raise ValueError(f"cannot map Low-dM Sgamma sharing for {label}")
    group, isr_group, one_based_bin = match.groups()
    recoil_bin = int(one_based_bin) - 1
    shared_key = f"{group}_{isr_group}"
    shared = factors["photon"]["lowdm_nb_isr_shared"][shared_key]
    if shared.get("group") != group or shared.get("isr_group") != isr_group:
        raise ValueError(
            f"Low-dM shared Sgamma metadata mismatch for {shared_key}"
        )
    bins = shared["bins"]
    if recoil_bin < 0 or recoil_bin >= len(bins):
        raise ValueError(
            f"Low-dM shared Sgamma bin out of range for {label}: "
            f"{recoil_bin}/{len(bins)}"
        )
    parameter = (
        f"RZshape_low_{group}_{isr_group}_u{recoil_bin}"
    )
    return (
        parameter,
        initial_factor(bins[recoil_bin]),
        shared_key,
        recoil_bin,
    )


def build_channels(
    exact: dict[str, Any],
    factors: dict[str, Any],
    measurement: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    channels: list[dict[str, Any]] = []
    high = exact["highdm"]
    high_nbin = len(high["recoil_edges"]) - 1
    photon_high = factors["photon"]["highdm"]
    data_high = measurement["gcr_data"]["highdm"]["yields"]

    for region in HIGH_CONTROL_REGIONS:
        for group in HIGH_GROUPS:
            by_sample = high["recoil"][region][group]
            q = (
                require_factor(photon_high[group]["Q"], f"Q high/{group}")
                if region == "GCR"
                else 1.0
            )
            for recoil_bin in range(high_nbin):
                backgrounds = {}
                rate_params: dict[str, str] = {}
                rate_initial: dict[str, float] = {}
                for process in BACKGROUND_PROCESS_ORDER:
                    record = one_bin_background(
                        by_sample, process, recoil_bin, high_nbin
                    )
                    if region == "GCR" and process == "PhotonJet":
                        record = scaled_record(record, q)
                    if record is not None:
                        backgrounds[process] = record
                if region == "LLCR":
                    parameter = high_parameter(
                        "RLL", group, recoil_bin
                    )
                    for process in ("Top", "WtoLNu"):
                        if process in backgrounds:
                            rate_params[process] = parameter
                            rate_initial[process] = 1.0
                elif region == "QCDCR" and "QCD" in backgrounds:
                    parameter = high_parameter(
                        "RQCD", group, recoil_bin
                    )
                    rate_params["QCD"] = parameter
                    rate_initial["QCD"] = 1.0
                elif region == "GCR" and "PhotonJet" in backgrounds:
                    parameter = high_parameter(
                        "RZshape", group, recoil_bin
                    )
                    sgamma = initial_factor(
                        photon_high[group]["bins"][recoil_bin]["Sgamma"],
                    )
                    rate_params["PhotonJet"] = parameter
                    rate_initial["PhotonJet"] = sgamma
                observation = None
                if region == "GCR":
                    observation = float(
                        (
                            (data_high.get(group) or {}).get(
                                str(recoil_bin), {}
                            )
                            or {}
                        ).get("sumw", 0.0)
                    )
                channels.append(
                    {
                        "name": f"h{region}_{group}_u{recoil_bin}",
                        "kind": "highdm_an_control",
                        "regime": "highdm",
                        "region": region,
                        "nb_group": group,
                        "source_bin": recoil_bin,
                        "backgrounds": backgrounds,
                        "rate_params": rate_params,
                        "rate_initial": rate_initial,
                        "signal_source": None,
                        "observation": observation,
                        "rz_lnN": {},
                    }
                )

    high_components = high["sr_components"]
    high_bins = 60
    for source_bin in range(high_bins):
        recoil_bin = source_bin % high_nbin
        backgrounds = {}
        rate_params = {}
        rate_initial = {}
        rz_lnN = {}
        for group in HIGH_GROUPS:
            by_sample = high_components[group]
            for process in CONTROLLED_PROCESSES:
                record = one_bin_background(
                    by_sample, process, source_bin, high_bins
                )
                if record is None:
                    continue
                component = f"{process}_{group}"
                if process == "Zto2Nu":
                    rz, rz_stat, nuisance = rz_payload(
                        factors, "highdm", group
                    )
                    record = scaled_record(record, rz)
                    parameter = high_parameter(
                        "RZshape", group, recoil_bin
                    )
                    sgamma = initial_factor(
                        photon_high[group]["bins"][recoil_bin][
                            "Sgamma"
                        ],
                    )
                    rz_lnN[component] = {
                        "name": nuisance,
                        "factor": 1.0 + rz_stat / rz,
                    }
                elif process in ("Top", "WtoLNu"):
                    parameter = high_parameter(
                        "RLL", group, recoil_bin
                    )
                    sgamma = 1.0
                else:
                    parameter = high_parameter(
                        "RQCD", group, recoil_bin
                    )
                    sgamma = 1.0
                backgrounds[component] = record
                rate_params[component] = parameter
                rate_initial[component] = sgamma
        for process in RARE_PROCESSES:
            record = sum_one_bin_backgrounds(
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
            if record is not None:
                backgrounds[process] = record
        channels.append(
            {
                "name": f"hSR_b{source_bin:02d}",
                "kind": "highdm_signal_searchbin",
                "regime": "highdm",
                "region": "SR",
                "source_bin": source_bin,
                "backgrounds": backgrounds,
                "rate_params": rate_params,
                "rate_initial": rate_initial,
                "signal_source": ("highdm", source_bin),
                "observation": None,
                "rz_lnN": rz_lnN,
            }
        )

    low = exact["lowdm"]
    labels = low["search_bin_labels"]
    low_nbin = len(labels)
    photon_low = factors["photon"]["lowdm"]
    data_low = measurement["gcr_data"]["lowdm"]["yields"]
    for region in LOW_CONTROL_REGIONS:
        for source_bin, label in enumerate(labels):
            group = "Nb1" if source_bin < 16 else "Nb2plus"
            by_sample = low["search_components"][region][group]
            family = low_family(label)
            q = (
                require_factor(
                    photon_low[family]["Q"], f"Q low/{family}"
                )
                if region == "GCR"
                else 1.0
            )
            backgrounds = {}
            rate_params = {}
            rate_initial = {}
            shared_key = None
            shared_bin = None
            for process in BACKGROUND_PROCESS_ORDER:
                record = one_bin_background(
                    by_sample, process, source_bin, low_nbin
                )
                if region == "GCR" and process == "PhotonJet":
                    record = scaled_record(record, q)
                if record is not None:
                    backgrounds[process] = record
            if region == "LLCR":
                parameter = low_parameter("RLL", group, source_bin)
                for process in ("Top", "WtoLNu"):
                    if process in backgrounds:
                        rate_params[process] = parameter
                        rate_initial[process] = 1.0
            elif region == "QCDCR" and "QCD" in backgrounds:
                parameter = low_parameter("RQCD", group, source_bin)
                rate_params["QCD"] = parameter
                rate_initial["QCD"] = 1.0
            elif region == "GCR" and "PhotonJet" in backgrounds:
                parameter, sgamma, shared_key, shared_bin = (
                    low_shared_sgamma_payload(factors, label)
                )
                rate_params["PhotonJet"] = parameter
                rate_initial["PhotonJet"] = sgamma
            else:
                shared_key = None
                shared_bin = None
            observation = (
                float(
                    (data_low.get(str(source_bin), {}) or {}).get(
                        "sumw", 0.0
                    )
                )
                if region == "GCR"
                else None
            )
            channels.append(
                {
                    "name": f"l{region}_b{source_bin:02d}",
                    "kind": "lowdm_an_control",
                    "regime": "lowdm",
                    "region": region,
                    "nb_group": group,
                    "source_bin": source_bin,
                    "bin_label": label,
                    "sgamma_shared_group": shared_key,
                    "sgamma_shared_bin": shared_bin,
                    "backgrounds": backgrounds,
                    "rate_params": rate_params,
                    "rate_initial": rate_initial,
                    "signal_source": None,
                    "observation": observation,
                    "rz_lnN": {},
                }
            )

    for source_bin, label in enumerate(labels):
        group = "Nb1" if source_bin < 16 else "Nb2plus"
        by_sample = low["search_components"]["SR"][group]
        (
            shared_parameter,
            shared_sgamma,
            shared_key,
            shared_bin,
        ) = low_shared_sgamma_payload(factors, label)
        backgrounds = {}
        rate_params = {}
        rate_initial = {}
        rz_lnN = {}
        for process in BACKGROUND_PROCESS_ORDER:
            record = one_bin_background(
                by_sample, process, source_bin, low_nbin
            )
            if record is None:
                continue
            if process == "Zto2Nu":
                rz, rz_stat, nuisance = rz_payload(
                    factors, "lowdm", group
                )
                record = scaled_record(record, rz)
                parameter = shared_parameter
                sgamma = shared_sgamma
                rz_lnN[process] = {
                    "name": nuisance,
                    "factor": 1.0 + rz_stat / rz,
                }
            elif process in ("Top", "WtoLNu"):
                parameter = low_parameter("RLL", group, source_bin)
                sgamma = 1.0
            elif process == "QCD":
                parameter = low_parameter("RQCD", group, source_bin)
                sgamma = 1.0
            else:
                parameter = None
                sgamma = 1.0
            backgrounds[process] = record
            if parameter:
                rate_params[process] = parameter
                rate_initial[process] = sgamma
        channels.append(
            {
                "name": f"lSR_b{source_bin:02d}",
                "kind": "lowdm_signal_searchbin",
                "regime": "lowdm",
                "region": "SR",
                "nb_group": group,
                "source_bin": source_bin,
                "bin_label": label,
                "sgamma_shared_group": shared_key,
                "sgamma_shared_bin": shared_bin,
                "backgrounds": backgrounds,
                "rate_params": rate_params,
                "rate_initial": rate_initial,
                "signal_source": ("lowdm", source_bin),
                "observation": None,
                "rz_lnN": rz_lnN,
            }
        )

    scopes: dict[str, set[str]] = {}
    for channel in channels:
        scope = "sr" if channel["region"] == "SR" else "cr"
        for parameter in channel["rate_params"].values():
            scopes.setdefault(parameter, set()).add(scope)
    invalid = sorted(
        parameter for parameter, scope in scopes.items() if scope != {"cr", "sr"}
    )
    invalid_set = set(invalid)
    for channel in channels:
        channel["rate_params"] = {
            process: parameter
            for process, parameter in channel["rate_params"].items()
            if parameter not in invalid_set
        }
        channel["rate_initial"] = {
            process: value
            for process, value in channel["rate_initial"].items()
            if process in channel["rate_params"]
        }
    return channels, invalid


def overwrite_gcr_observations(
    output_root: Path,
    channels: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    import ROOT

    root_file = ROOT.TFile(str(output_root), "UPDATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot update {output_root}")
    try:
        for channel in channels:
            observation = channel.get("observation")
            if observation is None:
                continue
            directory = root_file.GetDirectory(channel["name"])
            if not directory:
                raise RuntimeError(
                    f"missing ROOT directory {channel['name']}"
                )
            value = np.asarray([max(float(observation), 0.0)])
            write_hist(
                directory,
                "data_obs",
                value,
                value,
                np.asarray([0.0, 1.0]),
            )
            summary["channels"][channel["name"]]["observation"] = float(
                observation
            )
            summary["channels"][channel["name"]][
                "observation_source"
            ] = "2024 data"
    finally:
        root_file.Close()


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    summary: dict[str, Any],
    auto_mc_stats: int,
) -> str:
    signal = signal_process_name(mass_key)
    channel_names = [channel["name"] for channel in channels]
    backgrounds = sorted(
        {
            process
            for channel in channels
            for process in channel["backgrounds"]
        }
    )
    background_ids = {
        process: index + 1 for index, process in enumerate(backgrounds)
    }
    channel_map = {channel["name"]: channel for channel in channels}
    columns: list[tuple[str, str, int]] = []
    for channel in channels:
        name = channel["name"]
        if summary["signals"][mass_key]["channels"].get(name, 0.0) > 0.0:
            columns.append((name, signal, 0))
        for process in sorted(channel["backgrounds"]):
            columns.append((name, process, background_ids[process]))
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
        "# 2024 AN-style Zinv model: RZ from off/on-Z dilepton matrix; "
        "Q-normalized photon CR supplies Sgamma; Low-dM Sgamma is shared "
        "within four Nb x ISR-pT groups and 14 recoil-shape bins; only RLL "
        "and RQCD are direct CR/SR transfer factors.",
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {stable_path(template_root)} $CHANNEL/$PROCESS",
        "------------",
        "bin " + " ".join(channel_names),
        "observation " + " ".join(["-1"] * len(channel_names)),
        "------------",
        "bin " + " ".join(item[0] for item in columns),
        "process " + " ".join(item[1] for item in columns),
        "process " + " ".join(str(item[2]) for item in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
    ]
    signal_factors = summary["signals"][mass_key]["nuisance_factors"]
    for nuisance in nuisances:
        mask = []
        for channel_name, process, _ in columns:
            if process == signal:
                pair = (signal_factors.get(channel_name) or {}).get(
                    nuisance
                )
            else:
                pair = (
                    (
                        summary["channels"][channel_name]["backgrounds"].get(
                            process
                        )
                        or {}
                    ).get("nuisance_factors")
                    or {}
                ).get(nuisance)
            mask.append(
                "-"
                if not pair
                else f"{pair['down']:.8g}/{pair['up']:.8g}"
            )
        lines.append(nuisance + " lnN " + " ".join(mask))
    lines.append(
        LUMI_NAME
        + " lnN "
        + " ".join(
            (
                f"{LUMI_LNN:.3f}"
                if process == signal
                or (
                    process in {"VV_VVV", "DY", "PhotonJet"}
                    and process
                    not in channel_map[_channel]["rate_params"]
                )
                else "-"
            )
            for _channel, process, _ in columns
        )
    )
    rz_names = sorted(
        {
            item["name"]
            for channel in channels
            for item in channel["rz_lnN"].values()
        }
    )
    for nuisance in rz_names:
        mask = []
        for channel_name, process, _ in columns:
            item = channel_map[channel_name]["rz_lnN"].get(process)
            mask.append(
                f"{item['factor']:.8g}"
                if item and item["name"] == nuisance
                else "-"
            )
        lines.append(nuisance + " lnN " + " ".join(mask))
    rate_lines = []
    for channel in channels:
        for process, parameter in sorted(channel["rate_params"].items()):
            initial = float(channel["rate_initial"].get(process, 1.0))
            initial = min(max(initial, 1.0e-4), 4.999)
            rate_lines.append(
                f"{parameter} rateParam {channel['name']} {process} "
                f"{initial:.8g} [0,5]"
            )
    lines.extend(rate_lines)
    if auto_mc_stats >= 0:
        lines.append(f"* autoMCStats {auto_mc_stats}")
    lines.extend(
        [
            "# Grouping: Top=TT+ST; VV_VVV is displayed as VV+VVV; "
            "PhotonJet is displayed as Photon+jet.",
            "# DY2E/DY2M are measurement-only inputs for RZ and are not "
            "direct likelihood channels.",
        ]
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
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    hists = read_json(args.hists)
    exact = read_json(args.exact_inputs)
    measurement = read_json(args.measurement)
    factors = read_json(args.factors)
    if exact.get("status") != "complete":
        raise SystemExit(f"exact inputs incomplete: {exact.get('status')}")
    if factors.get("status") != "complete":
        raise SystemExit(f"factors incomplete: {factors.get('status')}")
    channels, unmatched_rate_parameters = build_channels(
        exact, factors, measurement
    )
    masses = mass_points(hists, args.only, args.max_mstop)
    if not masses:
        raise SystemExit("no signal mass points selected")
    output_dir = args.output_dir
    template_root = output_dir / "templates_highdm60_lowdm34_an_zinv.root"
    card_dir = output_dir / "datacards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_combine_expected.sh"
    summary = build_root(channels, hists, masses, template_root)
    overwrite_gcr_observations(template_root, channels, summary)
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
        limit_dir, masses, output_dir / "expected_limits.json"
    )
    contour_png = (
        output_dir / "expected_limit_highdm60_lowdm34_an_zinv_x1800.png"
    )
    contour_complete = False
    if limits["status"] in {"complete", "partial"}:
        contour_complete = plot_contour(
            limits,
            contour_png,
            analysis_label=(
                r"2024 High-$\Delta m$ 60-bin + Low-$\Delta m$ 34-bin, "
                r"AN-style $Z\to\nu\bar{\nu}$"
            ),
            x_max=float(args.max_mstop),
        )
    rate_parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel["rate_params"].values()
        }
    )
    low_sgamma_parameters = [
        parameter
        for parameter in rate_parameters
        if parameter.startswith("RZshape_low_")
    ]
    shared_factor_groups = factors["photon"]["lowdm_nb_isr_shared"]
    expected_low_sgamma_count = sum(
        len(payload["bins"]) for payload in shared_factor_groups.values()
    )
    if set(shared_factor_groups) != {
        "Nb1_PISR300to500",
        "Nb1_PISR500plus",
        "Nb2plus_PISR300to500",
        "Nb2plus_PISR500plus",
    }:
        raise ValueError(
            "Low-dM Sgamma sharing does not contain the four adopted groups"
        )
    if (
        expected_low_sgamma_count != 14
        or len(low_sgamma_parameters) != expected_low_sgamma_count
    ):
        raise ValueError(
            "Low-dM shared Sgamma parameter count mismatch: "
            f"{len(low_sgamma_parameters)}/{expected_low_sgamma_count}"
        )
    manifest = {
        "schema_version": (
            "highdm60_lowdm34_an_zinv_2024_v2_nb_isr_shared_sgamma"
        ),
        "status": (
            "combine_outputs_complete"
            if limits["status"] == "complete" and contour_complete
            else "combine_inputs_ready"
        ),
        "model": {
            "zinv": (
                "RZ normalization from on/off-Z matrix; Q-normalized photon "
                "CR Sgamma shape"
            ),
            "lowdm_sgamma_sharing": {
                "groups": sorted(shared_factor_groups),
                "recoil_shape_parameter_count": len(
                    low_sgamma_parameters
                ),
                "mapping": (
                    "one Sgamma rateParam per Nb x ISR-pT x recoil bin, "
                    "shared across the Low-dM PTb/Nj categories"
                ),
            },
            "lost_lepton": "direct shared RLL CR/SR rateParams",
            "qcd": "direct RQCD CR/SR rateParams",
            "dilepton_in_likelihood": False,
            "gcr_observation": "2024 data",
            "other_observations": "background-only Asimov for expected result",
        },
        "inputs": {
            "hists": str(args.hists),
            "exact": str(args.exact_inputs),
            "measurement": str(args.measurement),
            "factors": str(args.factors),
        },
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
                item["kind"] == "highdm_an_control" for item in channels
            ),
            "highdm_signal": sum(
                item["kind"] == "highdm_signal_searchbin"
                for item in channels
            ),
            "lowdm_control": sum(
                item["kind"] == "lowdm_an_control" for item in channels
            ),
            "lowdm_signal": sum(
                item["kind"] == "lowdm_signal_searchbin"
                for item in channels
            ),
        },
        "rate_parameter_count": len(rate_parameters),
        "rate_parameters": rate_parameters,
        "unmatched_rate_parameters_dropped": unmatched_rate_parameters,
        "auto_mc_stats": args.auto_mc_stats,
        "weight_nuisance_representation": {
            "type": "asymmetric_lnN",
            "scope": "exact Up/Down yield ratios per one-bin channel",
            "correlations": (
                "shared nuisance name across all affected channels/processes"
            ),
        },
        "template_regularization": {
            "minimum_nominal_rate": MIN_BIN,
            "scope": (
                "only nonpositive one-bin weighted templates needed by Combine"
            ),
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
                "rate_parameters": len(rate_parameters),
                "mass_points": len(masses),
                "limits": limits["status"],
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

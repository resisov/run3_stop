#!/usr/bin/env python3
"""Build provisional 2024 limits with externally unconstrained backgrounds.

This model deliberately removes the adopted CR-to-SR transfer-factor and
R_Z/S_gamma background-estimation constraints.  Each canonical background
group instead receives one global rateParam shared across every High-dM and
Low-dM control and signal channel.  The parameter has no external prior; the
finite range is only a numerical positivity guard for Combine.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_combine_inputs_from_preview import collect_limits, plot_contour
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner
from build_nb_recoil_tf_combine_inputs_2024 import (
    DEFAULT_SIGNAL_TOPOLOGY,
    SUPPORTED_SIGNAL_TOPOLOGIES,
    build_channels,
    build_root,
    mass_points,
    read_json,
    sum_one_bin_backgrounds,
)
from build_boosted_an17_combine_inputs import (
    LUMI_LNN,
    signal_process_name,
    stable_path,
    write_json,
)


DECAY_LABELS = {
    "T2tt": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow t\tilde{\chi}_1^0$"
    ),
    "T2tb": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^{+*}\tilde{\chi}_1^0\ (50\%),\ "
        r"\bar{\tilde{t}}_1\rightarrow\bar{t}\tilde{\chi}_1^0\ (50\%)"
        r"$"
    ),
    "T2bW": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^+\tilde{\chi}_1^0"
        r"$"
    ),
}

NPS_LUMI_NAME = "lumi_13p6TeV_2024"
NPS_NUISANCE_NAMES = {
    "btagSF_bc_correlated": "CMS_btag_fixedWP_bc_correlated",
    "btagSF_bc_uncorrelated": "CMS_btag_fixedWP_bc_uncorrelated_2024",
    "btagSF_light_correlated": "CMS_btag_fixedWP_light_correlated",
    "btagSF_light_uncorrelated": (
        "CMS_btag_fixedWP_light_uncorrelated_2024"
    ),
    "electron_hlt": "CMS_SUS26090_eff_e_trigger_2024",
    "electron_id": "CMS_eff_e_id_13p6TeV",
    "met_trigger": "CMS_SUS26090_eff_met_trigger_2024",
    "muon_hlt": "CMS_SUS26090_eff_m_trigger_2024",
    "muon_id": "CMS_SUS26090_eff_m_id_2024",
    "photon_id": "CMS_eff_g_id_13p6TeV",
    "photon_trigger": "CMS_SUS26090_eff_g_trigger_2024",
    "pileup": "CMS_pileup",
}
LOW_SHARED_CR_LABEL = re.compile(
    r"(Nb1|Nb2plus)_(PISR300to500|PISR500plus)_.+_recoil_(\d+)"
)


def canonical_background(process: str) -> str:
    for name in ("Top", "WtoLNu", "QCD", "Zto2Nu"):
        if process == name or process.startswith(name + "_"):
            return name
    return process


def nps_nuisance_name(name: str) -> str:
    return NPS_NUISANCE_NAMES.get(name, name)


def share_lowdm_control_channels(
    channels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Share Low-dM controls by Nb x ISR-pT x recoil bin.

    The adopted Low-dM SR categories retain their 34-bin definition.  Only
    the five control regions are summed over the pTb/Njet category axes that
    share identical recoil boundaries.  No control observation is copied.
    """

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    retained: list[dict[str, Any]] = []
    insertion_index: int | None = None
    for channel in channels:
        if channel["kind"] != "lowdm_control_searchbin":
            retained.append(channel)
            continue
        if insertion_index is None:
            insertion_index = len(retained)
        match = LOW_SHARED_CR_LABEL.fullmatch(channel["bin_label"])
        if not match:
            raise ValueError(
                "cannot share Low-dM control channel with label "
                f"{channel['bin_label']!r}"
            )
        nb_group, isr_group, one_based_recoil = match.groups()
        key = (
            channel["region"],
            nb_group,
            isr_group,
            int(one_based_recoil) - 1,
        )
        grouped.setdefault(key, []).append(channel)

    shared: list[dict[str, Any]] = []
    for (region, nb_group, isr_group, recoil_bin), sources in sorted(
        grouped.items()
    ):
        processes = sorted(
            {
                process
                for channel in sources
                for process in channel["backgrounds"]
            }
        )
        backgrounds = {}
        for process in processes:
            record = sum_one_bin_backgrounds(
                [channel["backgrounds"].get(process) for channel in sources]
            )
            if record is not None:
                backgrounds[process] = record
        shared.append(
            {
                "name": (
                    f"l{region}_{nb_group}_{isr_group}_u{recoil_bin}"
                ),
                "kind": "lowdm_control_nb_isr_recoil_shared",
                "regime": "lowdm",
                "region": region,
                "nb_group": nb_group,
                "source_bin": recoil_bin,
                "bin_label": (
                    f"{nb_group}_{isr_group}_recoil_{recoil_bin + 1}"
                ),
                "backgrounds": backgrounds,
                "rate_params": {},
                "signal_source": None,
            }
        )

    expected = 5 * (4 + 4 + 3 + 3)
    if len(shared) != expected:
        raise ValueError(
            f"unexpected shared Low-dM control count: {len(shared)} != "
            f"{expected}"
        )
    if insertion_index is None:
        raise ValueError("no Low-dM control channels found")
    return retained[:insertion_index] + shared + retained[insertion_index:]


def make_backgrounds_free(
    channels: list[dict[str, Any]],
) -> list[str]:
    parameters: set[str] = set()
    for channel in channels:
        rate_params = {}
        for process in channel["backgrounds"]:
            parameter = (
                "CMS_SUS26090_bkgNorm_"
                + canonical_background(process)
            )
            rate_params[process] = parameter
            parameters.add(parameter)
        channel["rate_params"] = rate_params
    return sorted(parameters)


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    summary: dict[str, Any],
    auto_mc_stats: int,
) -> str:
    template_reference = (
        stable_path(template_root)
        if template_root.is_absolute()
        else str(template_root)
    )
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
        if summary["signals"][mass_key]["channels"].get(channel_name, 0.0) > 0:
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
        "# Provisional 2024 free-background model.  No transfer-factor, "
        "R_Z, S_gamma, lost-lepton, or QCD background-estimation constraint "
        "is applied.  Canonical background normalizations are global "
        "rateParams without external priors.",
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        (
            f"shapes * * {template_reference} "
            "$CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC"
        ),
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
            else:
                mask.append(
                    f"{float(factors['down']):.8g}/"
                    f"{float(factors['up']):.8g}"
                )
        lines.append(nps_nuisance_name(nuisance) + " lnN " + " ".join(mask))

    # Background luminosity normalization is redundant with the free
    # rateParams.  Retain the luminosity nuisance on the signal only.
    lines.append(
        NPS_LUMI_NAME
        + " lnN "
        + " ".join(
            f"{LUMI_LNN:.3f}" if process == signal else "-"
            for _channel, process, _process_id in columns
        )
    )
    for channel in channels:
        for process, parameter in sorted(channel["rate_params"].items()):
            lines.append(
                f"{parameter} rateParam {channel['name']} "
                f"{process} 1 [0,10]"
            )
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
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topology",
        choices=SUPPORTED_SIGNAL_TOPOLOGIES,
        default=DEFAULT_SIGNAL_TOPOLOGY,
    )
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    hists = read_json(args.hists)
    exact = read_json(args.exact_inputs)
    if exact.get("status") != "complete":
        raise SystemExit(
            f"exact inputs are not complete: {exact.get('status')}"
        )
    channels = share_lowdm_control_channels(build_channels(exact))
    free_parameters = make_backgrounds_free(channels)
    masses = mass_points(
        hists,
        args.only,
        args.max_mstop,
        args.topology,
    )
    if not masses:
        raise SystemExit(
            f"no {args.topology} signal mass points selected"
        )

    output_dir = args.output_dir
    topology_lower = args.topology.lower()
    template_root = (
        output_dir
        / f"templates_highdm60_lowdm34_free_background_{topology_lower}.root"
    )
    card_dir = output_dir / "datacards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_combine_expected.sh"
    if args.collect_only:
        summary = read_json(output_dir / "template_summary.json")
        cards = {
            mass_key: str(card_dir / f"datacard_{mass_key}.txt")
            for mass_key in masses
        }
    else:
        summary = build_root(
            channels,
            hists,
            masses,
            template_root,
            args.topology,
        )
        write_json(output_dir / "template_summary.json", summary)
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
    contour_png = (
        output_dir
        / (
            f"expected_limit_{topology_lower}_"
            "highdm60_lowdm34_free_background_x1800.png"
        )
    )
    contour_complete = False
    if limits["status"] in {"complete", "partial"}:
        run2_contour_paths = {
            "T2tt": Path(
                "/eos/user/t/taiwoo/run2_sus19010_contours.json"
            ),
            "T2tb": Path(
                "/eos/user/t/taiwoo/run2_sus19010_contours_t2tb.json"
            ),
            "T2bW": Path(
                "/eos/user/t/taiwoo/run2_sus19010_contours_t2bw.json"
            ),
        }
        contour_complete = plot_contour(
            limits,
            contour_png,
            run2_contours=run2_contour_paths[args.topology],
            luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
            analysis_label=None,
            x_max=float(args.max_mstop),
            decay_label=DECAY_LABELS[args.topology],
        )

    output_status = "combine_inputs_ready"
    if contour_complete:
        if limits["status"] == "complete":
            output_status = "combine_outputs_complete"
        elif limits["status"] == "partial":
            output_status = "combine_outputs_partial"
    manifest = {
        "status": output_status,
        "schema_version": (
            "highdm60_lowdm34_sharedcr_free_background_2024_v1"
        ),
        "model": "free_background_global_process_normalizations",
        "signal_topology": args.topology,
        "hists": str(args.hists),
        "exact_inputs": str(args.exact_inputs),
        "template_root": str(template_root),
        "template_summary": str(output_dir / "template_summary.json"),
        "datacard_dir": str(card_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "mass_points": masses,
        "mass_point_count": len(masses),
        "max_mstop": args.max_mstop,
        "channels": {
            "total": len(channels),
            "highdm_control": sum(
                channel["kind"].startswith("highdm_control")
                for channel in channels
            ),
            "highdm_signal": sum(
                channel["kind"] == "highdm_signal_searchbin"
                for channel in channels
            ),
            "lowdm_control": sum(
                channel["kind"].startswith("lowdm_control")
                for channel in channels
            ),
            "lowdm_signal": sum(
                channel["kind"] == "lowdm_signal_searchbin"
                for channel in channels
            ),
        },
        "free_background_parameters": free_parameters,
        "free_background_parameter_count": len(free_parameters),
        "external_background_constraints": [],
        "numerical_rate_parameter_range": [0.0, 10.0],
        "auto_mc_stats": args.auto_mc_stats,
        "limits": limits,
        "contour_png": str(contour_png) if contour_complete else None,
        "contour_pdf": (
            str(contour_png.with_suffix(".pdf"))
            if contour_complete
            else None
        ),
        "run2_overlay": True,
        "run2_overlay_policy": (
            f"topology-matched CMS-SUS-19-010 {args.topology} "
            "observed and expected contours"
        ),
        "data_mode": "asimov",
    }
    write_json(output_dir / "combine_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "topology": args.topology,
                "mass_points": len(masses),
                "free_background_parameters": len(free_parameters),
                "limit_status": limits["status"],
                "run2_overlay": manifest["run2_overlay"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

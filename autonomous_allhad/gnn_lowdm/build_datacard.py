"""Canonical Low-dM GNN datacard writer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


CONTROL_CATEGORIES = (
    "Nb1_NISR0",
    "Nb1_NISR1plus",
    "Nb2plus_NISR0",
    "Nb2plus_NISR1plus",
)
CONTROL_REGIONS = ("LLCR", "QCDCR", "GCR")
BACKGROUND_GROUPS = ("Top", "W", "Zinv", "QCD", "DY", "Photon", "VV")
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def canonical_sr_labels() -> list[str]:
    """Return the adopted six-category, five-bin Low-dM SR definition."""
    definition = json.loads(CONFIG_PATH.read_text())["sr_binning"]
    labels = list(definition["category_labels"])
    edges = definition["edges_by_category"]
    bins = sum(len(edges[label]) - 1 for label in labels)
    if len(labels) != 6 or bins != 30 or definition["total_bins"] != 30:
        raise RuntimeError("config.json is not the adopted 6 x 5 = 30-bin SR")
    return labels


def parent_category(label: str) -> str:
    if label.startswith("Nb1_"):
        nb_group = "Nb1"
    elif label.startswith(("Nb2", "Nb3")):
        nb_group = "Nb2plus"
    else:
        raise ValueError(f"unknown SR category: {label}")
    nisr_group = "NISR0" if "NISR0" in label else "NISR1plus"
    return f"{nb_group}_{nisr_group}"


def rate_parameter_lines(
    sr_labels: list[str], active: dict[str, set[str]]
) -> list[str]:
    lines: list[str] = []
    for parent in CONTROL_CATEGORIES:
        sr_channels = [
            f"SR_{label}" for label in sr_labels if parent_category(label) == parent
        ]
        for channel in sr_channels + [f"LLCR_{parent}"]:
            for process in ("Top", "W"):
                if process in active[channel]:
                    lines.append(
                        f"ll_norm_{parent} rateParam {channel} {process} 1 [0,10]"
                    )
        for channel in sr_channels + [f"QCDCR_{parent}"]:
            if "QCD" in active[channel]:
                lines.append(
                    f"qcd_norm_{parent} rateParam {channel} QCD 1 [0,10]"
                )
        for channel in sr_channels:
            if "Zinv" in active[channel]:
                lines.append(
                    f"zinv_norm_{parent} rateParam {channel} Zinv 1 [0,10]"
                )
        if "Photon" in active[f"GCR_{parent}"]:
            lines.append(
                f"zinv_norm_{parent} rateParam GCR_{parent} Photon 1 [0,10]"
            )
    return lines


def write_cards(
    output: Path,
    template_reference: str,
    signals: dict[str, Any],
    sr_labels: list[str],
    summary: dict[str, Any],
    *,
    year: str = "2024",
    signal_name_prefix: str = "",
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    channels = [f"SR_{label}" for label in sr_labels] + [
        f"{region}_{category}"
        for region in CONTROL_REGIONS
        for category in CONTROL_CATEGORIES
    ]
    active = {
        channel: {
            process
            for process, value in summary[channel]["processes"].items()
            if value > 0.0
        }
        for channel in channels
    }
    process_ids = {
        process: index + 1 for index, process in enumerate(BACKGROUND_GROUPS)
    }
    cards: dict[str, str] = {}
    for mass_key, signal_by_category in signals.items():
        row_channels: list[str] = []
        row_processes: list[str] = []
        row_ids: list[str] = []
        row_rates: list[str] = []
        for channel in channels:
            if channel.startswith("SR_"):
                label = channel[3:]
                if float(signal_by_category[label][0].sum()) > 0.0:
                    row_channels.append(channel)
                    row_processes.append("signal")
                    row_ids.append("0")
                    row_rates.append("-1")
            for process in BACKGROUND_GROUPS:
                if process in active[channel]:
                    row_channels.append(channel)
                    row_processes.append(process)
                    row_ids.append(str(process_ids[process]))
                    row_rates.append("-1")
        lines = [
            f"imax {len(channels)}",
            "jmax *",
            "kmax *",
            "------------",
            f"shapes data_obs * {template_reference} $CHANNEL/data_obs",
            (
                f"shapes signal * {template_reference} "
                f"$CHANNEL/signal_{signal_name_prefix}{mass_key}"
            ),
            f"shapes * * {template_reference} $CHANNEL/$PROCESS",
            "------------",
            "bin " + " ".join(channels),
            "observation " + " ".join("-1" for _ in channels),
            "------------",
            "bin " + " ".join(row_channels),
            "process " + " ".join(row_processes),
            "process " + " ".join(row_ids),
            "rate " + " ".join(row_rates),
            "------------",
            f"lumi_13p6TeV_{year} lnN "
            + " ".join(
                "1.014" if process == "signal" else "-"
                for process in row_processes
            ),
            *rate_parameter_lines(sr_labels, active),
            "* autoMCStats 10 1 1",
            "",
        ]
        card = output / f"datacard_{mass_key}.txt"
        card.write_text("\n".join(lines))
        cards[mass_key] = str(card)
    return cards


def signals_from_root(
    template: Path,
    topology: str,
    sr_labels: list[str],
    requested_mass_keys: set[str] | None = None,
) -> dict[str, dict[str, tuple[np.ndarray]]]:
    """Read nominal signal availability and yields from a flat template ROOT."""
    import ROOT

    root_file = ROOT.TFile.Open(str(template), "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"cannot read {template}")
    pattern = re.compile(
        rf"^signal_{re.escape(topology)}_(mStop[0-9]+_mLSP[0-9]+)$"
    )
    masses: set[str] = set()
    try:
        for label in sr_labels:
            directory = root_file.Get(f"SR_{label}")
            if not directory:
                raise KeyError(f"missing SR_{label} in {template}")
            for key in directory.GetListOfKeys():
                match = pattern.fullmatch(key.GetName())
                if match:
                    masses.add(match.group(1))
        if requested_mass_keys is not None:
            missing = sorted(requested_mass_keys - masses)
            if missing:
                raise KeyError(f"requested signal histograms are missing: {missing}")
            masses = requested_mass_keys
        signals: dict[str, dict[str, tuple[np.ndarray]]] = {}
        for mass_key in sorted(masses):
            by_category = {}
            for label in sr_labels:
                histogram = root_file.Get(
                    f"SR_{label}/signal_{topology}_{mass_key}"
                )
                integral = float(histogram.Integral()) if histogram else 0.0
                by_category[label] = (np.asarray([integral], dtype=float),)
            signals[mass_key] = by_category
        return signals
    finally:
        root_file.Close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--topology", required=True, choices=("T2tt", "T2bW", "T2tb"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--year", required=True, choices=("2024", "2025"))
    parser.add_argument("--mass-key", action="append", default=[])
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    sr_labels = list(payload["categories"]["SR"])
    expected_labels = canonical_sr_labels()
    if sr_labels != expected_labels:
        raise RuntimeError(
            "template summary does not use the canonical 30-bin SR: "
            f"expected {expected_labels}, found {sr_labels}"
        )
    requested = set(args.mass_key) or None
    signals = signals_from_root(
        args.template, args.topology, sr_labels, requested
    )
    cards = write_cards(
        args.output,
        str(args.template.absolute()),
        signals,
        sr_labels,
        payload["channels"],
        year=args.year,
        signal_name_prefix=f"{args.topology}_",
    )
    print(json.dumps({"status": "complete", "cards": len(cards)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regularize one High-dM signal shape without changing its total yield.

This diagnostic replaces the 60-bin High-dM signal fractions at one mass
point by the component-wise median of the fractions at the nearest lower,
target, and nearest higher stop masses with the same LSP mass.  The resulting
fractions are renormalized to unity and multiplied by the target point's
original High-dM yield.  Backgrounds, control regions, Low-dM channels,
normalization, and all nuisance definitions are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path


MASS_KEY_RE = re.compile(r"^mStop(?P<mstop>\d+)_mLSP(?P<mlsp>\d+)$")
HIGHDM_CHANNELS = tuple(f"hSR_b{index:02d}" for index in range(60))


def parse_mass_key(key: str) -> tuple[int, int]:
    match = MASS_KEY_RE.fullmatch(key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group("mstop")), int(match.group("mlsp"))


def signal_process(key: str) -> str:
    return f"sig_{key}"


def read_shape(root_file: object, key: str) -> tuple[list[float], list[float]]:
    process = signal_process(key)
    values: list[float] = []
    errors: list[float] = []
    for channel in HIGHDM_CHANNELS:
        hist = root_file.Get(f"{channel}/{process}")
        if not hist:
            raise RuntimeError(f"missing histogram {channel}/{process}")
        values.append(float(hist.GetBinContent(1)))
        errors.append(float(hist.GetBinError(1)))
    return values, errors


def nearest_triplet(signals: dict[str, object], target: str) -> list[str]:
    target_stop, target_lsp = parse_mass_key(target)
    same_lsp = sorted(
        (
            (parse_mass_key(key)[0], key)
            for key in signals
            if parse_mass_key(key)[1] == target_lsp
        ),
        key=lambda item: item[0],
    )
    lower = [item for item in same_lsp if item[0] < target_stop]
    upper = [item for item in same_lsp if item[0] > target_stop]
    if not lower or not upper:
        raise RuntimeError(
            f"target {target} lacks bracketing same-LSP mass points"
        )
    return [lower[-1][1], target, upper[0][1]]


def component_median(rows: list[list[float]]) -> list[float]:
    if len(rows) != 3:
        raise ValueError("exactly three rows are required")
    return [sorted(values)[1] for values in zip(*rows)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--template-summary", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument(
        "--topology", choices=("T2tt", "T2bW", "T2tb"), required=True
    )
    parser.add_argument("--mass-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import ROOT

    ROOT.gROOT.SetBatch(True)
    summary = json.loads(args.template_summary.read_text())
    signals = summary.get("signals") or {}
    if args.mass_key not in signals:
        raise SystemExit(f"mass point absent from summary: {args.mass_key}")
    if signals[args.mass_key].get("topology") != args.topology:
        raise SystemExit(
            f"topology mismatch for {args.mass_key}: "
            f"{signals[args.mass_key].get('topology')} != {args.topology}"
        )

    triplet = nearest_triplet(signals, args.mass_key)
    source = ROOT.TFile.Open(str(args.template_root), "READ")
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open {args.template_root}")

    shapes: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    fractions: dict[str, list[float]] = {}
    totals: dict[str, float] = {}
    for key in triplet:
        values, errs = read_shape(source, key)
        total = sum(values)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"invalid High-dM total for {key}: {total}")
        shapes[key] = values
        errors[key] = errs
        totals[key] = total
        fractions[key] = [value / total for value in values]
    source.Close()

    median = component_median([fractions[key] for key in triplet])
    median_sum = sum(median)
    if not math.isfinite(median_sum) or median_sum <= 0.0:
        raise RuntimeError(f"invalid median fraction sum: {median_sum}")
    regularized_fractions = [value / median_sum for value in median]
    target_total = totals[args.mass_key]
    regularized_values = [
        target_total * fraction for fraction in regularized_fractions
    ]

    # Preserve the target point's original per-bin relative MC uncertainty
    # where defined.  For a newly populated bin, use the largest finite
    # relative uncertainty from the two neighboring templates.
    regularized_errors: list[float] = []
    for index, value in enumerate(regularized_values):
        old_value = shapes[args.mass_key][index]
        old_error = errors[args.mass_key][index]
        if old_value > 0.0 and math.isfinite(old_error):
            relative = old_error / old_value
        else:
            candidates = [
                errors[key][index] / shapes[key][index]
                for key in triplet
                if shapes[key][index] > 0.0
                and math.isfinite(errors[key][index])
            ]
            relative = max(candidates, default=1.0)
        regularized_errors.append(value * relative)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_dir / "templates.root"
    output_card = args.output_dir / f"datacard_{args.mass_key}.txt"
    shutil.copy2(args.template_root, output_root)

    output = ROOT.TFile.Open(str(output_root), "UPDATE")
    if not output or output.IsZombie():
        raise RuntimeError(f"cannot update {output_root}")
    process = signal_process(args.mass_key)
    for index, channel in enumerate(HIGHDM_CHANNELS):
        directory = output.GetDirectory(channel)
        if not directory:
            raise RuntimeError(f"missing output directory {channel}")
        hist = directory.Get(process)
        if not hist:
            raise RuntimeError(f"missing output histogram {channel}/{process}")
        hist.SetBinContent(1, regularized_values[index])
        hist.SetBinError(1, regularized_errors[index])
        directory.cd()
        hist.Write(process, ROOT.TObject.kOverwrite)
    output.Write()
    output.Close()

    card_text = args.source_card.read_text()
    if str(args.template_root) not in card_text:
        raise RuntimeError("source template path absent from datacard")
    output_card.write_text(
        card_text.replace(str(args.template_root), str(output_root), 1)
    )

    before = shapes[args.mass_key]
    manifest = {
        "status": "complete",
        "schema_version": "highdm_signal_shape_regularization_diagnostic_v1",
        "topology": args.topology,
        "mass_key": args.mass_key,
        "triplet": triplet,
        "method": (
            "component-wise median of the nearest lower, target, and nearest "
            "higher same-LSP High-dM 60-bin fractions; renormalized to the "
            "unchanged target High-dM total"
        ),
        "source_template_root": str(args.template_root),
        "source_template_summary": str(args.template_summary),
        "source_card": str(args.source_card),
        "output_template_root": str(output_root),
        "output_card": str(output_card),
        "highdm_total_before": sum(before),
        "highdm_total_after": sum(regularized_values),
        "highdm_total_difference": sum(regularized_values) - sum(before),
        "l1_fraction_change": sum(
            abs(new - old)
            for new, old in zip(
                regularized_fractions, fractions[args.mass_key]
            )
        ),
        "maximum_absolute_fraction_change": max(
            abs(new - old)
            for new, old in zip(
                regularized_fractions, fractions[args.mass_key]
            )
        ),
        "channels": {
            channel: {
                "before": before[index],
                "after": regularized_values[index],
                "before_fraction": fractions[args.mass_key][index],
                "after_fraction": regularized_fractions[index],
            }
            for index, channel in enumerate(HIGHDM_CHANNELS)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

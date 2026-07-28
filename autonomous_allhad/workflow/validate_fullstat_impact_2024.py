#!/usr/bin/env python3
"""Validate a complete 2024 High-dM + Low-dM Combine impact result."""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from math import ceil
from pathlib import Path

import uproot
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def pdf_pages(path):
    text = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in text.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1])
    raise RuntimeError("pdfinfo did not report a page count")


def png_dimensions(path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("invalid PNG signature")
    return list(struct.unpack(">II", header[16:24]))


def png_nonwhite_fraction(path):
    with Image.open(path) as image:
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        nonwhite = sum(histogram[:250])
        return nonwhite / float(image.width * image.height)


def card_nuisance_contract(card_text):
    ln_n = set()
    rate_parameters = set()
    auto_mc_stats = None
    for raw_line in card_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "lnN":
            ln_n.add(fields[0])
        elif len(fields) >= 2 and fields[1] == "rateParam":
            rate_parameters.add(fields[0])
        elif len(fields) >= 3 and fields[1] == "autoMCStats":
            auto_mc_stats = int(float(fields[2]))
    return {
        "lnN": sorted(ln_n),
        "rate_parameters": sorted(rate_parameters),
        "autoMCStats": auto_mc_stats,
    }


def card_has_no_2025_content(card_text):
    """Check semantic card identifiers, not accidental numeric substrings."""
    for raw_line in card_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if re.search(r"\b2025\b", line):
                return False
            continue
        fields = line.split()
        if any("2025" in field for field in fields[:2]):
            return False
    return True


def analysis_contract(manifest, card_text):
    schema = manifest.get("schema_version") or manifest.get("schema") or ""
    lowdm_match = re.search(r"lowdm(\d+)", schema)
    if lowdm_match:
        lowdm_bins = int(lowdm_match.group(1))
    else:
        lowdm_bins = int(manifest["lowdm"]["signal_bins"])
    highdm_match = re.search(r"highdm(\d+)", schema)
    highdm_bins = int(highdm_match.group(1)) if highdm_match else 60
    root_channels = (manifest.get("root_summary") or {}).get("channels")
    if isinstance(root_channels, dict):
        channel_count = len(root_channels)
        total_bins = channel_count
        model = (
            "an_zinv"
            if "an_zinv" in schema
            else "nb_recoil_transfer_factor"
        )
    else:
        channel_count = int(manifest["channel_count"])
        total_bins = int(manifest["total_analysis_bins"])
        model = "legacy_multibin"
    card_contract = card_nuisance_contract(card_text)
    return {
        "schema": schema,
        "model": model,
        "highdm_bins": highdm_bins,
        "lowdm_bins": lowdm_bins,
        "channel_count": channel_count,
        "total_bins": total_bins,
        "card_contract": card_contract,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stem")
    args = parser.parse_args()

    work = args.work
    impact_json = work / "impacts_mStop1200_mLSP500.json"
    label_map_path = work / "impact_labels_2024.json"

    impacts = json.loads(impact_json.read_text())
    label_map = json.loads(label_map_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    card_text = args.card.read_text()
    contract = analysis_contract(manifest, card_text)
    highdm_bins = contract["highdm_bins"]
    lowdm_bins = contract["lowdm_bins"]
    total_bins = contract["total_bins"]
    channel_count = contract["channel_count"]
    card_contract = contract["card_contract"]
    plot_stem = args.stem or (
        f"impacts_2024_highdm{highdm_bins}_lowdm{lowdm_bins}_"
        + (
            "an_zinv_"
            if contract["model"] == "an_zinv"
            else (
                "nb_recoil_tf_"
                if contract["model"] == "nb_recoil_transfer_factor"
                else ""
            )
        )
        + "mStop1200_mLSP500_full_mcstat"
    )
    plot_png = work / f"{plot_stem}.png"
    plot_pdf = work / f"{plot_stem}.pdf"
    summary_pdf = work / f"{plot_stem}_summary.pdf"
    summary_png = work / f"{plot_stem}_summary.png"
    params = impacts["params"]
    names = [param["name"] for param in params]
    stat_names = [name for name in names if name.startswith("prop_bin")]
    nonstat_names = [name for name in names if not name.startswith("prop_bin")]

    missing_roots = []
    invalid_roots = []
    root_entry_counts = {}
    for name in names:
        path = work / f"higgsCombine_paramFit_Test_{name}.MultiDimFit.mH1200.root"
        if not path.is_file() or path.stat().st_size == 0:
            missing_roots.append(name)
            continue
        try:
            with uproot.open(path) as root_file:
                tree = root_file["limit"]
                count = int(tree.num_entries)
                root_entry_counts[name] = count
                if count < 3:
                    invalid_roots.append(
                        {"name": name, "reason": f"limit tree has {count} entries"}
                    )
                    continue
                for branch in ("deltaNLL", "r"):
                    values = tree[branch].array(library="np")
                    if not all(math.isfinite(float(value)) for value in values):
                        invalid_roots.append(
                            {"name": name, "reason": f"non-finite {branch}"}
                        )
                        break
        except Exception as error:
            invalid_roots.append({"name": name, "reason": repr(error)})

    required_nonstat = set(card_contract["lnN"]) | set(
        card_contract["rate_parameters"]
    )
    poi = impacts["POIs"][0]
    mass_points = manifest.get("mass_points", [])
    manifest_limits = manifest.get("limits") or {}
    manifest_boundary_partial = (
        manifest_limits.get("status") == "partial"
        and manifest_limits.get("collected_point_count")
        == manifest_limits.get("requested_point_count") - 1
        and manifest_limits.get("missing_points")
        == ["mStop1800_mLSP1600"]
    )
    manifest_complete = manifest.get("status") in {
        "combine_outputs_complete",
        "complete",
    } or manifest_boundary_partial
    card_has_expected_channels = (
        "hSR_b59" in card_text and f"lSR_b{lowdm_bins - 1:02d}" in card_text
        if contract["model"] in {"nb_recoil_transfer_factor", "an_zinv"}
        else (
            "cat7_SR_selected_recoil60_nb2_nt2plus_w0" in card_text
            and "cat7_SR_lowDeltaM" in card_text
        )
    )
    checks = {
        "impact_json_finite": finite_tree(impacts),
        "parameter_count_positive": len(params) > 0,
        "parameter_names_unique": len(names) == len(set(names)),
        "label_map_covers_all_parameters": set(label_map) == set(names),
        "label_map_values_unique": len(label_map.values())
        == len(set(label_map.values())),
        "statistical_parameters_present": len(stat_names) > 0,
        "nonstatistical_parameter_count_matches_card": len(nonstat_names)
        == len(required_nonstat),
        "all_expected_nonstatistical_parameters": set(nonstat_names)
        == required_nonstat,
        "all_parameter_fit_roots_present": not missing_roots,
        "all_parameter_fit_roots_valid": not invalid_roots,
        "poi_is_r": poi.get("name") == "r",
        "poi_fit_is_finite_triplet": len(poi.get("fit", [])) == 3
        and finite_tree(poi["fit"]),
        "manifest_status_complete": manifest_complete,
        "manifest_schema_2024_only": "2024" in contract["schema"],
        "manifest_channel_count_positive": channel_count > 0,
        "manifest_total_bins_consistent": total_bins == channel_count
        if contract["model"] in {"nb_recoil_transfer_factor", "an_zinv"}
        else total_bins == 5 * 6 + highdm_bins + 6 * lowdm_bins,
        "manifest_highdm_60_bins": highdm_bins == 60,
        "manifest_lowdm_bins_positive": lowdm_bins > 0,
        "manifest_mass_point_present": "mStop1200_mLSP500"
        in mass_points,
        "manifest_auto_mc_stats_10": card_contract["autoMCStats"] == 10,
        "card_auto_mc_stats_10": "* autoMCStats 10" in card_text,
        "card_has_expected_highdm_lowdm_channels": card_has_expected_channels,
        "card_has_no_2025_content": card_has_no_2025_content(card_text),
        "main_impact_pdf_page_count": pdf_pages(plot_pdf)
        == ceil(len(params) / 30),
        "summary_pdf_one_page": pdf_pages(summary_pdf) == 1,
        "impact_png_valid": all(value > 0 for value in png_dimensions(plot_png))
        and png_nonwhite_fraction(plot_png) > 0.01,
        "summary_png_valid": all(
            value > 0 for value in png_dimensions(summary_png)
        )
        and png_nonwhite_fraction(summary_png) > 0.01,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    ordered = sorted(params, key=lambda item: abs(item["impact_r"]), reverse=True)
    report = {
        "schema": (
            f"2024_highdm60_lowdm{lowdm_bins}_"
            "full_mcstat_impact_validation_v1"
        ),
        "status": "complete" if not failed_checks else "failed",
        "analysis": {
            "year": 2024,
            "model": contract["model"],
            "model_details": manifest.get("model") or {},
            "schema": contract["schema"],
            "highdm_signal_bins": highdm_bins,
            "lowdm_signal_bins": lowdm_bins,
            "channels": channel_count,
            "total_analysis_bins": total_bins,
            "data_mode": "asimov",
            "benchmark": {"mStop_GeV": 1200, "mLSP_GeV": 500},
            "expected_signal": 1.0,
            "autoMCStats_threshold": card_contract["autoMCStats"],
            "systematics": {
                "lnN": card_contract["lnN"],
                "rate_parameters": card_contract["rate_parameters"],
                "autoMCStats": card_contract["autoMCStats"],
            },
        },
        "impact_scan": {
            "parameter_count": len(params),
            "statistical_parameter_count": len(stat_names),
            "nonstatistical_parameter_count": len(nonstat_names),
            "missing_fit_count": len(missing_roots),
            "invalid_fit_count": len(invalid_roots),
            "fit_tree_entries_min": min(root_entry_counts.values()),
            "fit_tree_entries_max": max(root_entry_counts.values()),
            "poi": poi,
            "top_impacts": [
                {"name": item["name"], "impact_r": item["impact_r"]}
                for item in ordered[:20]
            ],
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                args.card,
                args.manifest,
                impact_json,
                label_map_path,
                plot_png,
                plot_pdf,
                summary_pdf,
                summary_png,
            )
        },
        "rendering": {
            "main_pdf_pages": pdf_pages(plot_pdf),
            "summary_pdf_pages": pdf_pages(summary_pdf),
            "png_dimensions": png_dimensions(plot_png),
            "png_nonwhite_fraction": png_nonwhite_fraction(plot_png),
            "summary_png_dimensions": png_dimensions(summary_png),
            "summary_png_nonwhite_fraction": png_nonwhite_fraction(summary_png),
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_fit_parameters": missing_roots,
        "invalid_fit_parameters": invalid_roots,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()

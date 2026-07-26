#!/usr/bin/env python3
"""Validate a complete 2024 High-dM + Low-dM Combine impact result."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    work = args.work
    impact_json = work / "impacts_mStop1200_mLSP500.json"
    plot_stem = "impacts_2024_highdm60_lowdm42_mStop1200_mLSP500_full_mcstat"
    plot_png = work / f"{plot_stem}.png"
    plot_pdf = work / f"{plot_stem}.pdf"
    summary_pdf = work / f"{plot_stem}_summary.pdf"
    summary_png = work / f"{plot_stem}_summary.png"
    label_map_path = work / "impact_labels_2024.json"

    impacts = json.loads(impact_json.read_text())
    label_map = json.loads(label_map_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    card_text = args.card.read_text()

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

    required_weight_nuisances = set(manifest["systematics"]["weight_shapes"])
    required_nonstat = required_weight_nuisances | {
        manifest["systematics"]["lumi"]["name"]
    }
    poi = impacts["POIs"][0]
    checks = {
        "impact_json_finite": finite_tree(impacts),
        "parameter_count_369": len(params) == 369,
        "parameter_names_unique": len(names) == len(set(names)),
        "label_map_covers_all_parameters": set(label_map) == set(names),
        "label_map_values_unique": len(label_map.values())
        == len(set(label_map.values())),
        "statistical_parameter_count_358": len(stat_names) == 358,
        "nonstatistical_parameter_count_11": len(nonstat_names) == 11,
        "all_expected_nonstatistical_parameters": set(nonstat_names)
        == required_nonstat,
        "all_parameter_fit_roots_present": not missing_roots,
        "all_parameter_fit_roots_valid": not invalid_roots,
        "poi_is_r": poi.get("name") == "r",
        "poi_fit_is_finite_triplet": len(poi.get("fit", [])) == 3
        and finite_tree(poi["fit"]),
        "manifest_status_complete": manifest.get("status")
        == "combine_outputs_complete",
        "manifest_schema_2024_only": manifest.get("schema")
        == "2024_highdm60_lowdm42_v1",
        "manifest_asimov": manifest.get("data_mode") == "asimov",
        "manifest_12_channels": manifest.get("channel_count") == 12,
        "manifest_342_total_bins": manifest.get("total_analysis_bins") == 342,
        "manifest_highdm_60_bins": manifest["highdm"].get("signal_bins") == 60,
        "manifest_lowdm_42_bins": manifest["lowdm"].get("signal_bins") == 42,
        "manifest_mass_point_present": "mStop1200_mLSP500"
        in manifest.get("mass_points", []),
        "manifest_auto_mc_stats_10": manifest["systematics"].get("autoMCStats")
        == 10,
        "card_auto_mc_stats_10": "* autoMCStats 10" in card_text,
        "card_has_highdm60_channel": "cat7_SR_selected_recoil60_nb2_nt2plus_w0"
        in card_text,
        "card_has_lowdm42_channel": "cat7_SR_lowDeltaM" in card_text,
        "card_has_no_2025_content": "2025" not in card_text,
        "main_impact_pdf_15_pages": pdf_pages(plot_pdf) == 15,
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
        "schema": "2024_highdm60_lowdm42_full_mcstat_impact_validation_v1",
        "status": "complete" if not failed_checks else "failed",
        "analysis": {
            "year": 2024,
            "highdm_signal_bins": 60,
            "lowdm_signal_bins": 42,
            "channels": manifest["channel_count"],
            "total_analysis_bins": manifest["total_analysis_bins"],
            "data_mode": manifest["data_mode"],
            "benchmark": {"mStop_GeV": 1200, "mLSP_GeV": 500},
            "expected_signal": 1.0,
            "autoMCStats_threshold": manifest["systematics"]["autoMCStats"],
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

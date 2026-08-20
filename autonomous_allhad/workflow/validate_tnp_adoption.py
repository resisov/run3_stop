#!/usr/bin/env python3
"""Apply auditable adoption gates to a low-pT TnP fit result."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FIT_VARIATIONS = (
    "nominal",
    "signal_template_combined",
    "background_linear",
    "mass_window_narrow",
    "mass_window_medium",
    "alternate_binning",
)


def _trigger_audit_blockers(config: dict[str, Any], audit: dict[str, Any], label: str) -> list[str]:
    blockers = []
    if not audit.get("files_processed") or audit.get("file_failures"):
        blockers.append(f"{label}: trigger audit has no clean processed file")
    present_by_file = audit.get("paths_present_by_file") or {}
    for path in config["reference_paths"]:
        if not any(path in paths for paths in present_by_file.values()):
            blockers.append(f"{label}: {path} is absent from audited files")
        apply_to_sample = label == "data" or bool(config.get("apply_reference_trigger_to_mc", True))
        fired = int((audit.get("event_counts") or {}).get(path, 0))
        if apply_to_sample and fired <= 0:
            blockers.append(f"{label}: {path} has no fired event in the audit")

    if not bool(config.get("tag_trigger_match_required", True)):
        return blockers

    bit_mask = int(config["tag_trigger_object_filter_bits"])
    if bit_mask <= 0 or bit_mask & (bit_mask - 1):
        blockers.append(f"{label}: trigger-object filter mask is not one bit: {bit_mask}")
        return blockers
    bit_index = int(math.log2(bit_mask))
    for path in config["reference_paths"]:
        matched = int((audit.get("matched_trigger_objects") or {}).get(path, 0))
        bit_matches = int(
            ((audit.get("individual_bit_index_counts_by_path") or {}).get(path) or {}).get(str(bit_index), 0)
        )
        if matched <= 0 or bit_matches <= 0:
            blockers.append(
                f"{label}: {path} has matched={matched}, filter-bit-{bit_index} matches={bit_matches}"
            )
    return blockers


def validate(
    *,
    result: dict[str, Any],
    histograms: dict[str, Any],
    config: dict[str, Any],
    data_trigger_audit: dict[str, Any],
    mc_trigger_audit: dict[str, Any],
    max_chi2_ndf: float,
    adopt_after_visual_review: bool,
    visual_review_note: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    expected_probe_definition = config.get("probe_definition")
    if histograms.get("probe_definition") != expected_probe_definition:
        blockers.append(
            "histogram probe definition does not match config: "
            f"{histograms.get('probe_definition')!r} != {expected_probe_definition!r}"
        )
    if result.get("probe_definition") != expected_probe_definition:
        blockers.append(
            "fit-result probe definition does not match config: "
            f"{result.get('probe_definition')!r} != {expected_probe_definition!r}"
        )
    expected_tag_pt = config.get("tag_pt_min_gev")
    for label, payload in (("histogram", histograms), ("result", result)):
        if payload.get("tag_pt_min_gev") != expected_tag_pt:
            blockers.append(
                f"{label} tag_pt_min_gev mismatch: "
                f"{payload.get('tag_pt_min_gev')!r} != {expected_tag_pt!r}"
            )
    if histograms.get("adoption_blockers"):
        blockers.extend(f"histograms: {item}" for item in histograms["adoption_blockers"])
    if int(histograms.get("files_processed", 0)) != int(histograms.get("files_expected", -1)):
        blockers.append(
            f"ROOT coverage is {histograms.get('files_processed')}/{histograms.get('files_expected')}"
        )
    if histograms.get("files_failed"):
        blockers.append(f"{len(histograms['files_failed'])} unresolved ROOT failures")
    for edge_key in ("probe_abseta_edges", "probe_pt_edges_gev"):
        if result.get(edge_key) != histograms.get(edge_key):
            blockers.append(
                f"fit-result {edge_key} does not match reduced histograms: "
                f"{result.get(edge_key)!r} != {histograms.get(edge_key)!r}"
            )
    expected_bins = (
        (len(histograms.get("probe_abseta_edges") or []) - 1)
        * (len(histograms.get("probe_pt_edges_gev") or []) - 1)
    )
    if len(result.get("bins") or []) != expected_bins:
        blockers.append(f"fit result has {len(result.get('bins') or [])}/{expected_bins} bins")
    fit_diagnostics = []
    for item in result.get("bins") or []:
        flat_index = int(item.get("flat_index", -1))
        if not item.get("valid"):
            blockers.append(f"bin {flat_index}: nominal/systematic fit result is invalid")
            continue
        sf = float(item["scale_factor"])
        uncertainty = float(item["scale_factor_uncertainty"])
        if not math.isfinite(sf) or not 0.2 <= sf <= 5.0:
            blockers.append(f"bin {flat_index}: scale factor {sf} is outside [0.2, 5.0]")
        if not math.isfinite(uncertainty) or uncertainty < 0.0 or uncertainty > 2.0 * sf:
            blockers.append(f"bin {flat_index}: uncertainty {uncertainty} is invalid or exceeds 200%")
        if "scale_factor_pileup_uncertainty" not in item:
            blockers.append(f"bin {flat_index}: pileup uncertainty is absent")
        for variation in FIT_VARIATIONS:
            for sample in ("data", "mc"):
                fitted = ((item.get("fits") or {}).get(variation) or {}).get(sample) or {}
                chi2_ndf = fitted.get("chi2_ndf")
                valid = bool(fitted.get("valid"))
                fit_diagnostics.append({
                    "flat_index": flat_index,
                    "variation": variation,
                    "sample": sample,
                    "valid": valid,
                    "chi2_ndf": chi2_ndf,
                })
                if not valid:
                    blockers.append(f"bin {flat_index}: {variation}/{sample} fit invalid")
                elif chi2_ndf is None or not math.isfinite(float(chi2_ndf)) or float(chi2_ndf) > max_chi2_ndf:
                    blockers.append(
                        f"bin {flat_index}: {variation}/{sample} chi2/ndf={chi2_ndf} exceeds {max_chi2_ndf}"
                    )
    blockers.extend(_trigger_audit_blockers(config, data_trigger_audit, "data"))
    blockers.extend(_trigger_audit_blockers(config, mc_trigger_audit, "mc"))
    blockers = list(dict.fromkeys(blockers))
    output = dict(result)
    output["validation"] = {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "files_processed": histograms.get("files_processed"),
        "files_expected": histograms.get("files_expected"),
        "valid_bins": sum(bool(item.get("valid")) for item in result.get("bins") or []),
        "expected_bins": expected_bins,
        "max_chi2_ndf": max_chi2_ndf,
        "fit_diagnostics": fit_diagnostics,
        "trigger_filter_mask": config.get("tag_trigger_object_filter_bits"),
        "data_trigger_audit": data_trigger_audit.get("created_unix"),
        "mc_trigger_audit": mc_trigger_audit.get("created_unix"),
        "pileup_uncertainty_source": result.get("pileup_uncertainty_source"),
        "visual_review_note": visual_review_note,
    }
    if blockers:
        output["status"] = "validation_blocked"
    elif adopt_after_visual_review:
        if not visual_review_note:
            raise ValueError("--adopt-after-visual-review requires --visual-review-note")
        output["status"] = "adopted"
    else:
        output["status"] = "adoption_ready"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("histograms", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-trigger-audit", type=Path, required=True)
    parser.add_argument("--mc-trigger-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chi2-ndf", type=float, default=12.0)
    parser.add_argument("--adopt-after-visual-review", action="store_true")
    parser.add_argument("--visual-review-note")
    args = parser.parse_args()
    output = validate(
        result=json.loads(args.result.read_text()),
        histograms=json.loads(args.histograms.read_text()),
        config=json.loads(args.config.read_text()),
        data_trigger_audit=json.loads(args.data_trigger_audit.read_text()),
        mc_trigger_audit=json.loads(args.mc_trigger_audit.read_text()),
        max_chi2_ndf=args.max_chi2_ndf,
        adopt_after_visual_review=args.adopt_after_visual_review,
        visual_review_note=args.visual_review_note,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": output["status"],
        "blockers": output["validation"]["blockers"],
        "valid_bins": output["validation"]["valid_bins"],
        "expected_bins": output["validation"]["expected_bins"],
    }, indent=2, sort_keys=True))
    return 0 if output["status"] != "validation_blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())

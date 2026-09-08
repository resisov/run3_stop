#!/usr/bin/env python3
"""Audit and summarize the two-year histogram-only background products."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from gnn_background_histograms import project_double_ratio_uncertainty, sha256


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def preserve_300(output_root):
    """Freeze exact rollback products before adopting the approved domain."""
    receipt_path = output_root / "low250_adoption_baseline.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for path, expected in receipt["protected_files"].items():
            if sha256(output_root / path) != expected:
                raise ValueError(f"protected product changed: {path}")
        return receipt
    previous = read(output_root / "campaign_state.json")
    receipt = {
        "status": "preserved", "approval": {
            "source_thread": "019f8407-f810-7f13-b2e9-4af7f0d99f07",
            "user_text": "승인한다", "approved_at_utc": "2026-09-07T22:29:50Z",
            "scope": "Adopt Low-dM double ratio from 250 GeV in both years; no card, limit, event-selection or nominal-factor changes"},
        "protected_files": {}, "preserved_300": {}, "validated_250_reference": {},
        "source_products_commit": "b754e59aefdaccde9f60297f93450b75eee5b933",
    }
    for year, info in previous["years"].items():
        base = output_root / year
        if read(base / "zgamma/zgamma_double_ratio.json")["lowdm"]["edges"][0] != 300:
            raise ValueError("300-start must be preserved before canonical replacement")
        for relative, expected in info["artifact_hashes"].items():
            path = base / relative
            if sha256(path) != expected:
                raise ValueError(f"baseline checksum mismatch: {path}")
            if not relative.startswith("zgamma/") and relative != "double_ratio_domain_comparison.json":
                receipt["protected_files"][str(path.relative_to(output_root))] = expected
        old = base / "zgamma_300_previous"
        if old.exists():
            raise ValueError(f"unregistered rollback directory exists: {old}")
        shutil.copytree(base / "zgamma", old)
        shutil.copy2(base / "double_ratio_domain_comparison.json", old / "domain_comparison_before_adoption.json")
        receipt["preserved_300"][year] = {str(p.relative_to(output_root)): sha256(p) for p in sorted(old.iterdir())}
        proposal = base / "zgamma_250_proposal/zgamma_double_ratio.json"
        receipt["validated_250_reference"][year] = {"path": str(proposal.relative_to(output_root)), "sha256": sha256(proposal)}
    prior = output_root / "before_low250"
    prior.mkdir(exist_ok=False)
    for name in ("campaign_state.json", "sync_receipt.json", "README.md", "RUN.md"):
        shutil.copy2(output_root / name, prior / name)
    receipt["previous_global_records"] = {str(p.relative_to(output_root)): sha256(p) for p in prior.iterdir()}
    write(receipt_path, receipt)
    return receipt


def comparison(compact, old, new, mapping):
    a, b = old["lowdm"]["bins"], new["lowdm"]["bins"]
    common = []
    for previous, proposed in zip(a, b[1:]):
        assert previous["low"] == proposed["low"] and previous["high"] == proposed["high"]
        common.append({
            "low": previous["low"], "high": previous["high"],
            **{field: {"existing_300": previous[field], "proposal_250": proposed[field], "difference": proposed[field] - previous[field]}
               for field in ("double_ratio", "double_ratio_stat", "systematic", "downstream_central_abs_deviation")},
        })
    first_bin = {}
    for region, target in (("GCR", "GJ"), ("DY2E", "DY"), ("DY2M", "DY")):
        by_group = {}
        for group in ("Nb1", "Nb2plus"):
            records = compact["lowdm"]["recoil"][region][group]
            datum = records["data_obs"]["nominal"]
            mc = records[target]["nominal"]
            other = sum(r["nominal"]["sumw"][0] for sample, r in records.items() if sample not in {"data_obs", target})
            by_group[group] = {"data": datum["sumw"][0], "target_mc": mc["sumw"][0], "target_mc_sumw2": mc["sumw2"][0], "other_mc": other, "data_minus_other": datum["sumw"][0] - other}
        first_bin[region] = by_group
    impacts = {}
    for category, node in mapping["zinv_projection"].items():
        components = np.asarray(node["shape_components"])
        shape = np.asarray(node["Sgamma"])
        weighted = components * shape * node["RZ"]
        predicted = np.asarray(node["rz_sgamma"])
        fraction = weighted / predicted[:, None]
        old_delta = np.asarray([r["downstream_central_abs_deviation"] for r in a])
        new_delta = np.asarray([r["downstream_central_abs_deviation"] for r in b])
        old_response = fraction[:, 1:] * old_delta
        new_response = fraction[:, 1:] * new_delta[1:]
        impacts[category] = {
            "predicted_Z_fraction_from_250_300": fraction[:, 0].tolist(),
            "existing_common_bin_fractional_response": old_response.tolist(),
            "proposed_common_bin_fractional_response": new_response.tolist(),
            "proposed_250_300_fractional_response": (fraction[:, 0] * new_delta[0]).tolist(),
            "max_common_bin_response_change": float(np.max(np.abs(new_response - old_response))),
        }
    return {
        "status": "comparison_complete_adopted",
        "threshold_evidence": "Frozen region core has recoil>250; populated 250-300 joint projections match compact CR histograms. No event input read.",
        "first_bin_support": first_bin,
        "normalizations": {field: {"existing_300": a[0][field], "proposal_250": b[0][field], "relative_change": b[0][field] / a[0][field] - 1}
                           for field in ("z_normalization", "photon_normalization")},
        "common_bins": common,
        "new_250_300_bin": b[0],
        "gnn_impacts": impacts,
        "max_common_D_absolute_shift": max(abs(row["double_ratio"]["difference"]) for row in common),
        "max_common_GNN_relative_response_shift": max(row["max_common_bin_response_change"] for row in impacts.values()),
        "existing_domain_gap": "No assigned double-ratio central deviation for 250-300; unavailable, not zero uncertainty.",
        "response_definition": "Per-UT-bin +delta linear template response divided by RZ*Sgamma nominal; no nuisance correlations or card changes assumed.",
        "proposal_applied": True,
    }


def input_policy(manifest, historical=False):
    """Derive selection validity from the authoritative manifest, never a date."""
    is_ten = (manifest.get("electron_veto_pt_min_gev") == 10
              and manifest.get("muon_veto_pt_min_gev") == 10
              and manifest.get("lepton_threshold_operator") == ">")
    if not historical:
        if manifest.get("status") != "canonical" or not is_ten:
            raise ValueError("fresh campaign requires canonical strict 10-GeV inputs")
        if manifest.get("lowdm_double_ratio_min_gev") != 250:
            raise ValueError("fresh campaign requires the approved 250-GeV domain")
        if set(manifest.get("excluded_sf_components", [])) != {"veto_electron_5to10", "loose_muon_5to10"}:
            raise ValueError("retired low-pT SF policy absent from manifest")
    return {"status": "validated_10gev_inputs" if is_ten else "blocked",
            "valid_only_for_recorded_input_hashes": True,
            "evidence_source": "canonical manifest plus input checksums and producer validation",
            "nominal_or_SF_reprocessed_by_estimator": False}


def recorded_plots(base, product, product_path, minimum):
    paths = product.get("plots", [])
    if not paths:
        receipt = read(base / "plot_manifest.json")
        if receipt["source_sha256"] != sha256(product_path):
            raise ValueError("plot receipt belongs to different measured factors")
        paths = receipt["plots"]
    if len(paths) < minimum or any(not Path(p).is_file() for p in paths):
        raise ValueError("missing factor plots")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-low250", action="store_true",
                        help="Historical recovery only: preserve verified 300-start products.")
    args = parser.parse_args()
    if args.prepare_low250:
        preserve_300(args.output_root)
        print(json.dumps({"status": "300_start_preserved"}))
        return
    baseline_path = args.output_root / "low250_adoption_baseline.json"
    baseline = read(baseline_path) if baseline_path.exists() else None
    manifest = read(args.manifest)
    policy = input_policy(manifest, historical=baseline is not None)
    approval = baseline["approval"] if baseline else {
        "source": str(args.manifest), "domain_min_gev": manifest["lowdm_double_ratio_min_gev"],
        "status": "previously_approved_domain_remeasured_with_new_canonical_inputs"}
    result = {"status": "complete", "input_manifest": str(args.manifest),
              "input_manifest_sha256": sha256(args.manifest), "intermediate_root_reread": False,
              "SR_blinded": True, "years": {}, "downstream_10gev_veto_use": policy,
              "lowdm_double_ratio_adoption": {"min_ut_gev": 250, "status": "adopted", "approval": approval},
              "cards_limits_web_untouched": True}
    code_paths = ["autonomous_allhad/workflow/" + name for name in (
        "build_histogram_tf_inputs_2024.py", "gnn_background_histograms.py",
        "plot_recoil_transfer_factors_2024.py", "build_sgamma_ut_report_2024.py",
        "build_an_zinv_factors_2024.py", "build_zgamma_double_ratio_2024.py",
        "audit_background_histogram_products.py")]
    code_paths += ["autonomous_allhad/autonomous_allhad/dy_estimation/" + name
                   for name in ("build_measurement.py", "model.py", "report.py", "__main__.py")]
    result["code_sha256"] = {path: sha256(Path(path)) for path in code_paths}
    lines = ["# Histogram-only background estimation — 2024 and 2025", "",
             f"Canonical input manifest: {args.manifest}. Selection validity: {policy['status']}.",
             "No intermediate ROOT or NanoAOD input was opened. Nominal histograms were not modified or reweighted.", "",
             "High-dM: retained 73 bins. Low-dM: frozen GNN30. Nb1/Nb2plus are factor groups, not new SR categories.", "",
             "## RZ results", "", "| Year | Region | Nb = 1 | Nb ≥ 2 |", "|---|---|---|---|"]
    for year, info in manifest["years"].items():
        base = args.output_root / year
        inputs = {}
        for name in ("main", "background_estimation", "gnn"):
            source = Path(info[name]["path"])
            actual = sha256(source)
            if actual != info[name]["sha256"]:
                raise ValueError(f"input checksum mismatch: {source}")
            if Path(info[name]["canonical_alias"]).resolve() != source.resolve():
                raise ValueError(f"canonical alias changed: {name}/{year}")
            inputs[name] = {"path": str(source), "sha256": actual}
        compact = read(Path(inputs["background_estimation"]["path"]))
        rz = read(base / "dy_measurement.json")
        sg = read(base / "sgamma/sgamma_ut.json")
        tf = read(base / "tf_inputs.json")
        plots = read(base / "tf" / f"transfer_factors_{year}_nb_recoil.json")
        new = read(base / "zgamma/zgamma_double_ratio.json")
        if new["adoption_status"] != "adopted" or new["lowdm"]["edges"] != [250, 300, 350, 400, 500, 1500]:
            raise ValueError("double-ratio domain differs from the adopted definition")
        mapping = tf["lowdm_gnn"]
        provenance = mapping["provenance"]
        for key, expected in (
            ("gnn_input", inputs["gnn"]["sha256"]),
            ("gnn_config", manifest["lowdm"]["configuration_sha256"]),
            ("sgamma_input", sha256(base / "sgamma/sgamma_ut.json")),
            ("dy_measurement", sha256(base / "dy_measurement.json"))):
            if provenance[key]["sha256"] != expected:
                raise ValueError(f"GNN mapping source checksum mismatch: {key}")
        for name, product in (("RZ", rz), ("Sgamma", sg), ("TF", tf), ("double_ratio", new)):
            if product["status"] != "complete" or product["provenance"]["hist_input_sha256"] != inputs["background_estimation"]["sha256"]:
                raise ValueError(f"invalid/stale {name} for {year}")
        if policy["status"] == "validated_10gev_inputs":
            forbidden = ("veto_electron_5to10", "loose_muon_5to10")
            if any(any(term in v for term in forbidden) for v in mapping["histograms"]):
                raise ValueError("retired low-pT SF variations present in GNN product")
            validation_path = Path(info["validation_report"])
            if sha256(validation_path) != info["validation_report_sha256"] or read(validation_path)["status"] != "complete":
                raise ValueError("producer validation is missing or changed")
        for regime, key in (("highdm", "rz_high"), ("lowdm", "rz_low")):
            for group, record in rz[key]["combined"].items():
                if record["status"] != "complete" or record["RZ"] <= 0:
                    raise ValueError(f"invalid RZ: {year}/{regime}/{group}")
            for channel in rz[key]["channels"].values():
                for record in channel.values():
                    if record["status"] != "complete" or np.linalg.eigvalsh(record["covariance"]).min() < -1e-10:
                        raise ValueError("invalid RZ/RT fitted covariance")
            write(base / (key + ".json"), {"status": "complete", key: rz[key], "provenance": rz["provenance"]})
            r = rz[key]["combined"]
            lines.append(f"| {year} | {regime} | {r['Nb1']['RZ']:.4f} ± {r['Nb1']['RZ_stat']:.4f} | {r['Nb2plus']['RZ']:.4f} ± {r['Nb2plus']['RZ_stat']:.4f} |")
        keys = [("highdm", "rz_high", "Nb1"), ("highdm", "rz_high", "Nb2plus"),
                ("lowdm", "rz_low", "Nb1"), ("lowdm", "rz_low", "Nb2plus")]
        covariance = {"status": "complete", "order": [reg + "_" + group for reg, _, group in keys],
                      "central": [rz[key]["combined"][group]["RZ"] for _, key, group in keys],
                      "covariance": np.diag([rz[key]["combined"][group]["RZ_stat"]**2 for _, key, group in keys]).tolist(),
                      "cross_group_policy": "Existing downstream diagonal assumption retained; cross-regime covariance is not measured by these marginal counts.",
                      "channel_RZ_RT_covariance": {reg: rz[key]["channels"] for reg, key, _ in keys}}
        write(base / "rz_covariance.json", covariance)
        if baseline:
            old = read(base / "zgamma_300_previous/zgamma_double_ratio.json")
            reference = read(base / "zgamma_250_proposal/zgamma_double_ratio.json")
            if new["lowdm"] != reference["lowdm"] or new["highdm"] != old["highdm"]:
                raise ValueError("historical adoption comparison changed")
            compare = comparison(compact, old, new, mapping)
            previous = read(base / "zgamma_300_previous/domain_comparison_before_adoption.json")
            for key in ("first_bin_support", "normalizations", "common_bins", "new_250_300_bin", "gnn_impacts"):
                if compare[key] != previous[key]:
                    raise ValueError(f"historical comparison changed: {year}/{key}")
            write(base / "double_ratio_domain_comparison.json", compare)
        write(base / "lowdm_gnn_backgrounds.json", {key: value for key, value in mapping.items() if key != "histograms"})
        uncertainty = project_double_ratio_uncertainty(mapping, new)
        uncertainty["provenance"] = {
            "double_ratio": {"path": str(base / "zgamma/zgamma_double_ratio.json"), "sha256": sha256(base / "zgamma/zgamma_double_ratio.json")},
            "gnn_mapping": {"path": str(base / "lowdm_gnn_backgrounds.json"), "sha256": sha256(base / "lowdm_gnn_backgrounds.json")},
            "approval": approval, "intermediate_root_reread": False}
        write(base / "lowdm_gnn_double_ratio.json", uncertainty)
        zero_bins, signed_bins = [], []
        for route, categories in mapping["transfer_factors"]["nominal"].items():
            for category, record in categories.items():
                if record["status"] not in {"complete", "signed_mc_bins"}:
                    raise ValueError(f"invalid nominal GNN TF {year}/{route}/{category}")
                num = np.asarray(record["numerator"])
                values = np.asarray(record["transfer_factor"])
                if not np.isfinite(values).all() or not np.allclose(values * record["denominator_total"], num):
                    raise ValueError("GNN TF reconstruction failed")
                zero_bins.extend(f"{route}/{category}/{i}" for i, value in enumerate(num) if value == 0)
                signed_bins.extend(f"{route}/{category}/{i}" for i, value in enumerate(num) if value < 0)
        all_plots = recorded_plots(base / "sgamma", sg, base / "sgamma/sgamma_ut.json", 6)
        all_plots += recorded_plots(base / "zgamma", new, base / "zgamma/zgamma_double_ratio.json", 4)
        tf_plot_minimum = 14 if plots["provenance"].get("lowdm_plot_layout") == "categories_overlaid" else 54
        all_plots += recorded_plots(base / "tf", plots, base / "tf" / f"transfer_factors_{year}_nb_recoil.json", tf_plot_minimum)
        for regime in ("highdm", "lowdm"):
            report_dir = base / "dy_report" / regime
            report = read(report_dir / "summary.json")
            for names in report["plots"].values():
                if any(not (report_dir / name).is_file() for name in names):
                    raise ValueError("missing DY report figure")
        result["years"][year] = {
            "inputs": inputs, "input_count": info["input_roots"],
            **{k: info[k] for k in ("data_luminosity_coverage_complete", "data_source_bad_file_count", "source_bad_file_count", "normalization_sha256")},
            "gnn_checks": mapping["mechanical_checks"], "nominal_GNN_zero_numerator_bins": zero_bins,
            "nominal_GNN_signed_numerator_bins": signed_bins,
            "signed_bin_policy": "Retained without clipping; existing downstream signed-bin treatment must be reviewed if this list is nonempty.",
            "GNN_response_checks": {"nominal_reconstruction": True, "response_count": sum(len(r["responses"]) for r in uncertainty["categories"].values()),
                                    "max_fractional_up_response": max(abs(x) for r in uncertainty["categories"].values() for response in r["responses"] for x in response["fractional_up_response"])},
            "factor_plot_files": len(all_plots),
            "artifact_hashes": {str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob("*")) if p.is_file()}}
    if baseline:
        for relative, expected in baseline["protected_files"].items():
            if sha256(args.output_root / relative) != expected:
                raise ValueError("historically protected product changed: " + relative)
        for group in list(baseline["preserved_300"].values()) + [baseline["previous_global_records"]]:
            for relative, expected in group.items():
                if sha256(args.output_root / relative) != expected:
                    raise ValueError("historical rollback checksum mismatch")
    lines += ["", "RZ errors use the existing on/off-Z profile fit and inverse-variance ee/μμ combination. Per-channel RZ–RT covariance is exported. Combined cross-group diagonal covariance is the existing downstream assumption, not a measured absent correlation.",
              "", "## GNN transfer and shape propagation", "",
              "- Top = TT + ST. Top/W parameter sharing is unchanged; W and QCD factors remain separately exported.",
              "- GNN TF = SR score-bin MC / mapped CR parent MC integral. Parent score fractions and shared-denominator covariance are retained.",
              "- Z prediction = RZ(Nb) × sum_UT H_Zinv(GNN,UT) Sgamma(Nb,UT). Q is not transferred; no additional Z-integral rescaling is applied.",
              "- RZ/Sgamma factor categories are Nb1/Nb2plus. Final Low-dM templates stay in the frozen six categories × five GNN bins.",
              "", "## Double-ratio uncertainty", "",
              "The already-approved 250-GeV lower boundary and merged 500–1500 display tail are used. Only downstream_central_abs_deviation = abs(D−1) is projected into GNN bins, with up=1+delta and down=1/(1+delta). The plotted max(abs(D−1),stat) band is not substituted for the existing card convention. No nuisance names or correlations were changed.",
              "", "## Coverage and caveats", "",
              "| Year | Retained inputs | Source bad files | Data bad files | Complete data luminosity coverage | Zero / signed GNN TF numerator bins |",
              "|---|---|---|---|---|---|"]
    for year, info in result["years"].items():
        lines.append(f"| {year} | {info['input_count']} | {info['source_bad_file_count']} | {info['data_source_bad_file_count']} | {info['data_luminosity_coverage_complete']} | {len(info['nominal_GNN_zero_numerator_bins'])} / {len(info['nominal_GNN_signed_numerator_bins'])} |")
    lines += ["", "No smoothing, floors, fabricated missing templates, SR observations, or VR likelihood channels were introduced. RZ post plots use fitted data and are not independent closure tests; TF reconstruction and Q/Sgamma identities are mechanical checks, not physics closure.",
              "", "## Reused plotting entrypoints", "",
              "dy_estimation report, build_sgamma_ut_report_2024.py, build_zgamma_double_ratio_2024.py, and plot_recoil_transfer_factors_2024.py. Plot-only rendering does not change fitted factor JSONs. Cards, limits, impacts and web publication remain with the main task."]
    (args.output_root / "README.md").write_text("\n".join(lines) + "\n")
    write(args.output_root / "campaign_state.json", result)
    print(json.dumps({"status": result["status"], "report": str(args.output_root / "README.md"),
                      "years": {y: {"input_count": v["input_count"], "signed_bins": v["nominal_GNN_signed_numerator_bins"]} for y, v in result["years"].items()}}))


if __name__ == "__main__":
    main()

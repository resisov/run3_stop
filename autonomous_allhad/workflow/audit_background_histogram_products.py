#!/usr/bin/env python3
"""Audit and summarize the two-year histogram-only background products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnn_background_histograms import sha256


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
        "status": "comparison_complete_pending_adoption",
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
        "proposal_applied": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    result = {"status": "products_complete_with_open_domain_decision", "input_manifest": str(args.manifest), "input_manifest_sha256": sha256(args.manifest), "intermediate_root_reread": False, "SR_blinded": True, "years": {}}
    code_paths = ["autonomous_allhad/workflow/" + name for name in (
        "build_histogram_tf_inputs_2024.py", "gnn_background_histograms.py",
        "plot_recoil_transfer_factors_2024.py", "build_sgamma_ut_report_2024.py",
        "build_an_zinv_factors_2024.py", "build_zgamma_double_ratio_2024.py",
        "audit_background_histogram_products.py")]
    code_paths += ["autonomous_allhad/autonomous_allhad/dy_estimation/" + name
                   for name in ("build_measurement.py", "model.py", "report.py", "__main__.py")]
    result["code_sha256"] = {path: sha256(Path(path)) for path in code_paths}
    lines = ["# Histogram-only background estimation — 2024 and 2025", "", "Inputs: canonical DY window 71 < mll < 111 GeV, promoted 2026-09-07. No intermediate ROOT or NanoAOD input was opened. Nominal histograms were not modified or reweighted.", "", "High-dM final template: retained 73 bins. Low-dM final template: frozen GNN30; Nb1/Nb2plus are factor-measurement groups, not new SR categories.", "", "## RZ results", "", "| Year | Region | Nb = 1 | Nb ≥ 2 |", "|---|---|---|---|"]
    for year, info in manifest["years"].items():
        base = args.output_root / year
        inputs = {}
        for name in ("main", "background_estimation", "gnn"):
            source = Path(info[name]["path"])
            actual = sha256(source)
            if actual != info[name]["sha256"]:
                raise ValueError(f"input checksum mismatch: {source}")
            alias = Path(info[name]["canonical_alias"])
            if alias.resolve() != source.resolve():
                raise ValueError(f"canonical alias no longer points at frozen input: {alias}")
            inputs[name] = {"path": str(source), "sha256": actual}
        compact = read(Path(inputs["background_estimation"]["path"]))
        rz = read(base / "dy_measurement.json")
        sg = read(base / "sgamma/sgamma_ut.json")
        tf = read(base / "tf_inputs.json")
        if tf["lowdm_gnn"]["provenance"]["gnn_input"]["sha256"] != inputs["gnn"]["sha256"]:
            raise ValueError("GNN mapping source checksum mismatch")
        if tf["lowdm_gnn"]["provenance"]["gnn_config"]["sha256"] != manifest["lowdm"]["configuration_sha256"]:
            raise ValueError("GNN mapping configuration checksum mismatch")
        plots = read(base / "tf" / f"transfer_factors_{year}_nb_recoil.json")
        old = read(base / "zgamma/zgamma_double_ratio.json")
        new = read(base / "zgamma_250_proposal/zgamma_double_ratio.json")
        for name, product in (("RZ", rz), ("Sgamma", sg), ("TF", tf), ("double_ratio", old)):
            if product["status"] != "complete" or product["provenance"]["hist_input_sha256"] != inputs["background_estimation"]["sha256"]:
                raise ValueError(f"invalid/stale {name} for {year}")
        for regime in ("highdm", "lowdm"):
            key = "rz_high" if regime == "highdm" else "rz_low"
            for group, record in rz[key]["combined"].items():
                if record["status"] != "complete" or record["RZ"] <= 0:
                    raise ValueError(f"invalid RZ: {year}/{regime}/{group}")
            for channel in rz[key]["channels"].values():
                for record in channel.values():
                    if record["status"] != "complete" or np.linalg.eigvalsh(record["covariance"]).min() < -1e-10:
                        raise ValueError("invalid RZ/RT fitted covariance")
            label = "rz_high.json" if regime == "highdm" else "rz_low.json"
            write(base / label, {"status": "complete", key: rz[key], "provenance": rz["provenance"]})
            r = rz[key]["combined"]
            lines.append(f"| {year} | {regime} | {r['Nb1']['RZ']:.4f} ± {r['Nb1']['RZ_stat']:.4f} | {r['Nb2plus']['RZ']:.4f} ± {r['Nb2plus']['RZ_stat']:.4f} |")
        group_keys = [("highdm", "rz_high", "Nb1"), ("highdm", "rz_high", "Nb2plus"), ("lowdm", "rz_low", "Nb1"), ("lowdm", "rz_low", "Nb2plus")]
        covariance = {"status": "complete", "order": [reg + "_" + group for reg, _, group in group_keys],
                      "central": [rz[key]["combined"][group]["RZ"] for _, key, group in group_keys],
                      "covariance": np.diag([rz[key]["combined"][group]["RZ_stat"]**2 for _, key, group in group_keys]).tolist(),
                      "cross_group_policy": "Existing downstream diagonal assumption retained; cross-regime covariance is not measured by these marginal counts.",
                      "channel_RZ_RT_covariance": {reg: rz[key]["channels"] for reg, key, _ in group_keys}}
        write(base / "rz_covariance.json", covariance)
        compare = comparison(compact, old, new, tf["lowdm_gnn"])
        write(base / "double_ratio_domain_comparison.json", compare)
        write(base / "lowdm_gnn_backgrounds.json", {key: value for key, value in tf["lowdm_gnn"].items() if key != "histograms"})
        zero_bins = []
        for route, categories in tf["lowdm_gnn"]["transfer_factors"]["nominal"].items():
            for category, record in categories.items():
                if record["status"] != "complete":
                    raise ValueError(f"invalid nominal GNN TF {year}/{route}/{category}")
                zero_bins.extend(f"{route}/{category}/{i}" for i, val in enumerate(record["numerator"]) if val == 0)
        recorded_plots = sg["plots"] + old["plots"] + plots["plots"]
        if any(not Path(path).is_file() for path in recorded_plots):
            raise ValueError("missing output plot")
        result["years"][year] = {
            "inputs": inputs, "input_count": info["input_roots"],
            "data_luminosity_coverage_complete": info["data_luminosity_coverage_complete"],
            "data_source_bad_file_count": info["data_source_bad_file_count"],
            "source_bad_file_count": info["source_bad_file_count"],
            "normalization_sha256": info["normalization_sha256"],
            "gnn_checks": tf["lowdm_gnn"]["mechanical_checks"], "nominal_GNN_zero_numerator_bins": zero_bins,
            "double_ratio_domain_comparison": {key: compare[key] for key in ("status", "max_common_D_absolute_shift", "max_common_GNN_relative_response_shift", "proposal_applied")},
            "artifact_hashes": {str(path.relative_to(base)): sha256(path) for path in sorted(base.rglob("*")) if path.is_file()},
        }
    lines += ["", "Errors above use the existing on/off-Z profile fit (Poisson data, weighted-MC template constraints) and inverse-variance ee/μμ combination. Per-channel RZ–RT covariance is exported; the combined cross-group diagonal covariance retains the existing downstream assumption, not a new measurement of absent cross-region correlations.", "", "## GNN transfer and shape propagation", "", "`lowdm_gnn_backgrounds.json` contains the frozen 6×5 SR bins, 4 CR parent categories, all available weight variations, parent-integrated TF coefficients, and Zinv joint components. `tf_inputs.json` additionally retains the selected background/data-CR joint histograms. No SR data are exported.", "", "- Top = TT + ST. W and QCD are separate measured processes; existing card parameter sharing is not changed.", "- GNN coefficient: SR category score-bin yield divided by its existing CR parent's target-process integral. CR score fractions preserve the five-bin CR shape. Equal SR/CR score indices are never divided because their physical edges differ.", "- Z prediction: RZ(Nb) × ΣUT H_Zinv(GNN,UT) Sgamma(Nb,UT). Q is not propagated. Sgamma is shape-normalized in the GCR; no extra normalization of the Z template is imposed.", "- The shared CR denominator induces covariance across GNN bins and SR children of the same parent. Within-category covariance and the shared denominator total/variance are exported; parent coefficients are not independent measurements.", "- U_T TF plots remain Nb-only diagnostics, not Low-dM final templates. GNN TF plots are in each year's `tf/gnn/` directory.", "", "## Double-ratio domain decision (not yet adopted)", "", "The adopted 300-start definition and proposed 250-start definition are both recomputed from the same new inputs. Proposal plots and JSON are isolated in `zgamma_250_proposal/`. `double_ratio_domain_comparison.json` records CR support, normalization changes, every common bin, and per-UT-bin GNN response changes.", "", "| Year | Max common-bin |ΔD| | Max GNN fractional-response change |", "|---|---|---|"]
    for year, item in result["years"].items():
        c = item["double_ratio_domain_comparison"]
        lines.append(f"| {year} | {c['max_common_D_absolute_shift']:.6f} | {100*c['max_common_GNN_relative_response_shift']:.4f}% |")
    lines += ["", "Downstream field: `downstream_central_abs_deviation = abs(D−1)`. The historical figure's `systematic = max(abs(D−1), stat)` is a reporting quantity and is not substituted for the current card nuisance. The 300-start result does not constrain 250–300; missing support must not be interpreted as zero uncertainty. No nuisance or card changes were made.", "", "## Limits and validation", "", "- 2025 retained inputs are complete as a retained set (8390/8390), but upstream skips include one data file: luminosity coverage is not complete. Existing normalization bookkeeping was retained.", "- Zero QCD GNN numerator bins remain zero and are listed in `campaign_state.json`; no smoothing, floors, or invented events were introduced.", "- RZ mll post plots use the fitted data and are not independent closure. Q/Sgamma normalization identities and TF reconstruction checks are mechanical identities, not physics closure.", "- High-dM source histograms contain 79 original bins; the main agent must continue its established 1–6 exclusion for the final 73. Legacy Low-dM34 is not a final template here.", "- The double-ratio domain change remains a proposal until adopted. Background products are not automatically injected into cards or published to the web.", "", "## Existing plotting entrypoints", "", "`python -m autonomous_allhad.dy_estimation report`, `build_sgamma_ut_report_2024.py`, `build_zgamma_double_ratio_2024.py`, and `plot_recoil_transfer_factors_2024.py` were reused for both years. No independent replacement plotting implementation was created."]
    (args.output_root / "README.md").write_text("\n".join(lines).replace("Max common-bin |ΔD|", "Max common-bin absolute ΔD") + "\n")
    write(args.output_root / "campaign_state.json", result)
    print(json.dumps({"status": result["status"], "report": str(args.output_root / "README.md"), "years": {y: v["double_ratio_domain_comparison"] for y, v in result["years"].items()}}))


if __name__ == "__main__":
    main()

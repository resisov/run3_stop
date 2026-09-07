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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-low250", action="store_true", help="Preserve verified 300-start products before approved adoption.")
    args = parser.parse_args()
    if args.prepare_low250:
        preserve_300(args.output_root)
        print(json.dumps({"status": "300_start_preserved", "baseline": str(args.output_root / "low250_adoption_baseline.json")}))
        return
    baseline_path = args.output_root / "low250_adoption_baseline.json"
    baseline = read(baseline_path)
    manifest = read(args.manifest)
    result = {"status": "complete", "input_manifest": str(args.manifest), "input_manifest_sha256": sha256(args.manifest), "intermediate_root_reread": False, "SR_blinded": True, "years": {},
              "lowdm_double_ratio_adoption": {"min_ut_gev": 250, "status": "adopted", "approval": baseline["approval"], "baseline_path": str(baseline_path), "baseline_sha256": sha256(baseline_path)}}
    result["downstream_10gev_veto_use"] = {
        "status": "blocked", "valid_only_for_recorded_input_hashes": True,
        "evidence_source": "Main-agent audit of dy_window20_20260907 production.argv, frozen builder defaults and GNN object masks",
        "reason": "Recorded canonical histograms still use the 5-GeV lepton veto and low-pT SF inputs; these are not 10-GeV-veto results.",
        "nominal_or_SF_reprocessed": False,
    }
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
        old = read(base / "zgamma_300_previous/zgamma_double_ratio.json")
        new = read(base / "zgamma/zgamma_double_ratio.json")
        reference = read(base / "zgamma_250_proposal/zgamma_double_ratio.json")
        if new["adoption_status"] != "adopted" or new["lowdm"] != reference["lowdm"] or new["highdm"] != old["highdm"]:
            raise ValueError("adoption changed validated 250 values or High-dM physics")
        for name, product in (("RZ", rz), ("Sgamma", sg), ("TF", tf), ("double_ratio", new)):
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
        prior_comparison = read(base / "zgamma_300_previous/domain_comparison_before_adoption.json")
        for key in ("first_bin_support", "normalizations", "common_bins", "new_250_300_bin", "gnn_impacts"):
            if compare[key] != prior_comparison[key]:
                raise ValueError(f"historical comparison values changed: {year}/{key}")
        write(base / "double_ratio_domain_comparison.json", compare)
        write(base / "lowdm_gnn_backgrounds.json", {key: value for key, value in tf["lowdm_gnn"].items() if key != "histograms"})
        uncertainty = project_double_ratio_uncertainty(tf["lowdm_gnn"], new)
        uncertainty["provenance"] = {
            "double_ratio": {"path": str(base / "zgamma/zgamma_double_ratio.json"), "sha256": sha256(base / "zgamma/zgamma_double_ratio.json")},
            "gnn_mapping": {"path": str(base / "lowdm_gnn_backgrounds.json"), "sha256": sha256(base / "lowdm_gnn_backgrounds.json")},
            "approval": baseline["approval"], "intermediate_root_reread": False,
        }
        write(base / "lowdm_gnn_double_ratio.json", uncertainty)
        zero_bins = []
        for route, categories in tf["lowdm_gnn"]["transfer_factors"]["nominal"].items():
            for category, record in categories.items():
                if record["status"] != "complete":
                    raise ValueError(f"invalid nominal GNN TF {year}/{route}/{category}")
                zero_bins.extend(f"{route}/{category}/{i}" for i, val in enumerate(record["numerator"]) if val == 0)
        recorded_plots = sg["plots"] + new["plots"] + plots["plots"]
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
            "adoption_checks": {"Highdm_physics_unchanged": True, "validated_250_values_unchanged": True,
                                "historical_comparison_values_unchanged": True, "GNN_nominal_unchanged": True,
                                "new_250_300_max_fractional_up_response": max(abs(r["responses"][0]["fractional_up_response"][i]) for r in uncertainty["categories"].values() for i in range(5)),
                                "response_count": sum(len(r["responses"]) for r in uncertainty["categories"].values())},
            "artifact_hashes": {str(path.relative_to(base)): sha256(path) for path in sorted(base.rglob("*")) if path.is_file()},
        }
    lines += ["", "Errors above use the existing on/off-Z profile fit (Poisson data, weighted-MC template constraints) and inverse-variance ee/μμ combination. Per-channel RZ–RT covariance is exported; the combined cross-group diagonal covariance retains the existing downstream assumption, not a new measurement of absent cross-region correlations.", "", "## GNN transfer and shape propagation", "", "`lowdm_gnn_backgrounds.json` contains the frozen 6×5 SR bins, 4 CR parent categories, all available weight variations, parent-integrated TF coefficients, and Zinv joint components. `tf_inputs.json` additionally retains the selected background/data-CR joint histograms. No SR data are exported.", "", "- Top = TT + ST. W and QCD are separate measured processes; existing card parameter sharing is not changed.", "- GNN coefficient: SR category score-bin yield divided by its existing CR parent's target-process integral. CR score fractions preserve the five-bin CR shape. Equal SR/CR score indices are never divided because their physical edges differ.", "- Z prediction: RZ(Nb) × ΣUT H_Zinv(GNN,UT) Sgamma(Nb,UT). Q is not propagated. Sgamma is shape-normalized in the GCR; no extra normalization of the Z template is imposed.", "- The shared CR denominator induces covariance across GNN bins and SR children of the same parent. Within-category covariance and the shared denominator total/variance are exported; parent coefficients are not independent measurements.", "- U_T TF plots remain Nb-only diagnostics, not Low-dM final templates. GNN TF plots are in each year's `tf/gnn/` directory.", "", "## Double-ratio domain decision (not yet adopted)", "", "The adopted 300-start definition and proposed 250-start definition are both recomputed from the same new inputs. Proposal plots and JSON are isolated in `zgamma_250_proposal/`. `double_ratio_domain_comparison.json` records CR support, normalization changes, every common bin, and per-UT-bin GNN response changes.", "", "| Year | Max common-bin |ΔD| | Max GNN fractional-response change |", "|---|---|---|"]
    for year, item in result["years"].items():
        c = item["double_ratio_domain_comparison"]
        lines.append(f"| {year} | {c['max_common_D_absolute_shift']:.6f} | {100*c['max_common_GNN_relative_response_shift']:.4f}% |")
    lines += ["", "Downstream field: `downstream_central_abs_deviation = abs(D−1)`. The historical figure's `systematic = max(abs(D−1), stat)` is a reporting quantity and is not substituted for the current card nuisance. The 300-start result does not constrain 250–300; missing support must not be interpreted as zero uncertainty. No nuisance or card changes were made.", "", "## Limits and validation", "", "- 2025 retained inputs are complete as a retained set (8390/8390), but upstream skips include one data file: luminosity coverage is not complete. Existing normalization bookkeeping was retained.", "- Zero QCD GNN numerator bins remain zero and are listed in `campaign_state.json`; no smoothing, floors, or invented events were introduced.", "- RZ mll post plots use the fitted data and are not independent closure. Q/Sgamma normalization identities and TF reconstruction checks are mechanical identities, not physics closure.", "- High-dM source histograms contain 79 original bins; the main agent must continue its established 1–6 exclusion for the final 73. Legacy Low-dM34 is not a final template here.", "- The double-ratio domain change remains a proposal until adopted. Background products are not automatically injected into cards or published to the web.", "", "## Existing plotting entrypoints", "", "`python -m autonomous_allhad.dy_estimation report`, `build_sgamma_ut_report_2024.py`, `build_zgamma_double_ratio_2024.py`, and `plot_recoil_transfer_factors_2024.py` were reused for both years. No independent replacement plotting implementation was created."]
    for relative, expected in baseline["protected_files"].items():
        if sha256(args.output_root / relative) != expected:
            raise ValueError(f"RZ/TF/Sgamma/nominal protected product changed: {relative}")
    for files in list(baseline["preserved_300"].values()) + [baseline["previous_global_records"]]:
        for relative, expected in files.items():
            if sha256(args.output_root / relative) != expected:
                raise ValueError(f"rollback product changed: {relative}")
    result["adoption_invariance"] = {"protected_products_unchanged": len(baseline["protected_files"]), "rollback_checksums_match": True,
                                    "RZ_TF_Sgamma_nominal_unchanged": True, "cards_limits_web_untouched": True}
    text = "\n".join(lines).replace("Max common-bin |ΔD|", "Max common-bin absolute ΔD")
    text = text.replace("## Double-ratio domain decision (not yet adopted)", "## Double-ratio domain: 250 GeV adopted")
    text = text.replace("The adopted 300-start definition and proposed 250-start definition are both recomputed from the same new inputs. Proposal plots and JSON are isolated in `zgamma_250_proposal/`.", "The user-approved 250-start definition is now canonical in `zgamma/`. Exact previous 300-start products are preserved in `zgamma_300_previous/`, with checksums in `low250_adoption_baseline.json`. `zgamma_250_proposal/` preserves the original proposal as historical provenance.")
    text = text.replace("The double-ratio domain change remains a proposal until adopted.", "The 250-GeV domain was approved on 2026-09-07T22:29:50Z and adopted in both years.")
    text += "\n\n## GNN uncertainty product\n\n`YEAR/lowdm_gnn_double_ratio.json` exports 5 UT-source responses for each of the 6 frozen GNN categories. The nominal is unchanged. Only central abs(D−1) is propagated, with existing up=1+delta and down=1/(1+delta) conventions. No nuisance names, correlations, or rate parameters are introduced. New 250–300 responses (max 0.4754% / 1.3756% for 2024 / 2025) are distinct from common >=300 response changes (max 0.1909% / 0.5817%). These are individual template responses, not total uncertainty, nominal-yield changes, or limit changes.\n"
    text += "\n## Input-selection blocker\n\nThe main agent confirmed that these recorded histogram inputs still use a 5-GeV lepton veto and the associated low-pT SF inputs. They are not valid as results for the newly requested 10-GeV veto. Merely removing 5–10 GeV SF variations does not update event acceptance. Card/limit submission is on hold in the main task while it checks canonical inputs. The complete status here means the approved factor adoption is valid for the recorded input hashes only; no nominal/SF reprocessing was performed. See HANDOFF.md.\n"
    (args.output_root / "README.md").write_text(text.rstrip() + "\n")
    write(args.output_root / "campaign_state.json", result)
    print(json.dumps({"status": result["status"], "report": str(args.output_root / "README.md"), "years": {y: v["double_ratio_domain_comparison"] for y, v in result["years"].items()}}))


if __name__ == "__main__":
    main()

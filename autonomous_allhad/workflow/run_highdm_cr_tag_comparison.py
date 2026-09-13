#!/usr/bin/env python3
"""Isolated CR-only physics comparison; never promotes a main result."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import math
import re
from pathlib import Path
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def prepare(repo, study, chunk_size):
    if (study / "campaign_state.json").exists():
        raise RuntimeError("Existing campaign: inspect and resume, do not overwrite")
    base = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    frozen = base / "code_main_v2"
    code = study / "code"
    shutil.copytree(frozen, code, symlinks=True)
    overrides = ["build_flat_boosted_recoil_hists.py", "run_flat_hists_chunked.py",
                 "merge_flat_hist_chunks_streaming.py", "plot_control_search_bins_style.py"]
    hashes = {}
    for name in overrides:
        relative = "autonomous_allhad/workflow/" + name
        src, dest = repo / relative, code / relative
        pending = dest.with_name(dest.name + ".comparison")
        shutil.copy2(src, pending)
        os.replace(pending, dest)
        hashes[relative] = digest(dest)
    state = {"schema_version": "highdm_cr_tag_comparison_v1", "status": "prepared",
             "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "physics_status": "comparison_not_adopted", "main_replacement": False,
             "selection": "existing High-dM CR AND Nt + Nw + Nres >= 1",
             "years": {}, "code_sha256": hashes, "chunk_size": chunk_size,
             "base_manifest_sha256": digest(base / "plots/manifest.json"),
             "base_canonical_manifest_sha256": digest(repo / "autonomous_allhad/reports/lepton_veto10_canonical_20260908.json"),
             "regions": ["LLCR", "QCDCR", "GCR", "DY2E", "DY2M"],
             "scope_exclusions": ["SR", "low-dM", "VR", "limits", "impacts", "AN", "main web pages"],
             "caveats": ["Retains upstream input omissions and uncertified complete 2025 data luminosity coverage.",
                         "No recalibration of RZ or other background corrections for this comparison."]}
    jobs = []
    pilots = []
    for year in ("2024", "2025"):
        directory = study / year
        directory.mkdir(parents=True, exist_ok=True)
        original = (base / year / "inputs.txt").read_text().splitlines()
        roots = [x for x in original if Path(x).name.startswith(("data_shard_", "mc_shard_"))]
        excluded = [x for x in original if x not in set(roots)]
        if not all(Path(x).name.startswith("signal_shard_") for x in excluded):
            raise RuntimeError("Unexpected non-signal exclusion")
        if len(roots) != len(set(roots)):
            raise RuntimeError("Duplicate inputs")
        (directory / "inputs.txt").write_text("\n".join(roots) + "\n")
        norm = base / year / "norm.json"
        production = (base / year / "production.argv").read_text().splitlines()
        efficiency_hash = production[production.index("--expected-btag-efficiency-sha256") + 1]
        def argv(inputs, output):
            return ["/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python",
                    str(code / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"),
                    "--repo", str(code), "--inputs", *inputs, "--normalization", str(norm),
                    "--output", str(output), "--campaign-year", year, "--step-size", "50000",
                    "--only-regions", *state["regions"], "--distribution-only", "--highdm-only",
                    "--require-highdm-cr-tag", "--require-btag", "--require-branches",
                    "--require-normalization", "--allow-zero-entry-roots",
                    "--expected-btag-efficiency-sha256", efficiency_hash,
                    "--require-weight-components", "pileup", "btagSF", "electron_id", "electron_reco",
                    "muon_id", "muon_iso", "muon_hlt", "photon_id", "met_trigger", "photon_trigger",
                    "--analysis-sf-components", "met_trigger", "photon_trigger", "topw_tagging",
                    "--electron-veto-pt-min", "10", "--muon-veto-pt-min", "10", "--dy-ptll-policy", "all"]
        chunks = [roots[i:i + chunk_size] for i in range(0, len(roots), chunk_size)]
        for i, inputs in enumerate(chunks):
            job = f"{year}_{i:04d}"
            target = directory / "chunks" / f"{job}.json"
            target.parent.mkdir(exist_ok=True)
            saved = study / "argv" / f"{job}.argv"
            saved.parent.mkdir(exist_ok=True)
            saved.write_text("\n".join(argv(inputs, target)) + "\n")
            jobs.append(job)
        pilot_inputs = [next(x for x in roots if Path(x).name.startswith(prefix))
                        for prefix in ("data_shard_", "mc_shard_")]
        pilot_name = f"{year}_pilot"
        pilot_argv = study / "argv" / f"{pilot_name}.argv"
        pilot_argv.write_text("\n".join(argv(pilot_inputs, directory / "pilot.json")) + "\n")
        pilots.append(pilot_name)
        state["years"][year] = {"inputs": len(roots), "excluded_signal_inputs": len(excluded),
                                "original_inputs_sha256": digest(base / year / "inputs.txt"),
                                "inputs_sha256": digest(directory / "inputs.txt"),
                                "normalization_sha256": digest(norm), "chunks": len(chunks)}
    (study / "logs").mkdir(exist_ok=True)
    submission = f'''universe = vanilla
initialdir = {study}
executable = {repo}/autonomous_allhad/workflow/run_batch.sh
arguments = {study}/argv/$(job).argv
getenv = False
environment = "PYTHONPATH={code}/autonomous_allhad AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA=0"
output = {study}/logs/$(job).$(ClusterId).out
error = {study}/logs/$(job).$(ClusterId).err
log = {study}/logs/jobs.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = TRUE
transfer_input_files = /eos/user/t/taiwoo/run3_stop/runtime/py38.tgz
transfer_output_files = ""
use_x509userproxy = true
x509userproxy = /eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757
request_cpus = 1
request_memory = 8000MB
request_disk = 4000MB
+MaxRuntime = 21600
+JobBatchName = "NPS26012_HighDM_CR_TagGE1_Comparison"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue job in (
'''
    (study / "production.sub").write_text(submission + "\n".join(jobs) + "\n)\n")
    (study / "pilot.sub").write_text(submission + "\n".join(pilots) + "\n)\n")
    state["expected_jobs"] = len(jobs)
    save(study / "campaign_state.json", state)
    print(json.dumps(state, indent=2))


def validate_leaf_tree(node):
    if isinstance(node, dict) and {"sumw", "sumw2", "entries"} <= node.keys():
        if len({len(node[k]) for k in ("sumw", "sumw2", "entries")}) != 1:
            raise RuntimeError("Inconsistent histogram array lengths")
        if not all(math.isfinite(float(v)) for k in ("sumw", "sumw2", "entries") for v in node[k]):
            raise RuntimeError("Non-finite histogram content")
        if any(v < 0 for k in ("sumw2", "entries") for v in node[k]):
            raise RuntimeError("Negative histogram variance or count")
    elif isinstance(node, dict):
        for value in node.values():
            validate_leaf_tree(value)


def collect(repo, study):
    from run_flat_hists_chunked import merge_payloads
    state = json.loads((study / "campaign_state.json").read_text())
    cluster = state["production_cluster"]
    log = (study / "logs" / f"jobs.{cluster}.log").read_text()
    exits = {}
    for event in log.split("...\n"):
        match = re.search(r"^005 \((\d+)\.(\d+)\.\d+\)", event, re.M)
        if match and int(match[1]) == cluster:
            normal = re.search(r"Normal termination \(return value (\d+)\)", event)
            exits[int(match[2])] = int(normal[1]) if normal else -1
    if exits != {i: 0 for i in range(state["expected_jobs"])}:
        raise RuntimeError(f"Missing successful exit events: {len(exits)}/{state['expected_jobs']}")
    base = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    if digest(base / "plots/manifest.json") != state["base_manifest_sha256"]:
        raise RuntimeError("Baseline plot manifest changed")
    for year, record in state["years"].items():
        directory = study / year
        paths = [directory / "chunks" / f"{year}_{i:04d}.json" for i in range(record["chunks"])]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise RuntimeError(f"{year}: {len(missing)} chunks not complete")
        target = directory / "comparison_payload.json"
        merged = merge_payloads(paths, target, base / year / "norm.json", allow_zero_entry_roots=True)
        if merged["status"] != "complete":
            raise RuntimeError("Merged result is not complete")
        options = merged["summary"]["build_options"]
        if not options.get("require_highdm_cr_tag") or options["search_bins"] is not None:
            raise RuntimeError("Incorrect comparison build contract")
        expected = {str(Path(p).resolve()) for p in (directory / "inputs.txt").read_text().splitlines()}
        actual = {str(Path(p).resolve()) for p in merged["summary"]["input_roots"]}
        if actual != expected:
            raise RuntimeError("Merged input coverage mismatch")
        hist = merged["highdm_variable_histograms"]
        if set(hist) != set(state["regions"]):
            raise RuntimeError("Missing or extra CRs")
        if any(merged[k] for k in ("histograms", "search_bin_histograms", "lowdm_variable_histograms")):
            raise RuntimeError("Comparison contains out-of-scope histogram products")
        validate_leaf_tree(hist)
        for region, samples in merged["summary"]["highdm_cr_tag_audit"].items():
            for sample, audit in samples.items():
                for key in ("entries", "sumw", "sumw2"):
                    if not math.isclose(audit[f"baseline_{key}"], audit[f"selected_{key}"] + audit[f"rejected_{key}"], rel_tol=1e-10, abs_tol=1e-7):
                        raise RuntimeError(f"{region}/{sample}: failed {key} partition")
        merged["highdm_distribution_regions"] = {"control": state["regions"], "signal_categories": [], "validation": []}
        merged["comparison_provenance"] = {"physics_status": "comparison_not_adopted", "main_replacement": False,
                                           "baseline_manifest_sha256": state["base_manifest_sha256"]}
        save(target, merged)
        record.update({"status": "histograms_validated", "validated_inputs": len(actual),
                       "payload_sha256": digest(target), "payload_bytes": target.stat().st_size})
        print(json.dumps({"year": year, **record}), flush=True)
    state["status"] = "histograms_validated"
    save(study / "campaign_state.json", state)


def render(repo, study):
    import plot_control_search_bins_style as style
    base = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    state = json.loads((study / "campaign_state.json").read_text())
    if state["status"] not in ("histograms_validated", "plots_rendered"):
        raise RuntimeError("Histograms must be validated before plotting")
    plots = []
    comparison = {}
    for year, record in state["years"].items():
        payload_path = study / year / "comparison_payload.json"
        if digest(payload_path) != record["payload_sha256"]:
            raise RuntimeError("Comparison payload hash changed")
        payload = json.loads(payload_path.read_text())
        sources = set()
        for variables in payload["highdm_variable_histograms"].values():
            for samples in variables.values():
                for variations in samples.values():
                    sources.update(v[:-2] for v in variations if v.endswith("Up"))
        baseline_summary = json.loads((base / "plots" / year / "highdm/LLCR/plot_summary.json").read_text())
        style.LUMINOSITY_FB = baseline_summary["luminosity_fb"]
        style.LUMINOSITY_RELATIVE_UNCERTAINTY = baseline_summary["luminosity_relative_uncertainty"]
        style.PLOT_SYSTEMATIC_SOURCES[:] = [s for s in baseline_summary["background_systematic_sources"] if s in sources]
        for region in ("LLCR", "QCDCR", "GCR", "DYCR"):
            reference = json.loads((base / "plots" / year / "highdm" / region / "plot_summary.json").read_text())
            expected_variables = {p["variable"] for p in reference["plots"]}
            summary = style.draw_highdm_distribution_report(
                payload_path, study / "plots" / year / region, year,
                dy_rz_manifest=base / "plots" / year / "dy_rz_plotting.json", only_region=region,
                exclude_variables=[v for v in payload["highdm_distribution_variable_specs"] if v not in expected_variables])
            if {p["variable"] for p in summary["plots"]} != expected_variables:
                raise RuntimeError(f"{year}/{region}: plot variable coverage differs")
            for plot in summary["plots"]:
                for ext in ("pdf", "png"):
                    plot[ext] = str(Path(plot[ext]).resolve().relative_to(study))
                plots.append(plot)
        comparison[year] = payload["summary"]["highdm_cr_tag_audit"]
    manifest = {"status": "rendered_awaiting_visual_review", "physics_status": "comparison_not_adopted",
                "selection": state["selection"], "main_replacement": False,
                "plots": plots, "plot_count": len(plots), "caveats": state["caveats"],
                "event_selection_comparison": comparison,
                "uncertainty": "MC stat plus available weight variations and luminosity; GCR independently unit-area normalized, luminosity omitted",
                "baseline_manifest_sha256": state["base_manifest_sha256"]}
    save(study / "plots/manifest.json", manifest)
    rows = []
    for plot in plots:
        reference = base / "plots" / plot["year"] / "highdm" / plot["region"] / (plot["name"] + ".png")
        old = os.path.relpath(reference, study)
        rows.append(f'<section><h2>{plot["year"]} {plot["region"]}: {plot["variable"]}</h2>'
                    f'<div class="pair"><figure><figcaption>Existing main selection</figcaption><img loading="lazy" src="{old}"></figure>'
                    f'<figure><figcaption>Nt + Nw + Nres &ge; 1 (comparison only)</figcaption><a href="{plot["pdf"]}"><img loading="lazy" src="{plot["png"]}"></a></figure></div></section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>High-dM CR tag comparison</title>'
    page += '<style>body{font-family:system-ui;margin:24px;background:#f5f7fa;color:#182434}h1{margin-bottom:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}section{margin:32px 0}figcaption{text-align:center}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>'
    page += '<h1>High-dM CR: Nt + Nw + Nres &ge; 1</h1><p>Physics comparison only. Main plots, AN, SR, low-dM, limits and impacts are unchanged.</p><p>2024 and 2025. Existing SF, RZ, normalization and other selection policies retained. GCR is unit-area normalized. Complete 2025 data luminosity coverage remains uncertified.</p>'
    page += ''.join(rows) + '</html>'
    (study / "index.html").write_text(page)
    state["status"] = "plots_rendered"
    state["plot_count"] = len(plots)
    save(study / "campaign_state.json", state)
    print(json.dumps({"plots": len(plots), "gallery": str(study / "index.html")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "collect", "render"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    if args.mode == "prepare":
        # Keep the approved /eos/user spelling in saved batch arguments.
        # Path.resolve() changes it to /eos/home-t on lxplus.
        prepare(args.repo.absolute(), args.study.absolute(), args.chunk_size)
    elif args.mode == "collect":
        collect(args.repo.resolve(), args.study.resolve())
    else:
        render(args.repo.resolve(), args.study.resolve())


if __name__ == "__main__":
    main()

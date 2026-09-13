#!/usr/bin/env python3
"""Isolated CR-only physics comparison; never promotes a main result."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    prepare(args.repo.resolve(), args.study.resolve(), args.chunk_size)


if __name__ == "__main__":
    main()

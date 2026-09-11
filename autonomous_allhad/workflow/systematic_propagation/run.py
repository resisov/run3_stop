#!/usr/bin/env python3
"""Separate kinematic-variation jobs using the existing analysis executables."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).absolute().parent
SHIFTS = ("nominal", "metUnclusteredUp", "metUnclusteredDown")
TROTA_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh"
WEIGHTS = ("pileup", "btagSF", "electron_id", "electron_reco", "muon_id",
           "muon_iso", "muon_hlt", "photon_id", "met_trigger", "photon_trigger")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eos(path):
    path = Path(path).absolute()
    if not str(path).startswith("/eos/") or any(x in ("afs", "tmp") for x in path.parts):
        raise ValueError("expected EOS path: " + str(path))
    return path


def execute(argv, env, cwd):
    print(json.dumps({"command": list(map(str, argv))}), flush=True)
    subprocess.run(list(map(str, argv)), env=env, cwd=cwd, check=True)


def source_records(sidecar, year):
    metadata = read(sidecar)
    if metadata["status"] != "complete" or metadata.get("bad_files"):
        raise ValueError("source shard is not complete: " + str(sidecar))
    datasets = {v["dataset"]: v for v in metadata["datasets"].values()}
    records = []
    for index, item in enumerate(metadata["files"]):
        info = datasets[item["dataset"]]
        if info["is_data"] or item["read_status"] != "success":
            raise ValueError("pilot requires successful MC sources")
        record = dict(dataset=item["dataset"], process_group=item["process"],
                      file_path=item["file_path"], file_index=index, year=str(year),
                      is_data=False, is_signal=info["is_signal"],
                      is_background=info["is_background"], xsec_pb=info["xsec_pb"])
        if item.get("fastsim_trigger_bypass"):
            record["simulation_type"] = "FastSim signal dataset"
        records.append(record)
    if len(records) != metadata["files_processed"]:
        raise ValueError("source file accounting mismatch")
    return records


def prepare(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    canonical_path = repo / "autonomous_allhad/reports/lepton_veto10_canonical_20260908.json"
    canonical = read(canonical_path)
    nominal = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    main_repo = repo / canonical["production"]["frozen_main_code"]
    gnn_repo = repo / canonical["production"]["frozen_gnn_code"]
    all_jobs = []
    for year in (2024, 2025):
        base = campaign / str(year)
        source = repo / f"autonomous_allhad/workflow/flat{year}_v8/outputs/nominal/mc_shard_00000.root"
        source_list = nominal / str(year) / "inputs.txt"
        if str(source) not in source_list.read_text().splitlines():
            raise ValueError("pilot ROOT is not in canonical input list")
        records = source_records(source.with_suffix(".json"), year)
        shard = dict(schema_version="full_production_shard_spec_v2_boosted",
                     shard_id="mc_shard_00000", record_group="mc", records=records,
                     records_per_shard=len(records),
                     record_digest=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16])
        shard_path = base / "pilot/shard.json"
        write(shard_path, shard)
        model = repo / f"autonomous_allhad/workflow/flat{year}_v8/bundles/model_TopResolved_2024_TROTA2D_ptcut.h5"
        norm = nominal / str(year) / "norm.json"
        search = main_repo / f"autonomous_allhad/configs/search_bins_{year}.json"
        if sha(norm) != canonical["years"][str(year)]["normalization_sha256"]:
            raise ValueError("canonical normalization changed")
        if sha(search) != canonical["years"][str(year)]["search_bin_config_sha256"]:
            raise ValueError("canonical search configuration changed")
        gnn_model = gnn_repo / "autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010"
        config = dict(year=year, status="pilot_prepared", repo=str(repo),
                      main_repo=str(main_repo), gnn_repo=str(gnn_repo),
                      source_root=str(source), source_sidecar=str(source.with_suffix(".json")),
                      shard=str(shard_path), normalization=str(norm), search_config=str(search),
                      trota_model=str(model), trota_setup=TROTA_SETUP,
                      gnn_manifest=str(nominal / str(year) / "gnn/full_manifest.json"),
                      gnn_model=str(gnn_model / "diagonal_v3_numpy.npz"),
                      gnn_selection=str(gnn_model / "selection.json"),
                      gnn_configuration=str(gnn_repo / "autonomous_allhad/gnn_lowdm/config.json"),
                      stop_xsec=str(gnn_repo / "autonomous_allhad/signals/stop_xsec_13p6TeV.json"),
                      btag_efficiency_sha256="9326454608126467219a60f8d5763a6c92465127ca6ea32ac415ace8ed0ec004",
                      shifts=list(SHIFTS), root_retention="pilot only, on EOS",
                      canonical_report_sha256=sha(canonical_path), pins={})
        paths = [HERE / "run.py", HERE / "run.sh", canonical_path, source_list,
                 source.with_suffix(".json"), shard_path, norm, search, model,
                 Path(config["gnn_manifest"]), Path(config["gnn_model"]),
                 Path(config["gnn_selection"]), Path(config["gnn_configuration"]),
                 main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"]
        for directory in (repo / "autonomous_allhad/autonomous_allhad",
                          gnn_repo / "autonomous_allhad/gnn_lowdm/_implementation"):
            paths.extend(sorted(directory.glob("*.py")))
        paths.extend([repo / "analysis/data/ids.coffea", repo / "analysis/data/corrections.coffea"])
        for path in paths:
            config["pins"][str(path)] = sha(path)
        config_path = base / "campaign.json"
        write(config_path, config)
        for shift in SHIFTS:
            out = base / "pilot" / shift
            out.mkdir(parents=True, exist_ok=True)
            all_jobs.append(f"{year} {shift} {config_path} {out}")
    logs = campaign / "logs"
    logs.mkdir(exist_ok=True)
    (campaign / "pilot.queue").write_text("\n".join(all_jobs) + "\n")
    runtime = repo.parent / "runtime"
    proxy = Path("/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757")
    for path in (runtime / "py38.tgz", runtime / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
        if not path.is_file():
            raise FileNotFoundError(path)
    submit = f'''universe = vanilla
initialdir = {campaign}
executable = {campaign}/run.sh
arguments = {campaign}/run.py worker --config $(config) --shift $(shift) --output $(output)
getenv = False
output = {logs}/pilot_$(year)_$(shift).$(ClusterId).out
error = {logs}/pilot_$(year)_$(shift).$(ClusterId).err
log = {logs}/pilot.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {runtime}/py38.tgz, {runtime}/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 2
request_memory = 6000MB
request_disk = 18000MB
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "NPS26012_separate_shape_pilot"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue year,shift,config,output from {campaign}/pilot.queue
'''
    (campaign / "pilot.sub").write_text(submit)
    write(campaign / "campaign_state.json", dict(status="pilot_prepared", jobs=len(all_jobs),
          canonical_modified=False, runtime="py38; TROTA-only LCG_104", job_flavour="workday",
          submitted_clusters=[], pending=["pilot execution", "nominal closure", "full MC propagation",
          "JES/JER TROTA-input propagation", "EGM", "MUO/TAU decomposition", "JMS/JMR prescription"]))
    print(json.dumps({"status": "pilot_prepared", "jobs": len(all_jobs), "submit": str(campaign / "pilot.sub")}))


def worker(args):
    config = read(args.config)
    if args.shift not in SHIFTS:
        raise ValueError("shift not validated by this propagation stage")
    for path, expected in config["pins"].items():
        if sha(path) != expected:
            raise RuntimeError("pinned input changed: " + path)
    destination = eos(args.output)
    work = Path(os.environ["_CONDOR_SCRATCH_DIR"]) / "shape"
    work.mkdir()
    repo = Path(config["repo"])
    year = config["year"]
    env = dict(os.environ)
    vendor = os.environ.get("PYTHONPATH", "")
    env.update(PYTHONPATH=f"{repo}/autonomous_allhad:{repo}:{vendor}",
               AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0",
               AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="1", AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE="1",
               AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"),
               AUTONOMOUS_ALLHAD_FRAGMENT_DIR=str(work / "fragments"))
    root = work / "mc_shard_00000.root"
    metadata_path = root.with_suffix(".json")
    report = dict(status="running", year=year, shift=args.shift, stages={}, started=time.time())

    def checkpoint():
        write(destination / "result.json", report)

    try:
        checkpoint()
        module = ("autonomous_allhad.intermediate_2024_worker" if year == 2024
                  else "autonomous_allhad.intermediate_2025_data_worker")
        execute([sys.executable, "-u", "-m", module, "--repo", repo,
                 "--shard", config["shard"], "--output", root, "--metadata-output", metadata_path,
                 "--shift", args.shift, "--record-workers", "1", "--chunk-size", "25000"], env, work)
        metadata = read(metadata_path)
        if metadata["status"] != "complete" or metadata["bad_files"]:
            raise RuntimeError("incomplete shifted Events production")
        report["stages"]["events"] = dict(events_read=metadata["events_read"],
            events_written=metadata["events_written"], files=metadata["files_processed"])
        checkpoint()
        trota_env = dict(env)
        for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
            trota_env.pop(key, None)
        script = ('set -e; set +u; source "$1"; shift; '
                  'export PYTHONPATH="$1/autonomous_allhad:$1${PYTHONPATH:+:$PYTHONPATH}"; shift; '
                  'exec python3 -u -m autonomous_allhad.trota_resolved_2024_inplace "$@"')
        execute(["/bin/bash", "-c", script, "trota", config["trota_setup"], repo,
                 "--input", root, "--model", config["trota_model"],
                 "--metadata-output", work / "trota.json", "--target-year", str(year),
                 "--chunk-events", "20000", "--batch-size", "8192", "--allow-hadd-repair"], trota_env, work)
        trota = read(work / "trota.json")
        if trota["status"] != "complete" or trota["marker"]["status"] != "complete":
            raise RuntimeError("TROTA inference did not complete")
        metadata.update(root_sha256=sha(root), root_trees=["Events", "TROTA"])
        metadata[f"trota_topresolved_{year}"] = trota
        write(metadata_path, metadata)
        report["stages"]["trota"] = trota["counts"]
        checkpoint()
        execute([sys.executable, "-u", "-m", "autonomous_allhad.flat_ntuple_worker",
                 "--repo", repo, "--append-topw-truth", root, "--output", root,
                 "--metadata-output", work / "truth.json", "--truth-year", str(year),
                 "--truth-work-dir", work], env, work)
        report["stages"]["topw"] = read(work / "truth.json")["status"]
        checkpoint()
        main_repo = Path(config["main_repo"])
        hist_env = dict(env, PYTHONPATH=f"{main_repo}/autonomous_allhad:{main_repo}/autonomous_allhad/workflow:{main_repo}:{vendor}")
        execute([sys.executable, "-u", main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py",
                 "--repo", main_repo, "--inputs", root, "--normalization", config["normalization"],
                 "--output", work / "main.json", "--campaign-year", str(year),
                 "--nominal-only", "--require-btag", "--expected-btag-efficiency-sha256", config["btag_efficiency_sha256"],
                 "--require-weight-components", *WEIGHTS, "--analysis-sf-components", "met_trigger", "photon_trigger", "topw_tagging",
                 "--require-branches", "--require-normalization", "--allow-zero-entry-roots",
                 "--electron-veto-pt-min", "10", "--muon-veto-pt-min", "10",
                 "--search-bin-config", config["search_config"], "--dy-ptll-policy", "all"], hist_env, work)
        main = read(work / "main.json")
        if main["status"] != "complete":
            raise RuntimeError("main histogram validation failed")
        report["stages"]["main"] = main["status"]
        del main
        checkpoint()
        gnn_repo = Path(config["gnn_repo"])
        gnn_env = dict(env, PYTHONPATH=f"{gnn_repo}/autonomous_allhad:{gnn_repo}/autonomous_allhad/workflow:{gnn_repo}:{vendor}")
        request = dict(kind="mc", batch=0, manifest=config["gnn_manifest"], repository=str(gnn_repo),
                       stop_xsec=config["stop_xsec"], inputs=[dict(root=str(root), sidecar=str(metadata_path))])
        write(work / "gnn_request.json", request)
        execute([sys.executable, "-u", "-m", "gnn_lowdm.eval", "cr-partial",
                 "--request", work / "gnn_request.json", "--output", work / "gnn.json",
                 "--model", config["gnn_model"], "--selection", config["gnn_selection"],
                 "--configuration", config["gnn_configuration"], "--raw-dy",
                 "--regions", "SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M"], gnn_env, work)
        gnn = read(work / "gnn.json")
        if gnn["status"] != "complete" or gnn["bad_files"] or gnn["input_files_valid"] != 1:
            raise RuntimeError("GNN histogram validation failed")
        report["stages"]["gnn"] = gnn["status"]
        del gnn
        products = {}
        for name in (root.name, metadata_path.name, "main.json", "gnn.json", "trota.json", "truth.json"):
            target = destination / name
            if target.exists():
                raise FileExistsError(target)
            temporary = target.with_name(name + ".partial")
            shutil.copyfile(work / name, temporary)
            expected = sha(work / name)
            if sha(temporary) != expected:
                raise RuntimeError("stage-out checksum mismatch")
            temporary.replace(target)
            products[name] = dict(path=str(target), sha256=expected, bytes=target.stat().st_size)
        report.update(status="complete", products=products, finished=time.time(),
                      validation="individual shifted job complete; nominal closure pending",
                      config_sha256=sha(args.config))
        checkpoint()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", finished=time.time())
        checkpoint()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--repo", required=True, type=Path)
    job = commands.add_parser("worker")
    job.add_argument("--config", required=True, type=Path)
    job.add_argument("--shift", required=True, choices=SHIFTS)
    job.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else worker(args)


if __name__ == "__main__":
    main()

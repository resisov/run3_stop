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
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).absolute().parent
SHIFTS = ("nominal", "metUnclusteredUp", "metUnclusteredDown")
TROTA_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh"
WEIGHTS = ("pileup", "btagSF", "electron_id", "electron_reco", "muon_id",
           "muon_iso", "muon_hlt", "photon_id", "met_trigger", "photon_trigger")


def read(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    config = read(path)
    if "shared_config" in config:
        shared_path = eos(config["shared_config"])
        if sha(shared_path) != config["shared_config_sha256"]:
            raise RuntimeError("shared configuration changed")
        shared = read(shared_path)
        pins = {**shared.pop("pins"), **config.get("pins", {})}
        shared.update(config)
        config = dict(shared, pins=pins)
    return config


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


def prepare_payload(config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(config["object_payload_bundle"]) as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("unsupported payload member: " + member.name)
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    return destination


def source_records(sidecar, year, metadata=None):
    metadata = read(sidecar) if metadata is None else metadata
    if metadata["status"] not in ("complete", "complete_with_bad_files"):
        raise ValueError("source shard is not complete: " + str(sidecar))
    datasets = {v["dataset"]: v for v in metadata["datasets"].values()}
    records = []
    for index, item in enumerate(metadata["files"]):
        info = datasets[item["dataset"]]
        if info["is_data"]:
            raise ValueError("shape production requires MC sources")
        if item["read_status"] != "success":
            continue
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


def prepare_full(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    state = read(campaign / "campaign_state.json")
    target = eos(campaign / args.label)
    if (target / "campaign_state.json").exists():
        raise FileExistsError("campaign already prepared: " + str(target))
    summary = dict(status="prepared", scope="all canonical background and signal MC",
                   shifts=list(SHIFTS[1:]), job_flavour="workday", years={},
                   canonical_modified=False, canonical_promotion=False, submitted_clusters=[])
    for year in args.years:
        if state["pilot_validation"][str(year)]["validation_cli"]["status"] != "passed":
            raise RuntimeError("nominal pilot closure is required")
        pilot_config = campaign / str(year) / ("campaign.json" if year == 2024 else "pilot_retry1.json")
        pilot = read(pilot_config)
        shared = dict(pilot)
        for path, expected in shared["pins"].items():
            if Path(path).name in ("run.py", "run.sh"):
                continue
            if sha(path) != expected:
                raise RuntimeError("validated dependency changed: " + path)
        for key in ("source_root", "source_sidecar", "shard"):
            shared.pop(key)
        shared["pins"] = {p: h for p, h in shared["pins"].items()
                          if p not in (pilot["source_sidecar"], pilot["shard"])
                          and Path(p).name not in ("run.py", "run.sh")}
        for name in ("run.py", "run.sh"):
            shared["pins"][str(campaign / name)] = sha(campaign / name)
        shared.update(status="full_mc_prepared", shifts=list(SHIFTS[1:]),
                      keep_input_cache=False, root_retention="on EOS; never on laptop")
        base = target / str(year)
        shared_path = base / "common.json"
        write(shared_path, shared)
        shared_hash = sha(shared_path)
        source_list = repo / f"autonomous_allhad/workflow/histograms/lepton_veto10_20260908/{year}/inputs.txt"
        sources = [eos(line) for line in source_list.read_text().splitlines()
                   if line and not Path(line).name.startswith("data_")]
        seen, names, jobs, reused, input_failures = set(), set(), [], [], []
        counts = Counter()
        process_files = Counter()

        def inspect(source):
            metadata_path = source.with_suffix(".json")
            metadata = read(metadata_path)
            records = source_records(metadata_path, year, metadata)
            if not records or not source.is_file():
                raise RuntimeError("missing or empty canonical source: " + str(source))
            compact = {key: metadata.get(key) for key in ("events_read", "bad_files")}
            return source, metadata_path, compact, records

        def prepare_source(source, metadata_path, records):
            name = source.stem
            shard_path = base / "shards" / (name + ".json")
            write(shard_path, dict(schema_version="full_production_shard_spec_v2_boosted",
                shard_id=name, record_group="mc", records=records, records_per_shard=len(records),
                record_digest=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]))
            config_path = base / "configs" / (name + ".json")
            write(config_path, dict(shared_config=str(shared_path), shared_config_sha256=shared_hash,
                source_root=str(source), source_sidecar=str(metadata_path), shard=str(shard_path),
                pins={str(metadata_path): sha(metadata_path), str(shard_path): sha(shard_path)}))
            memory = 12000 if any(r["is_signal"] for r in records) else 6000
            rows = []
            for shift in SHIFTS[1:]:
                output = base / "outputs" / name / shift
                output.mkdir(parents=True, exist_ok=True)
                rows.append(f"{name} {shift} {config_path} {output} {memory}")
            return rows

        prepared = []
        with ThreadPoolExecutor(max_workers=4) as pool, ThreadPoolExecutor(max_workers=8) as writers:
            for source, metadata_path, metadata, records in pool.map(inspect, sources):
                name = source.stem
                if name in names:
                    raise RuntimeError("duplicate output shard name: " + name)
                names.add(name)
                for record in records:
                    key = record["file_path"].split("/store/", 1)[-1]
                    if key in seen:
                        raise RuntimeError("duplicate canonical NanoAOD: " + key)
                    seen.add(key)
                    process_files[record["process_group"]] += 1
                counts["signal_shards" if any(r["is_signal"] for r in records) else "background_shards"] += 1
                counts["files"] += len(records)
                counts["events_read"] += metadata["events_read"]
                input_failures.extend(metadata.get("bad_files", []))
                if str(source) == pilot["source_root"]:
                    old_base = campaign / str(year) / ("pilot" if year == 2024 else "pilot_retry1")
                    for shift in SHIFTS[1:]:
                        result = read(old_base / shift / "result.json")
                        if result["status"] != "complete" or any(
                                sha(p["path"]) != p["sha256"] for p in result["products"].values()):
                            raise RuntimeError("validated pilot product changed")
                        reused.append(dict(shard=name, shift=shift, output=str(old_base / shift)))
                    continue
                prepared.append(writers.submit(prepare_source, source, metadata_path, records))
                if len(names) % 500 == 0:
                    print(json.dumps(dict(year=year, inspected_shards=len(names))), flush=True)
            for future in prepared:
                jobs.extend(future.result())
        logs = base / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        queue = base / "jobs.queue"
        queue.write_text("\n".join(jobs) + "\n")
        runtime = repo.parent / "runtime"
        proxy = eos(args.proxy)
        for path in (runtime / "py38.tgz", runtime / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
            if not path.is_file():
                raise FileNotFoundError(path)
        submit_path = base / "jobs.sub"
        submit_path.write_text(f'''universe = vanilla
initialdir = {base}
executable = {campaign}/run.sh
arguments = {campaign}/run.py worker --config $(config) --shift $(shift) --output $(resultdir)
getenv = False
output = {logs}/$(shard)_$(shift).$(ClusterId).out
error = {logs}/$(shard)_$(shift).$(ClusterId).err
log = {logs}/jobs.$(ClusterId).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
transfer_input_files = {runtime}/py38.tgz, {runtime}/mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl
transfer_output_files = ""
use_x509userproxy = True
x509userproxy = {proxy}
request_cpus = 2
request_memory = $(memory)
request_disk = 25000MB
+JobFlavour = "workday"
+MaxRuntime = 28800
+JobBatchName = "NPS26012_shape_{args.label}_{year}"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue shard,shift,config,resultdir,memory from {queue}
''')
        year_state = dict(status="prepared", jobs=len(jobs), reused=reused, counts=dict(counts),
                          process_files=dict(process_files), unique_files=len(seen), duplicate_files=0,
                          source_bad_files=input_failures, input_list_sha256=sha(source_list),
                          common_sha256=shared_hash, queue_sha256=sha(queue), submit=str(submit_path))
        write(base / "campaign_state.json", year_state)
        summary["years"][str(year)] = year_state
        print(json.dumps(dict(year=year, jobs=len(jobs), files=len(seen), reused=len(reused))), flush=True)
    write(target / "campaign_state.json", summary)
    state.setdefault("full_campaigns", {})[args.label] = dict(path=str(target), status="prepared",
        jobs=sum(y["jobs"] for y in summary["years"].values()), shifts=list(SHIFTS[1:]))
    write(campaign / "campaign_state.json", state)


def prepare(args):
    repo = eos(args.repo)
    campaign = repo / "autonomous_allhad/workflow/systematic_propagation"
    canonical_path = repo / "autonomous_allhad/reports/lepton_veto10_canonical_20260908.json"
    canonical = read(canonical_path)
    nominal = repo / "autonomous_allhad/workflow/histograms/lepton_veto10_20260908"
    main_repo = repo / canonical["production"]["frozen_main_code"]
    gnn_repo = repo / canonical["production"]["frozen_gnn_code"]
    all_jobs = []
    for year in args.years:
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
        shard_path = base / args.label / "shard.json"
        write(shard_path, shard)
        model = repo / f"autonomous_allhad/workflow/flat{year}_v8/bundles/model_TopResolved_2024_TROTA2D_ptcut.h5"
        payload_name = "objectcorr_2024_payloads.tgz" if year == 2024 else "objectcorr_2025_data_payloads.tgz"
        object_payload = model.parent / payload_name
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
                      object_payload_bundle=str(object_payload),
                      gnn_manifest=str(nominal / str(year) / "gnn/full_manifest.json"),
                      gnn_model=str(gnn_model / "diagonal_v3_numpy.npz"),
                      gnn_selection=str(gnn_model / "selection.json"),
                      gnn_configuration=str(gnn_repo / "autonomous_allhad/gnn_lowdm/config.json"),
                      stop_xsec=str(gnn_repo / "autonomous_allhad/signals/stop_xsec_13p6TeV.json"),
                      btag_efficiency_sha256="9326454608126467219a60f8d5763a6c92465127ca6ea32ac415ace8ed0ec004",
                      shifts=list(SHIFTS), root_retention="pilot only, on EOS",
                      canonical_report_sha256=sha(canonical_path), pins={})
        paths = [HERE / "run.py", HERE / "run.sh", canonical_path, source_list,
                 source.with_suffix(".json"), shard_path, norm, search, model, object_payload,
                 Path(config["gnn_manifest"]), Path(config["gnn_model"]),
                 Path(config["gnn_selection"]), Path(config["gnn_configuration"]),
                 main_repo / "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"]
        for directory in (repo / "autonomous_allhad/autonomous_allhad",
                          gnn_repo / "autonomous_allhad/gnn_lowdm/_implementation"):
            paths.extend(sorted(directory.glob("*.py")))
        paths.extend([repo / "analysis/data/ids.coffea", repo / "analysis/data/corrections.coffea"])
        for path in paths:
            config["pins"][str(path)] = sha(path)
        config_path = base / ("campaign.json" if args.label == "pilot" else args.label + ".json")
        write(config_path, config)
        for shift in SHIFTS:
            out = base / args.label / shift
            out.mkdir(parents=True, exist_ok=True)
            all_jobs.append(f"{year} {shift} {config_path} {out}")
    logs = campaign / "logs"
    logs.mkdir(exist_ok=True)
    (campaign / (args.label + ".queue")).write_text("\n".join(all_jobs) + "\n")
    runtime = repo.parent / "runtime"
    proxy = Path("/eos/user/t/taiwoo/decaf/analysis/proxy/x509up_u147757")
    for path in (runtime / "py38.tgz", runtime / "mt2-1.2.0-cp38-cp38-manylinux2010_x86_64.whl", proxy):
        if not path.is_file():
            raise FileNotFoundError(path)
    submit = f'''universe = vanilla
initialdir = {campaign}
executable = {campaign}/run.sh
arguments = {campaign}/run.py worker --config $(config) --shift $(shift) --output $(resultdir)
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
queue year,shift,config,resultdir from {campaign}/{args.label}.queue
'''
    (campaign / (args.label + ".sub")).write_text(submit)
    if (campaign / "campaign_state.json").exists():
        state = read(campaign / "campaign_state.json")
        state.setdefault("preparations", {})[args.label] = dict(jobs=len(all_jobs), years=args.years)
    else:
        state = dict(status="pilot_prepared", jobs=len(all_jobs),
          canonical_modified=False, runtime="py38; TROTA-only LCG_104", job_flavour="workday",
          submitted_clusters=[], pending=["pilot execution", "nominal closure", "full MC propagation",
          "JES/JER TROTA-input propagation", "EGM", "MUO/TAU decomposition", "JMS/JMR prescription"])
    write(campaign / "campaign_state.json", state)
    print(json.dumps({"status": "pilot_prepared", "jobs": len(all_jobs), "submit": str(campaign / (args.label + ".sub"))}))


def preflight(args):
    import importlib
    config = read(args.config)
    repo = eos(config["repo"])
    work = eos(args.output)
    payload = prepare_payload(config, work / "payload")
    os.environ.update(AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0",
                      AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="0",
                      AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"))
    sys.path.insert(0, str(repo / "autonomous_allhad"))
    year = config["year"]
    correction = importlib.import_module(f"autonomous_allhad.object_corrections_{year}")
    validation = correction.validate_payloads(payload)
    if validation["status"] != "valid":
        raise RuntimeError(str(validation))
    module = ("autonomous_allhad.intermediate_2024_worker" if year == 2024
              else "autonomous_allhad.intermediate_2025_data_worker")
    importlib.import_module(module).install_backend()
    from autonomous_allhad import flat_ntuple_worker as flat
    record = read(config["shard"])["records"][0]
    rows, summary, bad = flat.process_record(record, payload, 500, "nominal",
                                             "feature_flat_preselection", False, 1, True)
    write(work / "result.json", dict(status="failed" if bad else "complete", summary=summary, bad_files=bad))
    print(json.dumps(dict(rows=len(rows), events_read=summary["events_read"], error=summary.get("error"))))
    if bad:
        raise RuntimeError(str(bad))


def worker(args):
    config = load_config(args.config)
    if args.shift not in SHIFTS:
        raise ValueError("shift not validated by this propagation stage")
    for path, expected in config["pins"].items():
        if sha(path) != expected:
            raise RuntimeError("pinned input changed: " + path)
    destination = eos(args.output)
    if (destination / "result.json").exists():
        existing = read(destination / "result.json")
        if existing.get("status") == "complete":
            if (existing.get("config_sha256") != sha(args.config) or any(
                    sha(item["path"]) != item["sha256"] for item in existing["products"].values())):
                raise RuntimeError("existing completed output changed")
            print(json.dumps({"status": "already_complete", "output": str(destination)}))
            return
    work = Path(os.environ["_CONDOR_SCRATCH_DIR"]) / "shape"
    work.mkdir()
    repo = Path(config["repo"])
    year = config["year"]
    env = dict(os.environ)
    vendor = os.environ.get("PYTHONPATH", "")
    env.update(PYTHONPATH=f"{repo}/autonomous_allhad:{repo}:{vendor}",
               AUTONOMOUS_ALLHAD_LOCAL_ANALYSIS_DATA="0",
               AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE="1",
               AUTONOMOUS_ALLHAD_XRD_KEEP_CACHE="1" if config.get("keep_input_cache", True) else "0",
               AUTONOMOUS_ALLHAD_XRD_CACHE=str(work / "xrd"),
               AUTONOMOUS_ALLHAD_FRAGMENT_DIR=str(work / "fragments"))
    root = work / Path(config["source_root"]).name
    metadata_path = root.with_suffix(".json")
    report = dict(status="running", year=year, shift=args.shift, stages={}, started=time.time())

    def checkpoint():
        write(destination / "result.json", report)

    try:
        checkpoint()
        payload_repo = prepare_payload(config, work / "payload") if config.get("object_payload_bundle") else repo
        module = ("autonomous_allhad.intermediate_2024_worker" if year == 2024
                  else "autonomous_allhad.intermediate_2025_data_worker")
        execute([sys.executable, "-u", "-m", module, "--repo", payload_repo,
                 "--shard", config["shard"], "--output", root, "--metadata-output", metadata_path,
                 "--shift", args.shift, "--record-workers", "1", "--chunk-size", "25000"], env, work)
        metadata = read(metadata_path)
        if metadata["status"] not in ("complete", "complete_with_bad_files"):
            raise RuntimeError("incomplete shifted Events production")
        report["bad_files"] = metadata["bad_files"]
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
        report.update(status="complete_with_bad_files" if report["bad_files"] else "complete",
                      products=products, finished=time.time(),
                      validation="individual shifted job complete; nominal closure pending",
                      config_sha256=sha(args.config))
        checkpoint()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", finished=time.time())
        checkpoint()
        raise


def histogram_leaves(value, path=()):
    if not isinstance(value, dict):
        return {}
    if "sumw" in value and "sumw2" in value:
        return {"/".join(path): value}
    result = {}
    for name, item in value.items():
        result.update(histogram_leaves(item, path + (name,)))
    return result


def compare_histograms(reference, current):
    import numpy as np
    left, right = histogram_leaves(read(reference)), histogram_leaves(read(current))
    if not left or left.keys() != right.keys():
        raise RuntimeError("histogram structure mismatch: " + str(current))
    checked, largest = 0, 0.0
    for path in left:
        for field in ("sumw", "sumw2", "entries"):
            if field not in left[path] and field not in right[path]:
                continue
            a, b = np.asarray(left[path][field]), np.asarray(right[path][field])
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise RuntimeError("invalid histogram array: " + path + "/" + field)
            checked += a.size
            difference = float(np.max(np.abs(a - b))) if a.size else 0.0
            largest = max(largest, difference)
            matches = np.array_equal(a, b) if field == "entries" else np.allclose(a, b, rtol=1e-10, atol=1e-10)
            if not matches:
                raise RuntimeError("nominal histogram mismatch: " + path + "/" + field)
    return dict(status="passed", histogram_leaves=len(left), checked_bin_values=checked,
                maximum_absolute_difference=largest, reference_sha256=sha(reference),
                current_sha256=sha(current))


def tree_identities(root, name):
    import uproot
    names = ["file_id", "entry", "run", "luminosityBlock", "event"]
    if name == "TROTA":
        names.extend(f"TopResolved1pct_sourceJetIdx{i}" for i in range(3))
    with uproot.open(root) as source:
        arrays = source[name].arrays(names, library="np")
        rows = list(zip(*(arrays[field].tolist() for field in names)))
    if len(set(rows)) != len(rows):
        raise RuntimeError("duplicate identities in " + str(root) + ":" + name)
    return set(rows)


def validate(args):
    import math
    import uproot
    config = read(args.config)
    output, reference = eos(args.output), eos(args.reference)
    sys.path.insert(0, str(Path(config["repo"]) / "autonomous_allhad"))
    from autonomous_allhad.flat_ntuple_worker import _root_content_digests
    from autonomous_allhad.trota_resolved_2024_inplace import verify_complete_root

    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, list):
            return all(finite(item) for item in value)
        return True

    campaign = Path(config["repo"]) / "autonomous_allhad/workflow/systematic_propagation"
    events = (campaign / f"logs/pilot.{args.cluster}.log").read_text().split("\n...\n")
    source_metadata = read(config["source_sidecar"])
    report = dict(status="passed", scope="single complete MC-shard pilot", year=config["year"],
                  config_sha256=sha(args.config), validator_sha256=sha(Path(__file__)),
                  reference=str(reference), shifts={})
    identities = {}
    for proc, shift in enumerate(SHIFTS):
        terminations = [event for event in events if event.startswith(f"005 ({args.cluster}.{proc:03d}.000)")]
        if not terminations or "Normal termination (return value 0)" not in terminations[-1]:
            raise RuntimeError(f"successful termination not recorded: {args.cluster}.{proc}")
        directory = output / shift
        result = read(directory / "result.json")
        if result["status"] != "complete" or result["config_sha256"] != sha(args.config):
            raise RuntimeError("incomplete or changed job: " + shift)
        expected_products = {"mc_shard_00000.root", "mc_shard_00000.json", "main.json", "gnn.json", "trota.json", "truth.json"}
        if set(result["products"]) != expected_products:
            raise RuntimeError("missing expected products: " + shift)
        for name, product in result["products"].items():
            path = directory / name
            if str(path) != product["path"] or sha(path) != product["sha256"] or path.stat().st_size != product["bytes"]:
                raise RuntimeError("product integrity mismatch: " + str(path))
            if name.endswith(".json") and not finite(read(path)):
                raise RuntimeError("non-finite JSON value: " + str(path))
        metadata = read(directory / "mc_shard_00000.json")
        if (metadata["status"] != "complete" or metadata["bad_files"]
                or metadata["files_processed"] != source_metadata["files_processed"]
                or metadata["events_read"] != source_metadata["events_read"]):
            raise RuntimeError("input coverage mismatch: " + shift)
        root = directory / "mc_shard_00000.root"
        if metadata["root_sha256"] != sha(root):
            raise RuntimeError("metadata ROOT checksum mismatch: " + shift)
        trota = verify_complete_root(root, target_year=config["year"])
        with uproot.open(root) as source:
            marker = json.loads(str(source["TopWTruth_metadata"]))
            if (marker["status"] != "complete" or marker["application_year"] != config["year"]
                    or source["TopWTruth"].num_entries != source["Events"].num_entries
                    or marker["events_entries"] != source["Events"].num_entries
                    or metadata["events_written"] != source["Events"].num_entries):
                raise RuntimeError("Top/W truth coverage mismatch: " + shift)
        contents = _root_content_digests(root)
        original = {key: value for key, value in contents.items() if key not in ("TopWTruth", "TopWTruth_metadata")}
        if contents["TopWTruth"] != marker["truth_content"] or original != marker["original_contents"]:
            raise RuntimeError("Top/W truth content mismatch: " + shift)
        for kind in ("main", "gnn"):
            histograms = read(directory / (kind + ".json"))
            if histograms["status"] != "complete" or histograms.get("bad_files"):
                raise RuntimeError("incomplete histogram output: " + shift + "/" + kind)
        identities[shift] = tree_identities(root, "Events")
        report["shifts"][shift] = dict(exit_code=0, files=metadata["files_processed"],
            events_read=metadata["events_read"], events_written=metadata["events_written"],
            trota=trota["counts"], root_sha256=sha(root), truth_integrity="passed")
    for shift in SHIFTS:
        report["shifts"][shift].update(entering=len(identities[shift] - identities["nominal"]),
                                      leaving=len(identities["nominal"] - identities[shift]))
    for tree in ("Events", "TROTA", "TopWTruth"):
        if tree_identities(Path(config["source_root"]), tree) != tree_identities(output / "nominal/mc_shard_00000.root", tree):
            raise RuntimeError("nominal identity mismatch: " + tree)
    report["histogram_closure"] = {
        kind: compare_histograms(reference / (kind + ".json"), output / "nominal" / (kind + ".json"))
        for kind in ("main", "gnn")}
    state_path = campaign / "campaign_state.json"
    state = read(state_path)
    validation = state.setdefault("pilot_validation", {})
    validation.setdefault(str(config["year"]), {})["validation_cli"] = report
    if all(validation.get(str(year), {}).get("validation_cli", {}).get("status") == "passed" for year in (2024, 2025)):
        state["status"] = "met_unclustered_pilot_validated"
        state["pending"] = [item for item in state["pending"] if item not in ("pilot execution", "nominal closure")]
    write(state_path, state)
    print(json.dumps(report, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--repo", required=True, type=Path)
    prep.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    prep.add_argument("--label", default="pilot", choices=("pilot", "pilot_retry1"))
    full = commands.add_parser("prepare-full")
    full.add_argument("--repo", required=True, type=Path)
    full.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    full.add_argument("--label", required=True, choices=("full_met_20260911",))
    full.add_argument("--proxy", required=True, type=Path)
    check = commands.add_parser("preflight")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    job = commands.add_parser("worker")
    job.add_argument("--config", required=True, type=Path)
    job.add_argument("--shift", required=True, choices=SHIFTS)
    job.add_argument("--output", required=True, type=Path)
    validation = commands.add_parser("validate")
    validation.add_argument("--config", required=True, type=Path)
    validation.add_argument("--output", required=True, type=Path)
    validation.add_argument("--reference", required=True, type=Path)
    validation.add_argument("--cluster", required=True, type=int)
    args = parser.parse_args()
    return {"prepare": prepare, "prepare-full": prepare_full, "preflight": preflight,
            "worker": worker, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()

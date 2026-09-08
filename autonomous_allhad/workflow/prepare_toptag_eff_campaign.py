#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import gzip
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "toptag_eff_campaign_v1"
DATA_GROUPS = {"JetMET", "EGamma", "Muon", "SingleMuon", "data"}


def eos_path(value: Path) -> Path:
    path = value.absolute()
    if not str(path).startswith('/eos/') or not str(path.resolve()).startswith('/eos/'):
        raise ValueError('Expected an EOS path: ' + str(value))
    return path


def validate_topw_inputs(args) -> int:
    import os
    import sys
    import uproot
    from coffea.util import load

    campaign = eos_path(args.campaign_dir)
    manifest = json.loads((campaign / 'input_manifest.json').read_text())
    # Validate the frozen worker package, not imports from the live checkout.
    package = campaign / 'validation_payload'
    package.mkdir(exist_ok=True)
    with tarfile.open(campaign / 'snapshot/analysis.tgz') as archive:
        for entry in archive.getmembers():
            if entry.issym() or entry.islnk() or entry.name.startswith('/') or '..' in Path(entry.name).parts:
                raise ValueError('Unsafe archive member: ' + entry.name)
        archive.extractall(package)
    os.chdir(package / 'analysis')
    sys.path.insert(0, str(package / 'analysis'))
    os.environ['X509_USER_PROXY'] = str(eos_path(Path(manifest['proxy'])))
    instance = load('data/topwtageff' + manifest['year'] + '.processor')
    by_dataset = {}
    for record in manifest['records']:
        by_dataset.setdefault(record['dataset'], record)

    def read_probe(record):
        with uproot.open(record['file_path'], timeout=120, num_workers=1) as root:
            tree = root['Events']
            missing = sorted(set(instance.branches) - set(tree.keys()))
            if missing:
                raise ValueError('Missing required branches: ' + ', '.join(missing))
            if tree.num_entries == 0:
                raise ValueError('Empty representative file')
            return tree.arrays(instance.branches, entry_stop=100, library='ak'), int(tree.num_entries)

    report = dict(status='running', year=manifest['year'], datasets_expected=len(by_dataset), datasets=[])
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(read_probe, record): record for record in by_dataset.values()}
        for future in as_completed(pending):
            record = pending[future]
            result = dict(dataset=record['dataset'], file_path=record['file_path'])
            try:
                arrays, entries = future.result()
                output = instance.process(arrays)
                result.update(status='passed', entries=entries, tested_entries=len(arrays),
                              histogram_sum={key: value.sum().value for key, value in output.items()})
            except Exception as exc:
                result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc)[:1200])
            report['datasets'].append(result)
            if len(report['datasets']) % 20 == 0 or result['status'] == 'failed':
                print(json.dumps(dict(checked=len(report['datasets']), total=len(by_dataset), latest=result)), flush=True)
            write_json(campaign / 'validation.json', report)
    report['failures'] = sum(item['status'] != 'passed' for item in report['datasets'])
    if report['failures'] == 0:
        import run as runner
        from types import SimpleNamespace

        representative = next((record for record in manifest['records']
                               if record['dataset'].startswith('TTtoLNu2Q')), manifest['records'][0])
        runner.options = SimpleNamespace(processor='topwtageff' + manifest['year'], chunk_size=10000)
        full_report = campaign / 'validation_fullfile.json'
        runner.run_recorded_files(instance, representative['dataset'], [representative['file_path']],
                                  str(campaign / 'validation_fullfile.futures'), str(full_report))
        report['full_file'] = json.loads(full_report.read_text())
        report['failures'] += int(report['full_file']['status'] != 'complete')
    report['status'] = 'passed' if report['failures'] == 0 else 'failed'
    report['completed_at'] = now()
    write_json(campaign / 'validation.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'datasets'}), flush=True)
    return int(report['failures'] != 0)


def prepare_topw(args) -> int:
    repo, campaign, norm_path, runtime, proxy = map(eos_path, (
        args.repo, args.campaign_dir, args.norm, args.py38_archive, args.proxy))
    if (campaign / 'input_manifest.json').exists():
        raise FileExistsError('Campaign already prepared; use its existing manifest: ' + str(campaign))
    norm = json.loads(norm_path.read_text())
    expected = {efficiency_key_from_item(value) for value in norm['dataset_factors'].values()
                if not value['is_data']}
    records, sources = {}, []
    def read_source(source):
        if Path(source['path']).name.startswith('data_shard_'):
            return None
        path = eos_path(Path(source['path']))
        raw = path.read_bytes()
        return path, json.loads(raw), hashlib.sha256(raw).hexdigest()

    def add_source(result):
        if result is None:
            return
        path, payload, digest = result
        sources.append(dict(path=str(path), sha256=digest))
        for item in payload['files']:
            rec = record_from_file(item, path)
            if rec is None or rec['dataset'] not in expected:
                continue
            rec['year'] = args.year
            file_path = rec['file_path']
            if file_path.startswith('/eos/'):
                eos_path(Path(file_path))
            elif not (file_path.startswith('root://') and '/store/' in file_path):
                raise ValueError('Unsupported original MC path: ' + file_path)
            identity = '/store/' + file_path.split('/store/', 1)[1] if '/store/' in file_path else file_path
            previous = records.get(identity)
            if previous and previous['dataset'] != rec['dataset']:
                raise ValueError('File assigned to different datasets: ' + identity)
            records[identity] = rec
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(norm['source_sidecars']), 32):
            for result in pool.map(read_source, norm['source_sidecars'][start:start + 32]):
                add_source(result)
            if start % 320 == 0:
                print('Sidecars {}, unique MC files {}'.format(start, len(records)), flush=True)
    records = sorted(records.values(), key=lambda item: (item['dataset'], item['file_path']))
    by_dataset = {}
    for rec in records:
        by_dataset.setdefault(rec['dataset'], []).append(rec)
    missing = expected - set(by_dataset)
    if missing:
        raise ValueError('Current MC datasets not covered: ' + ', '.join(sorted(missing)))
    for name in ('shards', 'outputs', 'reports', 'logs', 'snapshot'):
        (campaign / name).mkdir(parents=True, exist_ok=True)
    processor = 'topwtageff' + args.year
    files = ['analysis/run.py', 'analysis/processors/btageff.py', 'analysis/libs/mycoffea.py',
             'analysis/data/' + processor + '.processor',
             'analysis/data/JMESF/' + args.year + '/fatJet_jerc.json.gz']
    provenance = {file: hashlib.sha256((repo / file).read_bytes()).hexdigest() for file in files}
    for file in ('analysis/utils/corrections.py', 'analysis/data/corrections.coffea',
                 'analysis/data/JMESF/2024/jetid.json.gz'):
        provenance[file] = hashlib.sha256((repo / file).read_bytes()).hexdigest()
    archive = campaign / 'snapshot' / 'analysis.tgz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for file in files:
            bundle.add(repo / file, arcname=file)
    wrapper = campaign / 'snapshot' / 'run_toptag_eff_worker.sh'
    shutil.copy2(repo / 'autonomous_allhad/workflow/run_toptag_eff_worker.sh', wrapper)
    rows, jobs, aliases = [], [], {}
    for dataset, dataset_records in sorted(by_dataset.items()):
        key = dataset
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,190}', key) or '--' in key:
            primary = dataset.strip('/').split('/')[0]
            key = re.sub(r'[^A-Za-z0-9_]', '_', primary)[:100] + '_' + hashlib.sha256(dataset.encode()).hexdigest()[:16]
        if key in aliases:
            raise ValueError('Dataset filename collision: ' + key)
        aliases[key] = dataset
        for start in range(0, len(dataset_records), args.files_per_shard):
            subset = dataset_records[start:start + args.files_per_shard]
            name = key + '____' + str(start // args.files_per_shard) + '_'
            shard = campaign / 'shards' / (name + '.json.gz')
            with gzip.open(shard, 'wt') as stream:
                json.dump({dataset: {'files': [rec['file_path'] for rec in subset]}}, stream)
            rows.append('{} {}'.format(name, shard))
            jobs.append(dict(name=name, dataset=dataset, files=len(subset), metadata=str(shard)))
    write_json(campaign / 'dataset_aliases.json', aliases)
    arguments = campaign / 'arguments.txt'
    arguments.write_text('\n'.join(rows) + '\n')
    submit = f'''universe = vanilla
initialdir = {campaign}
executable = {wrapper}
arguments = --topw {processor} $(name).json.gz
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {runtime}, {proxy}, {archive}, $(shard)
transfer_output_files = out.futures, report.json
transfer_output_remaps = "out.futures={campaign}/outputs/$(name).futures; report.json={campaign}/reports/$(name).json"
output = {campaign}/logs/$(name).out
error = {campaign}/logs/$(name).err
log = {campaign}/logs/campaign.log
request_cpus = 1
request_memory = 3000MB
request_disk = 8000MB
queue name,shard from {arguments}
'''
    if '/tmp' in submit or '/afs' in submit or 'JobFlavour' in submit or 'MaxRuntime' in submit:
        raise ValueError('Invalid submit policy')
    (campaign / 'topwtageff.sub').write_text(submit)
    summary = dict(schema_version='topw_btag_entry_campaign_v1', created_at=now(), status='prepared',
                   year=args.year, files=len(records), datasets=len(by_dataset), jobs=len(jobs),
                   dataset_file_counts={key: len(value) for key, value in by_dataset.items()},
                   norm=str(norm_path), norm_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                   runtime=str(runtime), proxy=str(proxy), source_sha256=provenance,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    physics = dict(
        event_selection='None: full original MC files, no HLT or analysis-region selection',
        jet_selection='AK8 tight lepton veto ID; corrected pT > 200 GeV; abs(eta) < 2.0',
        nominal_jec='Existing compiled get_fjec_correction, with the analysis year; no extra JER',
        jet_id_year='2024 for both years, matching the current analysis',
        working_points={'GlobalParT3_Top': 0.5078, 'GlobalParT3_W': 0.9385},
        analysis_denominator={'Top': 'pT > 400 GeV, mSD > 105 GeV', 'W': '60 < mSD < 105 GeV'},
        score_only_denominator='50 < mSD < 220 GeV; retained separately, not an analysis-selection replacement',
        flavor={'0': 'other', '1': 'partial hadronic W', '2': 'partial hadronic top',
                '3': 'merged hadronic W without contained top b', '4': 'merged hadronic top (bqq)'},
        matching='Direct W/top decay partons, deltaR < 0.8, same-parent ancestry with generator-copy traversal',
        weights='Unweighted counts and separate full signed genWeight sums; sumw2 for each',
        efficiency='pass / (pass + fail), retaining per-dataset histograms; no SF applied',
    )
    write_json(campaign / 'input_manifest.json', dict(summary, records=records, jobs_manifest=jobs, sources=sources, physics=physics))
    write_json(campaign / 'summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def efficiency_key_from_item(item: dict[str, Any]) -> str:
    key = str(item.get("dataset") or item.get("dataset_key") or "unknown")
    return re.sub(r"____\d+_$", "", key)


def record_from_file(item: dict[str, Any], source: Path) -> dict[str, Any] | None:
    process = str(item.get("process") or item.get("process_group") or "unknown")
    dataset = efficiency_key_from_item(item)
    file_path = str(item.get("file_path") or item.get("physical_file_path") or "")
    if not file_path or process in DATA_GROUPS:
        return None
    if item.get("is_data"):
        return None
    if item.get("read_status") not in (None, "success"):
        return None
    return {
        "dataset": dataset,
        "sample_name": dataset,
        "file_path": file_path,
        "process_group": process,
        "year": str(item.get("year") or "2024"),
        "is_data": False,
        "is_background": bool(item.get("is_background", process != "SMS")),
        "is_signal": bool(item.get("is_signal", process == "SMS")),
        "xsec_pb": item.get("xsec_pb"),
        "number_of_entries": item.get("number_of_entries"),
        "source_metadata": str(source),
    }


def collect_records(metadata_dirs: list[Path], include_signals: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, str]] = []
    metadata_files = 0
    metadata_bad = 0
    for directory in metadata_dirs:
        for path in sorted(directory.glob("mc_shard_*.json")):
            metadata_files += 1
            try:
                payload = json.loads(path.read_text())
            except Exception:
                metadata_bad += 1
                continue
            for item in payload.get("files", []):
                rec = record_from_file(item, path)
                if rec is None or (rec["is_signal"] and not include_signals):
                    continue
                old = by_path.get(rec["file_path"])
                if old and old["process_group"] != rec["process_group"]:
                    conflicts.append(
                        {
                            "file_path": rec["file_path"],
                            "first_process": old["process_group"],
                            "second_process": rec["process_group"],
                        }
                    )
                    continue
                by_path.setdefault(rec["file_path"], rec)
    records = sorted(by_path.values(), key=lambda x: (x["dataset"], x["file_path"]))
    return records, {
        "metadata_files_scanned": metadata_files,
        "metadata_files_unreadable": metadata_bad,
        "deduplication_conflicts": conflicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Top-tag efficiency campaign from flat-output metadata")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, action="append")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--files-per-shard", type=int, default=25)
    parser.add_argument("--include-signals", action="store_true")
    parser.add_argument('--tagger', choices=('top', 'topw'), default='top')
    parser.add_argument('--year', choices=('2024', '2025'))
    parser.add_argument('--norm', type=Path)
    parser.add_argument('--py38-archive', type=Path)
    parser.add_argument('--proxy', type=Path)
    parser.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args()
    if args.tagger == 'topw':
        if args.validate_inputs:
            return validate_topw_inputs(args)
        if not all((args.year, args.norm, args.py38_archive, args.proxy)):
            parser.error('--tagger topw requires --year, --norm, --py38-archive and --proxy')
        return prepare_topw(args)
    if not args.metadata_dir:
        parser.error('--metadata-dir is required for the legacy top mode')

    repo = args.repo.absolute()
    campaign = args.campaign_dir.absolute()
    shards_dir = campaign / "shards"
    outputs_dir = campaign / "outputs"
    logs_dir = campaign / "logs"
    for directory in (campaign, shards_dir, outputs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records, audit = collect_records(
        [path.absolute() for path in args.metadata_dir],
        include_signals=args.include_signals,
    )
    if not records:
        raise RuntimeError("no MC file records were found")

    by_process: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    for rec in records:
        by_process[rec["process_group"]] = by_process.get(rec["process_group"], 0) + 1
        by_dataset[rec["dataset"]] = by_dataset.get(rec["dataset"], 0) + 1

    shard_paths: list[Path] = []
    digest = hashlib.sha256("\n".join(x["file_path"] for x in records).encode()).hexdigest()
    for index, start in enumerate(range(0, len(records), args.files_per_shard)):
        subset = records[start : start + args.files_per_shard]
        path = shards_dir / f"toptageff_shard_{index:05d}.json"
        write_json(
            path,
            {
                "schema_version": "toptag_eff_input_shard_v1",
                "shard_id": path.stem,
                "created_at": now(),
                "record_start": start,
                "record_stop": start + len(subset),
                "records": subset,
            },
        )
        shard_paths.append(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "status": "prepared",
        "repo": str(repo),
        "metadata_dirs": [str(path.absolute()) for path in args.metadata_dir],
        "records": len(records),
        "files_per_shard": args.files_per_shard,
        "shards": len(shard_paths),
        "process_file_counts": dict(sorted(by_process.items())),
        "dataset_file_counts": dict(sorted(by_dataset.items())),
        "efficiency_grouping_policy": "dataset key / generator-bin key, matching btageff*.merged",
        "input_digest": digest,
        "audit": audit,
    }
    write_json(campaign / "input_manifest.json", {**manifest, "files": records})
    write_json(campaign / "summary.json", manifest)

    arguments = []
    for path in shard_paths:
        name = path.stem
        arguments.append(
            " ".join(
                [
                    name,
                    str(path),
                    str(outputs_dir / f"{name}.npz"),
                    str(outputs_dir / f"{name}.json"),
                ]
            )
        )
    arguments_path = campaign / "arguments.txt"
    arguments_path.write_text("\n".join(arguments) + "\n")

    wrapper = repo / "autonomous_allhad" / "workflow" / "run_toptag_eff_worker.sh"
    proxy = repo / "analysis" / "proxy" / "x509up_u147757"
    pyenv = repo / "condor" / "py38.tgz"
    submit = f"""universe = vanilla
executable = {wrapper}
arguments = $(name) $(shard) $(npz_dest) $(json_dest)
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {pyenv}, {proxy}
transfer_output_files = $(name).npz, $(name).json
transfer_output_remaps = "$(name).npz=$(npz_dest); $(name).json=$(json_dest)"
output = {logs_dir}/$(name).out
error = {logs_dir}/$(name).err
log = {logs_dir}/campaign.log
request_cpus = 1
request_memory = 3000MB
request_disk = 8000MB
+JobFlavour = "workday"
queue name,shard,npz_dest,json_dest from {arguments_path}
"""
    (campaign / "toptageff.sub").write_text(submit)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

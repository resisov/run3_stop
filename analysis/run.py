#!/usr/bin/env python
import pickle
import json
import time
import gzip
import os
import hashlib
from datetime import datetime, timezone
from optparse import OptionParser

import uproot
#uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource

import numpy as np
from coffea import processor
from coffea.util import load, save
from libs.mycoffea import CustomNanoAODSchema, AK15SubJet, AK15Jet

import warnings
warnings.filterwarnings("ignore")


def run_recorded_files(processor_instance, dataset, files, output_path, report_path):
    """Commit whole files only; retain read failures for targeted recovery."""
    report = dict(dataset=dataset, files=[], status='running', files_expected=len(files),
                  files_processed=0, files_failed=0, files_incomplete=0,
                  processor_sha256=hashlib.sha256(open('data/' + options.processor + '.processor', 'rb').read()).hexdigest())
    output = processor_instance.make_output()
    for path in (output_path, report_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def checkpoint():
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = report_path + '.partial'
        with open(temporary, 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temporary, report_path)

    checkpoint()
    try:
        for file_path in files:
            record = dict(file_path=file_path, status='failed', attempts=[])
            candidates = [file_path]
            if file_path.startswith('root://') and '/store/' in file_path:
                lfn = '/store/' + file_path.split('/store/', 1)[1]
                candidates.extend('root://' + host + '/' + lfn for host in
                                  ('cms-xrd-global.cern.ch', 'xrootd-cms.infn.it'))
            for candidate in dict.fromkeys(candidates):
                current = processor_instance.make_output()
                read = 0
                stage = 'open'
                try:
                    with uproot.open(candidate, timeout=120, num_workers=1) as root:
                        tree = root['Events']
                        stage = 'schema'
                        missing = sorted(set(processor_instance.branches) - set(tree.keys()))
                        if missing:
                            raise ValueError('Missing required branches: ' + ', '.join(missing))
                        expected = int(tree.num_entries)
                        iterator = iter(tree.iterate(processor_instance.branches, step_size=options.chunk_size, library='ak'))
                        while True:
                            stage = 'read'
                            try:
                                arrays = next(iterator)
                            except StopIteration:
                                break
                            stage = 'process'
                            chunk = processor_instance.process(arrays)
                            for key in current:
                                current[key] += chunk[key]
                            read += len(arrays)
                        if read != expected:
                            raise OSError('Incomplete file: {} / {} entries'.format(read, expected))
                    for key in output:
                        output[key] += current[key]
                    record.update(status='complete', events_read=read, expected_entries=expected,
                                  access_path=candidate)
                    report['files_processed'] += 1
                    break
                except Exception as exc:
                    record['attempts'].append(dict(path=candidate, stage=stage,
                                                    error=type(exc).__name__ + ': ' + str(exc)[:1000]))
                    if stage in ('schema', 'process'):
                        report.update(status='common_error', error=record['attempts'][-1])
                        report['files'].append(record)
                        raise
            if record['status'] != 'complete':
                report['files_failed'] += 1
            report['files'].append(record)
            print(json.dumps({key: record.get(key) for key in ('file_path', 'status', 'events_read')}), flush=True)
            checkpoint()
        for value in output.values():
            if not np.all(np.isfinite(value.values(flow=True))) or not np.all(np.isfinite(value.variances(flow=True))):
                raise ValueError('Non-finite output histogram')
        save(output, output_path + '.partial')
        checked = load(output_path + '.partial')
        if set(checked) != set(output):
            raise ValueError('Output round-trip failed')
        os.replace(output_path + '.partial', output_path)
        report.update(status='complete' if report['files_failed'] == 0 else 'partial',
                      output_sha256=hashlib.sha256(open(output_path, 'rb').read()).hexdigest(),
                      events_read=sum(item.get('events_read', 0) for item in report['files']))
    finally:
        report['files_incomplete'] = report['files_expected'] - report['files_processed'] - report['files_failed']
        checkpoint()


def run(processor_instance, samplefiles):
    fileslice = slice(None) if options.max_files is None else slice(options.max_files)
    for dataset, info in samplefiles.items():
        filelist = {}
        if options.dataset:
            if not any(_dataset in dataset for _dataset in options.dataset.split(',')): continue
        print('Processing:',dataset)
        files = []
        for file in info['files'][fileslice]:
            files.append(file)
        filelist[dataset] = files
        if options.file_report:
            if len(samplefiles) != 1 or not options.output:
                raise ValueError('--file-report requires one dataset and --output')
            run_recorded_files(processor_instance, dataset, files, options.output, options.file_report)
            continue
    
        tstart = time.time()
        output = processor.run_uproot_job(filelist,
                                          'Events',
                                          processor_instance=processor_instance,
                                          executor=processor.futures_executor,
                                          executor_args={'schema': CustomNanoAODSchema,
                                                         'workers': options.workers,
                                                         'skipbadfiles': True},
                                          ) 
        
        output_path = (
            options.output
            if options.output
            else 'hists/'+options.processor+'/'+dataset+'.futures'
        )
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        save(output, output_path)
        dt = time.time() - tstart
        nworkers = options.workers
        print("%.2f us*cpu overall" % (1e6*dt*nworkers, ))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--processor', help='processor', dest='processor')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
    parser.add_option('-w', '--workers', help='Number of workers to use for multi-worker executors (e.g. futures or condor)', dest='workers', type=int, default=8)
    parser.add_option('--max-files', help='Maximum number of files to process per selected dataset', dest='max_files', type=int, default=None)
    parser.add_option('--metadata-path', help='Explicit gzipped metadata path', dest='metadata_path')
    parser.add_option('--output', help='Explicit output path', dest='output')
    parser.add_option('--file-report', help='Per-file accounting JSON; atomic whole-file processing', dest='file_report')
    parser.add_option('--chunk-size', type=int, default=10000)
    (options, args) = parser.parse_args()
    
    processor_instance=load('data/'+options.processor+'.processor')
    metadata_path = (
        options.metadata_path
        if options.metadata_path
        else "metadata/"+options.metadata+".json.gz"
    )
    with gzip.open(metadata_path) as fin:
        samplefiles = json.load(fin)
    run(processor_instance, samplefiles)

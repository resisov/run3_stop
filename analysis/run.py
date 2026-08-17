#!/usr/bin/env python
import pickle
import json
import time
import gzip
import os
from optparse import OptionParser

import uproot
#uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource

import numpy as np
from coffea import processor
from coffea.util import load, save
from libs.mycoffea import CustomNanoAODSchema, AK15SubJet, AK15Jet

import warnings
warnings.filterwarnings("ignore")

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

"""Real-file gates for CMS-p4 central, weights and re-inferred JES/JER histograms."""
import argparse
import copy
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.nominal_campaign import prepare, current_contract, validate_payload, fingerprint

FOLDER = ROOT/'validation/systematic_histograms_20260924'


def prepare_representatives():
    source = read_json(ROOT/'campaigns/nominal_20260923/plan.json')
    chosen = []
    for year in (2024,2025):
        keys = {f'{year}_mc_shard_00000',f'{year}_mc_shard_03059',
                f'{year}_signal_shard_00568',f'{year}_signal_shard_00077',f'{year}_signal_shard_00294'}
        data = min((t for t in source['tasks'] if t['year']==year and '/data_' in t['input'] and t['entries']>100),
                   key=lambda t:t['entries'])
        keys.add(data['key'])
        for task in source['tasks']:
            if task['key'] in keys:
                t=copy.deepcopy(task);t.pop('seed_output',None);chosen.append(t)
        if len([t for t in chosen if t['year']==year])!=len(keys):
            raise ValueError('Representative input unavailable')
    manifest=dict(scope='representative_validation',shape_mode='cms_trota_jme',tasks=chosen)
    write_json(FOLDER/'inputs.json',manifest)
    return prepare(FOLDER/'inputs.json',FOLDER/'campaign',workers=12,chunk_size=250)


def finalize():
    from TROTASR.utils.signal_models import signal_topology
    plan=read_json(FOLDER/'campaign/plan.json')
    state=read_json(FOLDER/'campaign/state.json')
    if state['status']!='histograms_complete' or state['completed']!=len(plan['tasks']):
        raise ValueError('Representative histograms are not complete')
    records=[]; topologies=set(); modes=set(); regions=set(); variations=set()
    for task in plan['tasks']:
        receipt=read_json(FOLDER/'campaign/state'/(task['key']+'.json'))
        path=ROOT/receipt['output']
        if sha256(path)!=receipt['output_sha256'] or fingerprint(ROOT/task['input'])!=task['fingerprint']:
            raise ValueError('Changed validation input/output')
        payload=validate_payload(read_json(path),task,current_contract(task['year'],'cms_trota_jme'))
        actual=set(payload['audit']['trota_inference'])
        if '/data_' in task['input']:
            if actual!={'nominal'}:raise ValueError('Data systematic endpoints generated')
        elif task['entries'] and actual!={'nominal','jesTotalUp','jesTotalDown','jerUp','jerDown'}:
            raise ValueError('Incomplete physical shape evaluation')
        for d in payload['input_metadata'].values():
            if d.get('is_signal'):
                topologies.add(signal_topology(d['dataset']))
        for mode,hist in payload['histograms'].items():
            modes.add(mode);variations.update(hist)
            for v in hist.values():regions.update(v)
        records.append(dict(key=task['key'],entries=task['entries'],seconds=payload['seconds'],
            output=receipt['output'],sha256=receipt['output_sha256'],bytes=path.stat().st_size,
            inference=payload['audit']['trota_inference']))
    if topologies!={'T2tt','T2bW','T2tb'} or modes!={'highdm','lowdm'}:
        raise ValueError('Incomplete topology/regime validation')
    if not {'nominal','jesTotalUp','jesTotalDown','jerUp','jerDown'}<=variations:
        raise ValueError('Missing actual shape histograms')
    closure=read_json(FOLDER/'neutral_closure.json')
    if closure.get('status')!='passed' or closure.get('code')!=current_contract(2024,'cms_trota_jme')['code']:
        raise ValueError('Neutral closure not validated for current code')
    gate=dict(status='passed',histogram_shape_validation=True,three_signal_topologies_tested=True,
        contracts={str(y):current_contract(y,'cms_trota_jme') for y in (2024,2025)},
        records=records,regions=sorted(regions),variations=sorted(variations),
        neutral_closure='validation/systematic_histograms_20260924/neutral_closure.json',
        scope='central_weights_JES_Total_JER_only_not_complete_systematics_or_fits')
    write_json(FOLDER/'validation_gate.json',gate)
    source=read_json(ROOT/'campaigns/nominal_20260923/plan.json')
    tasks=copy.deepcopy(source['tasks'])
    seeds={r['key']:r['output'] for r in records}
    for t in tasks:
        t.pop('seed_output',None)
        if t['key'] in seeds:t['seed_output']=seeds[t['key']]
    full=dict(scope='full_cms_trota_jme_production',shape_mode='cms_trota_jme',tasks=tasks,
        expected_files_by_year={str(y):sum(t['year']==y for t in tasks) for y in (2024,2025)},
        integrated_input_inventory_complete=True,
        validation_gate='validation/systematic_histograms_20260924/validation_gate.json')
    write_json(FOLDER/'production_inputs.json',full)
    return gate


def neutral_closure():
    """Fresh independent stored-kinematics vs rebuilt CMS central selections/GNN."""
    import awkward as ak
    import numpy as np
    import uproot
    from TROTASR.utils.trota_inference import load_models,p4_view,infer_chunk,P4_FIELDS,bounded_slices
    from TROTASR.utils.systematic_views import anchored_endpoints
    from TROTASR.utils.shape_kinematics import varied_view
    from TROTASR.utils.event_selections import select,REGIONS
    from TROTASR.utils.reader import resolved_candidates
    from TROTASR.gnn4lowdm.inference import Rank005Numpy
    from TROTASR.gnn4lowdm.region_features import feature_arrays
    models=load_models(2024);gnn=Rank005Numpy(ROOT/'gnn4lowdm/diagonal_v3_numpy.npz')
    records=[]
    for year in (2024,2025):
        path=ROOT/f'inputs/{year}/mc_shard_00000.root'
        with uproot.open(path) as f:
            old_resolved=resolved_candidates(f)
            for a in f['Events'].iterate(step_size=250,library='ak'):
                values=anchored_endpoints(a,year)['nominal']
                view=p4_view(a,{n:values[n.replace('nanoaod','corrected')] for n in P4_FIELDS})
                for sl in bounded_slices(view):
                    old=a[sl]
                    inferred,resolved,_,_=infer_chunk(view[sl],models)
                    for n in P4_FIELDS:inferred=ak.with_field(inferred,old[n],n)
                    rebuilt=varied_view(inferred,{n:v[sl] for n,v in values.items()})
                    direct, _=select(ak.Array(inferred),resolved,dy_mass_window=None)
                    neutral,_=select(ak.Array(rebuilt),resolved,dy_mass_window=None)
                    historical,_=select(ak.Array(old),old_resolved,dy_mass_window=None)
                    record=dict(year=year,events=len(old),regions={})
                    for region in REGIONS:
                        d,n=direct[region],neutral[region]
                        for key in ('high','high_search','low','counts'):
                            if not np.array_equal(d[key],n[key]):
                                raise ValueError('Neutral selection mismatch: '+region+'/'+key)
                        selected=n['low']
                        if np.any(selected):
                            fd=feature_arrays(inferred,d['block'],region,selected)
                            fn=feature_arrays(rebuilt,n['block'],region,selected)
                            sd=gnn.predict(*fd[:5],batch_size=128);sn=gnn.predict(*fn[:5],batch_size=128)
                            if not np.allclose(sd,sn,rtol=0,atol=1e-6):
                                raise ValueError('Neutral GNN mismatch: '+region)
                            edges=np.asarray(read_json(ROOT/'gnn4lowdm/config.json')['cr_binning']['score_edges'])
                            if not np.array_equal(np.searchsorted(edges,sd),np.searchsorted(edges,sn)):
                                raise ValueError('Neutral GNN-bin mismatch')
                        h=historical[region]
                        record['regions'][region]={k:dict(before=int(h[k].sum()),after=int(n[k].sum()),
                            gained=int((n[k]&~h[k]).sum()),lost=int((h[k]&~n[k]).sum())) for k in ('high_search','low')}
                    records.append(record)
    out=dict(status='passed',code=current_contract(2024,'cms_trota_jme')['code'],records=records,
             description='Neutral rebuilt selection/counts/GNN closure; historical central difference recorded separately')
    write_json(FOLDER/'neutral_closure.json',out)
    return out


def validate_nonjme():
    """Real files spanning all regions, with unchanged stored-TROTA weights."""
    import subprocess, sys, unittest, resource
    from concurrent.futures import ThreadPoolExecutor
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    from TROTASR.utils.signal_models import signal_topology
    folder=ROOT/'validation'
    tasks=read_json(folder/'inputs.json')
    suite=unittest.TestSuite()
    for name in ('test_nonjme_systematics','test_systematic_views','test_nominal_campaign',
                 'test_campaign_resources','test_systematic_continuation'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('TROTASR.tests.'+name))
    test=unittest.TextTestRunner().run(suite)
    if not test.wasSuccessful():raise ValueError('Non-JME regression failure')
    def work(t):
        outputs={}
        for label,extra,variation in [('weights',[],'all_weights'),('endpoints',['--stored-trota-nonjme'],'nominal')]:
            path=folder/label/(t['key']+'.json.gz')
            expected=current_contract(t['year'],'stored_trota_nonjme' if label=='endpoints' else None,[variation])
            if path.exists():
                validate_payload(read_json(path),t,expected)
                outputs[label]=path
                continue
            cmd=[sys.executable,str(ROOT/'workflows/build_systematics.py'),'--input',str(ROOT/t['input']),
                 '--year',str(t['year']),'--output',str(path),'--variation',variation,'--chunk-size','500',*extra]
            log=folder/(t['key']+'.'+label+'.log')
            with log.open('w') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
            outputs[label]=path
        return t,outputs
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(work,tasks))
    records=[];topologies=set();regions=set();total_response={}
    for t,paths in results:
        old=read_json(folder/'baseline'/(t['key']+'.json.gz'))
        weights,shape=read_json(paths['weights']),read_json(paths['endpoints'])
        for section in ('histograms','physical_histograms','dy_rz'):
            if old[section]!=weights[section]:raise ValueError('Weight baseline changed: '+t['key']+'/'+section)
        for mode in ('highdm','lowdm'):
            for section in ('histograms','physical_histograms'):
                if old[section].get(mode,{}).get('nominal',{})!=shape[section].get(mode,{}).get('nominal',{}):
                    raise ValueError('Shape nominal changed: '+t['key']+'/'+mode+'/'+section)
        if old['dy_rz']!=shape['dy_rz']:raise ValueError('DY nominal changed')
        if 'trota_inference' in shape['audit']:raise ValueError('Non-JME must not infer TROTA')
        endpoints=shape['audit']['object_endpoints']
        if set(endpoints)!=set(NONJME_ENDPOINTS) or any(e['events']!=t['entries'] for e in endpoints.values()):
            raise ValueError('Incomplete endpoint traversal')
        for d in shape['input_metadata'].values():
            if d.get('is_signal'):topologies.add(signal_topology(d['dataset']))
        for mode,hists in shape['histograms'].items():
            for v,rs in hists.items():
                regions.update(rs)
                if v!='nominal':
                    total_response[v]=total_response.get(v,False) or rs!=hists.get('nominal',{})
        records.append(dict(key=t['key'],year=t['year'],entries=t['entries'],input=t['input'],
            output=str(paths['endpoints'].relative_to(ROOT)),sha256=sha256(paths['endpoints']),
            seconds=shape['seconds'],all_weight_histograms_equal=True,nominal_histograms_equal=True,
            endpoints=endpoints,acceptance=shape['audit'].get('object_acceptance',{}),
            response=shape['audit'].get('object_response',{})))
        print('validated',t['key'],flush=True)
    if topologies!={'T2tt','T2bW','T2tb'} or regions!={'SR','LLCR','QCDCR','GCR','DY2E','DY2M'}:
        raise ValueError('Incomplete topology/region coverage: '+str((sorted(topologies),sorted(regions))))
    gate=dict(status='passed',stored_trota_nonjme_validation=True,three_signal_topologies_tested=True,
        unit_tests=test.testsRun,records=records,regions=sorted(regions),endpoint_histogram_response=total_response,
        contracts={str(y):current_contract(y,'stored_trota_nonjme',['nominal']) for y in (2024,2025)},
        baseline_weight_contracts=read_json(folder/'baseline_contracts.json'),
        tested_weight_contracts={str(y):current_contract(y,variations=['all_weights']) for y in (2024,2025)},
        source_root=str(ROOT),full_production_complete=False)
    write_json(folder/'validation_gate.json',gate)
    return gate


def validate_object_response():
    """Audit-key-only fix: mixed datasets plus immutable old success closure."""
    import subprocess,sys,unittest
    from concurrent.futures import ThreadPoolExecutor
    folder=ROOT/'validation';tasks=read_json(folder/'inputs.json')
    suite=unittest.TestSuite()
    for name in ('test_nonjme_systematics','test_systematic_views','test_nominal_campaign',
                 'test_campaign_resources','test_systematic_continuation'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('TROTASR.tests.'+name))
    test=unittest.TextTestRunner().run(suite)
    if not test.wasSuccessful():raise ValueError('Regression failure')
    def work(t):
        outputs={}
        for label,extra,variation in [('weights',[],'all_weights'),('endpoints',['--stored-trota-nonjme'],'nominal')]:
            path=folder/label/(t['key']+'.json.gz')
            expected=current_contract(t['year'],'stored_trota_nonjme' if label=='endpoints' else None,[variation])
            if not path.exists():
                with (folder/(t['key']+'.'+label+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(ROOT/'workflows/build_systematics.py'),
                        '--input',str(ROOT/t['input']),'--year',str(t['year']),'--output',str(path),
                        '--variation',variation,'--chunk-size','500',*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
            outputs[label]=validate_payload(read_json(path),t,expected)
        return t,outputs
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(work,tasks))
    records=[];mixed=set();old_success=set()
    for t,outputs in results:
        old=read_json(folder/'reference/weights'/(t['key']+'.json.gz'))
        for section in ('histograms','physical_histograms','dy_rz'):
            if old[section]!=outputs['weights'][section]:raise ValueError('Weight histograms changed')
        shape=outputs['endpoints']
        for section in ('histograms','physical_histograms'):
            for mode in ('highdm','lowdm'):
                if old[section].get(mode,{}).get('nominal',{})!=shape[section].get(mode,{}).get('nominal',{}):
                    raise ValueError('Shape nominal changed')
        baseline=folder/'reference/endpoints'/(t['key']+'.json.gz')
        if baseline.exists():
            previous=read_json(baseline)
            for section in ('histograms','physical_histograms','dy_rz','dy_rz_variations'):
                if previous.get(section)!=shape.get(section):raise ValueError('Previous valid histogram changed')
            old_success.add(t['year'])
        if len(shape['input_metadata'])>1:mixed.add(t['year'])
        output=folder/'endpoints'/(t['key']+'.json.gz')
        records.append(dict(key=t['key'],year=t['year'],entries=t['entries'],datasets=len(shape['input_metadata']),
            output=str(output.relative_to(ROOT)),sha256=sha256(output),seconds=shape['seconds']))
    if mixed!={2024,2025} or old_success!={2024,2025}:raise ValueError('Incomplete mixed-dataset/year closure')
    out=dict(status='passed',unit_tests=test.testsRun,multi_dataset_files_passed=True,
        old_success_histograms_equal=True,weight_histograms_equal=True,records=records,
        before_contracts=read_json(folder/'before_contracts.json'),
        after_contracts={m:{str(y):current_contract(y,None if m=='weights' else m,
             ['all_weights'] if m=='weights' else ['nominal']) for y in (2024,2025)}
             for m in ('weights','stored_trota_nonjme')},
        changed_scientific_source='utils/histogramming.py',physics_change=False,
        previous_successful_histograms_recomputed_in_production=False)
    write_json(folder/'object_response_validation.json',out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','neutral','finalize','nonjme','object-response'))
    a=p.parse_args()
    result={'prepare':prepare_representatives,'neutral':neutral_closure,'finalize':finalize,
            'nonjme':validate_nonjme,'object-response':validate_object_response}[a.action]()
    print(result.get('status','prepared'),flush=True)

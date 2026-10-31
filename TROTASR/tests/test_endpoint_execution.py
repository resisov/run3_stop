import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from TROTASR.utils.corrections import NONJME_ENDPOINTS
from TROTASR.workflows.nominal_campaign import execution_tasks, task_contract, input_progress
from TROTASR.workflows.merge_nominal import assemble_endpoints


def plan():
    return dict(shape_mode='stored_trota_nonjme',variations=['nominal'],
        tasks=[dict(key='old',year=2024,entries=2),dict(key='new',year=2024,entries=7)],
        endpoint_execution_update=dict(split_input_keys=['new']),
        contracts={'2024':dict(object_endpoints=list(NONJME_ENDPOINTS),code={'x':'hash'})})


def payload(endpoint):
    leaf=dict(sumw=[2.],sumw2=[4.],entries=[1],edges=[0.,1.])
    return dict(contract=dict(object_task_endpoint=endpoint,input='same',fingerprint={'size':3},year=2024),
        histograms={'highdm':{endpoint:{'x':leaf}}},physical_histograms={},
        input_metadata={'a':dict(sumw=5.,events_read=7)},
        dy_rz={'value':1} if endpoint=='nominal' else {},dy_rz_variations={},
        audit=dict(events_read=7,migrations={'x':1} if endpoint=='nominal' else {},
            chunks=[{'endpoint':endpoint}],weights=[{'endpoint':endpoint}],
            object_endpoints={} if endpoint=='nominal' else {endpoint:dict(events=7)}))


class EndpointExecutionTests(unittest.TestCase):
    def test_retained_bridge_rejects_unvalidated_physics_and_code(self):
        from TROTASR.workflows.continue_systematic_campaign import retained_contract_bridge
        old = dict(contracts={str(y):dict(code={'utils/histogramming.py':'old'},year=y,
                                         variations=['nominal']) for y in (2024,2025)})
        new = copy.deepcopy(old)
        for c in new['contracts'].values():
            c['code']['utils/histogramming.py']='new'
        new['endpoint_execution_update']={'chunk_size':25000}
        gate=dict(status='passed',unit_tests=104,
            records=[dict(year=2024+i%2,status='passed') for i in range(18)],
            three_signal_topologies_tested=True,large_batch_validation_passed=True,
            strict_endpoint_merge_tested=True,regions=['SR','LLCR','QCDCR','GCR','DY2E','DY2M'],
            before_contracts=copy.deepcopy(old['contracts']),after_contracts=copy.deepcopy(new['contracts']),
            selected_chunk_size=25000)
        retained_contract_bridge(new,gate,old)
        for field,value in [('unit_tests',0),('strict_endpoint_merge_tested',False),
                            ('records',[]),('selected_chunk_size',50000)]:
            broken=copy.deepcopy(gate);broken[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                retained_contract_bridge(new,broken,old)
        changed=copy.deepcopy(new);changed['contracts']['2024']['variations']=['all_weights']
        badgate=copy.deepcopy(gate);badgate['after_contracts']=changed['contracts']
        with self.assertRaisesRegex(ValueError,'physics'):
            retained_contract_bridge(changed,badgate,old)
        changed=copy.deepcopy(new);changed['contracts']['2024']['code']['utils/corrections.py']='changed'
        badgate=copy.deepcopy(gate);badgate['after_contracts']=changed['contracts']
        with self.assertRaisesRegex(ValueError,'scheduling'):
            retained_contract_bridge(changed,badgate,old)

    def test_recovered_histograms_keep_signed_yields_but_reject_invalid_arrays(self):
        from TROTASR.workflows.continue_systematic_campaign import finite_histogram_arrays
        good=dict(x=dict(sumw=[[-1.,2.]],sumw2=[[1.,4.]],entries=[[1,1]],edges=[[0,1],[0,1,2]]))
        finite_histogram_arrays(good)
        for field,value in [('sumw',float('nan')),('sumw',float('inf')),
                            ('sumw2',-1.),('entries',-1.),('sumw',True)]:
            bad=copy.deepcopy(good);bad['x'][field][0][0]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                finite_histogram_arrays(bad)
        bad=copy.deepcopy(good);del bad['x']['sumw2']
        with self.assertRaises(ValueError):finite_histogram_arrays(bad)

    def test_compatibility_binds_only_histogram_scheduling_change(self):
        from TROTASR.workflows.systematic_continuation import prior_contracts
        from TROTASR.utils.paths import ROOT
        from TROTASR.utils.io import write_json,sha256
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp)
            old={'contracts':{'2024':dict(code={'utils/histogramming.py':'before'},variations=['nominal'])}}
            new=copy.deepcopy(old);new['contracts']['2024']['code']['utils/histogramming.py']='after'
            gate=dict(status='passed',unit_tests=102,records=[{'year':2024+i%2,'status':'passed'} for i in range(18)],
                three_signal_topologies_tested=True,large_batch_validation_passed=True,strict_endpoint_merge_tested=True,
                regions=['SR','LLCR','QCDCR','GCR','DY2E','DY2M'],before_contracts=old['contracts'],
                after_contracts=new['contracts'],selected_chunk_size=25000)
            write_json(folder/'old.json',old);write_json(folder/'gate.json',gate)
            new.update(shape_mode='stored_trota_nonjme',variations=['nominal'],endpoint_execution_update=dict(
                previous_plan=str((folder/'old.json').relative_to(ROOT)),previous_plan_sha256=sha256(folder/'old.json'),
                validation=str((folder/'gate.json').relative_to(ROOT)),validation_sha256=sha256(folder/'gate.json'),chunk_size=25000))
            self.assertEqual(prior_contracts(new),{'2024':[old['contracts']['2024']]})
            new['endpoint_execution_update']['chunk_size']=50000
            with self.assertRaisesRegex(ValueError,'batch size'):prior_contracts(new)
            new['endpoint_execution_update']['chunk_size']=25000
            new['contracts']['2024']['variations']=['all_weights']
            with self.assertRaises(ValueError):prior_contracts(new)

    def test_strict_year_merge_with_split_file(self):
        from TROTASR.utils.paths import ROOT
        from TROTASR.utils.io import write_json,read_json,sha256
        from TROTASR.workflows.nominal_campaign import fingerprint,relative,COMPLETE
        from TROTASR.workflows.merge_nominal import merge
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);source=folder/'input.root';source.write_bytes(b'test-input-fingerprint')
            task=dict(key='new',year=2024,entries=7,input=relative(source),fingerprint=fingerprint(source))
            p=plan();p['tasks']=[task];p['scope']='representative_validation'
            p['contracts']['2024'].update(shape_mode='stored_trota_nonjme')
            write_json(folder/'plan.json',p)
            for child in execution_tasks(p):
                ep=child['object_endpoint'];raw=payload(ep)
                raw.update(status=COMPLETE,sr_data_blinded=True)
                raw['contract'].update(task_contract(p,child),input=str(source),fingerprint={
                    k:task['fingerprint'][k] for k in ('size','mtime_ns')})
                path=folder/'outputs'/(child['key']+'.json.gz');write_json(path,raw)
                write_json(folder/'state'/(child['key']+'.json'),dict(status='complete',output=relative(path),output_sha256=sha256(path)))
            with patch('TROTASR.workflows.merge_nominal.frozen_plan',return_value=p), \
                 patch('TROTASR.workflows.systematic_continuation.prior_contracts',return_value={}):
                output=merge(folder,2024);result=read_json(output)
                self.assertEqual(result['coverage']['completed_files'],1)
                self.assertEqual(result['coverage']['completed_execution_tasks'],17)
                self.assertEqual(result['audit']['events_read'],7)
                self.assertEqual(result['input_metadata']['a']['sumw'],5.)
                self.assertEqual(merge(folder,2024),output)
                state=folder/'state'/('new__'+NONJME_ENDPOINTS[-1]+'.json')
                write_json(state,dict(status='running'))
                with self.assertRaisesRegex(ValueError,'Incomplete task'):merge(folder,2024)

    def test_only_explicit_pending_split(self):
        p=plan();before=copy.deepcopy(p);tasks=execution_tasks(p)
        self.assertEqual(p,before);self.assertEqual(len(tasks),18)
        self.assertEqual(tasks[0],p['tasks'][0])
        self.assertEqual(tasks[1]['object_endpoint'],'nominal')
        self.assertNotIn('seed_output',tasks[1])
        self.assertEqual({t['object_endpoint'] for t in tasks[1:]},{'nominal',*NONJME_ENDPOINTS})

    def test_unknown_parent_rejected(self):
        p=plan();p['endpoint_execution_update']['split_input_keys']=['notthere']
        with self.assertRaises(ValueError):execution_tasks(p)

    def test_existing_seed_is_not_recomputed(self):
        p=plan();p['tasks'][1]['seed_output']='valid_pilot.json.gz'
        with self.assertRaisesRegex(ValueError,'seed output'):execution_tasks(p)
        p['endpoint_execution_update']['split_input_keys']=[]
        self.assertEqual(execution_tasks(p)[1],p['tasks'][1])

    def test_duplicate_parent_rejected(self):
        p=plan();p['endpoint_execution_update']['split_input_keys']=['new','new']
        with self.assertRaises(ValueError):execution_tasks(p)

    def test_non_nonjme_rejected(self):
        p=plan();p['shape_mode']='cms_trota_jme'
        with self.assertRaises(ValueError):execution_tasks(p)

    def test_contract_is_exact_one_endpoint(self):
        p=plan();tasks=execution_tasks(p)
        self.assertEqual(task_contract(p,tasks[1])['object_endpoints'],[])
        self.assertEqual(task_contract(p,tasks[2])['object_endpoints'],[tasks[2]['object_endpoint']])
        self.assertNotIn('object_task_endpoint',task_contract(p,tasks[0]))

    def test_progress_counts_input_once(self):
        p=plan();tasks={t['key']:t for t in execution_tasks(p)}
        states={k:dict(status='complete') for k in tasks}
        self.assertEqual(input_progress(p,tasks,states),dict(total=2,completed=2,completed_events=9,
                                                         execution_total=18,execution_completed=18))
        states['new__'+NONJME_ENDPOINTS[-1]]['status']='running'
        self.assertEqual(input_progress(p,tasks,states)['completed'],1)
        self.assertEqual(input_progress(p,tasks,states)['completed_events'],2)

    def test_disjoint_merge_counts_metadata_once(self):
        result=assemble_endpoints((payload(e) for e in ['nominal',*NONJME_ENDPOINTS]),NONJME_ENDPOINTS)
        self.assertEqual(len(result['histograms']['highdm']),17)
        self.assertEqual(result['audit']['events_read'],7)
        self.assertEqual(result['input_metadata']['a']['sumw'],5.)
        self.assertEqual(result['audit']['migrations'],{'x':1})
        self.assertEqual(result['histograms']['highdm']['nominal']['x']['sumw'],[2.])

    def test_missing_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            assemble_endpoints([payload('nominal')],NONJME_ENDPOINTS)

    def test_duplicate_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            assemble_endpoints([payload('nominal'),payload('up'),payload('up')],['up'])

    def test_shift_before_nominal_rejected(self):
        with self.assertRaisesRegex(ValueError,'first'):
            assemble_endpoints([payload('up'),payload('nominal')],['up'])

    def test_changed_metadata_rejected(self):
        p=payload('up');p['input_metadata']['a']['sumw']=6.
        with self.assertRaisesRegex(ValueError,'accounting'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_duplicate_nominal_rejected(self):
        p=payload('up');p['audit']['migrations']={'x':1}
        with self.assertRaisesRegex(ValueError,'nominal'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_changed_fingerprint_rejected(self):
        p=payload('up');p['contract']['fingerprint']={'size':4}
        with self.assertRaisesRegex(ValueError,'contract'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_unexpected_hist_endpoint_rejected(self):
        p=payload('up');p['histograms']['highdm']['nominal']={}
        with self.assertRaisesRegex(ValueError,'contamination'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_empty_shards_are_valid(self):
        values=[payload(e) for e in ['nominal','up']]
        for p in values:
            p.update(histograms={},dy_rz={},input_metadata={})
            p['audit'].update(events_read=0,migrations={},object_endpoints={},chunks=[],weights=[])
        result=assemble_endpoints(values,['up'])
        self.assertEqual(result['audit']['events_read'],0)


if __name__=='__main__':unittest.main()

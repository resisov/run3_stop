import copy
import unittest
from TROTASR.workflows.nominal_campaign import capacity, external_process_slots, current_contract
from unittest.mock import patch
from TROTASR.workflows.merge_nominal import merge_histograms, merge_datasets, merge_counters


class NominalCampaignTests(unittest.TestCase):
    def test_seed_relocation_requires_same_inode(self):
        from TROTASR.workflows.nominal_campaign import relocated_seed
        from TROTASR.utils.paths import ROOT
        t=dict(key='seed',year=2024,input='validation/deleted/inputs/2024/a.root',
               seed_output='validation/deleted/a.json.gz',fingerprint={'inode':17})
        with patch('pathlib.Path.exists',return_value=False), patch('TROTASR.workflows.nominal_campaign.fingerprint',return_value={'inode':17}):
            self.assertEqual(relocated_seed(t)['input'],'inputs/2024/a.root')
            self.assertEqual(relocated_seed(dict(t,seed_output=None))['input'],'inputs/2024/a.root')
        with patch('pathlib.Path.exists',return_value=False), patch('TROTASR.workflows.nominal_campaign.fingerprint',return_value={'inode':18}):
            with self.assertRaisesRegex(ValueError,'same original inode'):relocated_seed(t)

    def test_file_groups_never_repeat_running_or_successful_endpoints(self):
        from TROTASR.workflows.nominal_campaign import file_groups, execution_tasks
        from TROTASR.utils.corrections import NONJME_ENDPOINTS
        p=dict(shape_mode='stored_trota_nonjme',variations=['nominal'],
            tasks=[dict(key='a',year=2024,entries=2),dict(key='b',year=2025,entries=3)],
            endpoint_execution_update={'split_input_keys':['a','b']})
        states={'a__nominal':dict(status='complete'),'a__'+NONJME_ENDPOINTS[0]:dict(status='running')}
        groups=file_groups(p,states)
        self.assertEqual([g['parent_key'] for g in groups],['b','a'])
        keys=[k for g in groups for k in g['task_keys']]
        self.assertEqual(set(keys),{t['key'] for t in execution_tasks(p)}-set(states))
        self.assertEqual(len(keys),len(set(keys)))
        with self.assertRaises(ValueError):file_groups(p,{'foreign':{}})

    def test_file_group_does_not_recompute_seed(self):
        from TROTASR.workflows.nominal_campaign import file_groups
        p=dict(tasks=[dict(key='seed',entries=1,seed_output='valid.json.gz')])
        with self.assertRaisesRegex(ValueError,'reconciled'):file_groups(p,{})
        self.assertEqual(file_groups(p,{'seed':dict(status='complete')}),[])

    def test_roundoff_is_not_count_or_binning_permission(self):
        from TROTASR.workflows.run_systematic_campaign import compare_values
        compare_values({'sumw':[1.]},{'sumw':[1.+1e-8]})
        for a,b in (({'entries':[1]},{'entries':[2]}),({'a':1},{'b':1}),([1],[1,2]),(1.,float('nan')),(1.,1.01)):
            with self.assertRaises(ValueError):compare_values(a,b)

    def test_unregistered_legacy_result_is_rejected(self):
        from TROTASR.workflows.nominal_campaign import integration_payload
        p=dict(contracts={'2024':{'code':{'a':'new'},'year':2024}},
            _execution={'preserved':{},'retained':[],'legacy':None})
        task=dict(key='x',year=2024)
        raw={'contract':{'code':{'a':'old'},'year':2024}}
        with self.assertRaisesRegex(ValueError,'Unregistered'):integration_payload(raw,task,p,{'output':'x','output_sha256':'h'})

    def test_preserved_legacy_result_has_exact_hash_and_physics(self):
        from TROTASR.workflows.nominal_campaign import integration_payload
        p=dict(contracts={'2024':{'code':{'a':'new'},'year':2024,'normalization_sha256':'fixed'}},
            _execution={'preserved':{'x':dict(output='x',output_sha256='h')},'retained':[],'legacy':None})
        task=dict(key='x',year=2024)
        raw={'contract':{'code':{'a':'old'},'year':2024,'normalization_sha256':'fixed'}}
        with patch('TROTASR.workflows.nominal_campaign.validate_payload',return_value='checked') as validate:
            self.assertEqual(integration_payload(raw,task,p,dict(output='x',output_sha256='h')),'checked')
            self.assertEqual(validate.call_args.args[2],raw['contract'])
        with self.assertRaisesRegex(ValueError,'Unregistered'):
            integration_payload(raw,task,p,dict(output='x',output_sha256='different'))

    def test_nonjme_traversal_retains_valid_empty_inputs(self):
        from TROTASR.workflows.nominal_campaign import validate_payload, COMPLETE
        from TROTASR.utils.paths import ROOT
        expected=dict(shape_mode='stored_trota_nonjme',object_endpoints=['up','down'])
        task=dict(input='inputs/empty.root',entries=0,fingerprint=dict(size=1,mtime_ns=2))
        payload=dict(status=COMPLETE,sr_data_blinded=True,histograms={},
                     contract=dict(expected,input=str(ROOT/task['input']),fingerprint=task['fingerprint']),
                     audit=dict(events_read=0,chunks=[],object_endpoints={}))
        self.assertIs(validate_payload(payload,task,expected),payload)
        task['entries']=1;payload['audit']['events_read']=1
        with self.assertRaisesRegex(ValueError,'endpoint traversal'):validate_payload(payload,task,expected)
        payload['audit']['object_endpoints']={'up':{'events':1},'down':{'events':1}}
        self.assertIs(validate_payload(payload,task,expected),payload)

    def test_stored_trota_weight_contract_is_only_weight_selection(self):
        with patch('TROTASR.workflows.nominal_campaign.sha256', return_value='digest'):
            nominal = current_contract(2024)
            weights = current_contract(2024, variations=['all_weights'])
            self.assertEqual(weights, dict(nominal, variations=['all_weights']))
            self.assertNotIn('shape_mode', weights)
            self.assertNotIn('trota_models_sha256', weights)
            jme = current_contract(2024, 'cms_trota_jme')
            self.assertEqual(jme, current_contract(2024, 'cms_trota_jme', ['all_weights']))
            with self.assertRaises(ValueError):
                current_contract(2024, 'cms_trota_jme', ['nominal'])
            with self.assertRaises(ValueError):
                current_contract(2024, variations=['invented'])

    def test_global_budget_includes_other_controller(self):
        plan = dict(max_workers=99, aggregate_slots=100, minimum_available_memory_bytes=32,
                    admission_bytes_per_worker=4)
        self.assertEqual(capacity(plan, 52, 1000, 0), 47)
        self.assertEqual(capacity(plan, 100, 1000, 0), 0)
        self.assertEqual(capacity(plan, 0, 40, 3), 5)
        self.assertEqual(capacity(plan, 0, 20, 3), 3)

    def test_process_descendants_excluded(self):
        rows = [(1,0,'python3'), (2,1,'python3'), (3,2,'combine'), (4,0,'python3'),
                (5,4,'python3'), (6,0,'rsync'), (7,0,'sshd')]
        self.assertEqual(external_process_slots(rows, 1), 3)

    def test_histogram_addition_not_concatenation(self):
        one = {'x':dict(edges=[0,1,2],sumw=[-2.,3.],sumw2=[4.,9.],entries=[1,1])}
        two = copy.deepcopy(one)
        merge_histograms(one,two)
        self.assertEqual(one['x']['sumw'],[-4.,6.])
        self.assertEqual(one['x']['sumw2'],[8.,18.])
        self.assertEqual(one['x']['edges'],[0,1,2])

    def test_different_bin_edges_rejected(self):
        one = dict(edges=[0,1],sumw=[1.],sumw2=[1.],entries=[1])
        two = dict(edges=[0,2],sumw=[1.],sumw2=[1.],entries=[1])
        with self.assertRaises(ValueError):
            merge_histograms(one,two)

    def test_two_dimensional_template_preserved(self):
        one = dict(edges=[[0,1],[250,300,500]],sumw=[[1,2]],sumw2=[[1,4]],entries=[[1,1]])
        merge_histograms(one,copy.deepcopy(one))
        self.assertEqual(one['sumw'],[[2,4]])
        self.assertEqual(one['edges'],[[0,1],[250,300,500]])

    def test_dataset_ids_and_normalization_not_added(self):
        ds={'1':dict(dataset='sample',dataset_id=1,process='DY',events_written=7,
                      sumw=3.5,xsec_pb=10.,sumw_source_counts={'Runs':1})}
        merge_datasets(ds,copy.deepcopy(ds))
        self.assertEqual(ds['1']['dataset_id'],1)
        self.assertEqual(ds['1']['xsec_pb'],10.)
        self.assertEqual(ds['1']['events_written'],14)
        self.assertEqual(ds['1']['sumw_source_counts'],{'Runs':2})
        other=copy.deepcopy(ds);other['1']['xsec_pb']=20.
        with self.assertRaises(ValueError):
            merge_datasets(ds,other)

    def test_migrations_add_and_nan_fails(self):
        m={'p':{'SR':{'gained':2,'sumw':3.}}}
        merge_counters(m,copy.deepcopy(m))
        self.assertEqual(m['p']['SR']['gained'],4)
        with self.assertRaises(ValueError):
            merge_counters(m,{'p':{'SR':{'sumw':float('nan')}}})


if __name__=='__main__':
    unittest.main()

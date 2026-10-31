import unittest
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
from TROTASR.utils.paths import ROOT, payload_path
from TROTASR.utils.io import sha256, read_json
from TROTASR.utils.plot_adapter import category_label


class Contracts(unittest.TestCase):
    def test_new_update_contract_is_year_specific_without_historical_match(self):
        from TROTASR.workflows.run_systematic_campaign import refresh_result_contract, SUPPORT_REFRESH
        spec={'contracts':{'2024':{'year':2024},'2025':{'year':2025}}, SUPPORT_REFRESH:{}}
        for year in (2024,2025):
            self.assertEqual(refresh_result_contract(spec,{'key':'x','year':year},None),{'year':year})

    def test_muon_migrations_are_recorded_not_vetoed(self):
        from TROTASR.workflows.run_systematic_campaign import compare_updated_populations
        leaf=dict(sumw=[2.],sumw2=[4.],entries=[1],edges=[0.,1.])
        before=dict(histograms={'x':leaf},input_metadata={},audit={'events_read':2})
        after=copy.deepcopy(before);after['histograms']['x']['entries']=[2]
        records=compare_updated_populations(before,after)
        self.assertEqual(records,[dict(path='histograms/x',before=[1],after=[2])])
        after['histograms']['x']['edges']=[0.,2.]
        with self.assertRaisesRegex(ValueError,'axes'):compare_updated_populations(before,after)

    def test_refresh_prior_contract_requires_registered_exact_output(self):
        from TROTASR.workflows.run_systematic_campaign import refresh_result_contract, SUPPORT_REFRESH
        spec={'contract': {'current': True}, SUPPORT_REFRESH: {
            'previous_contract': {'previous': True}, 'retained_completed': {'one': 'digest'}}}
        self.assertEqual(refresh_result_contract(spec, {'key':'one'}, 'digest'), {'previous':True})
        self.assertEqual(refresh_result_contract(spec, {'key':'one'}, 'changed'), {'current':True})
        self.assertEqual(refresh_result_contract(spec, {'key':'two'}, 'digest'), {'current':True})

    def test_refresh_ignores_retired_output_location(self):
        from TROTASR.workflows import run_systematic_campaign as refresh
        from TROTASR.utils.io import write_json
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp); campaign=root/'campaigns/example'
            retired=root/'retired/x.json.gz'
            write_json(retired, {'old':True})
            record=dict(key='x',output='retired/x.json.gz',sha256=sha256(retired),endpoint=None)
            with patch.object(refresh,'ROOT',root):
                with self.assertRaises(FileNotFoundError):refresh.refresh_output_record(campaign,record)
                canonical=campaign/'outputs/x.json.gz'
                canonical.parent.mkdir(parents=True)
                canonical.write_bytes(retired.read_bytes())
                self.assertEqual(refresh.refresh_output_record(campaign,record)['output'],
                                 'campaigns/example/outputs/x.json.gz')

    def test_refresh_uses_exact_old_hash_not_new_canonical_bytes(self):
        from TROTASR.workflows import run_systematic_campaign as refresh
        from TROTASR.utils.io import write_json
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp); campaign=root/'campaigns/example'
            original=campaign/'outputs/x.json.gz'
            backup=campaign/'history'/refresh.REFRESH_ID/'outputs/x.json.gz'
            write_json(original,{'new':True});write_json(backup,{'old':True})
            record=dict(key='x',output=str(original.relative_to(root)),sha256=sha256(backup))
            with patch.object(refresh,'ROOT',root):
                self.assertEqual(refresh.refresh_output_record(campaign,record)['output'],str(backup.relative_to(root)))
                with self.assertRaises(FileNotFoundError):
                    refresh.refresh_output_record(campaign,dict(record,sha256='wrong'))

    def test_deleted_seed_never_claims_old_population_equality(self):
        from TROTASR.workflows.run_systematic_campaign import refreshed_population_verified,refresh_task_required
        task=dict(key='2025_data_shard_01052',refresh_required=False,rebuild_missing_seed=True)
        state=dict(previous_populations_identical=None,
                   population_validation='retained_input_contract_old_seed_unavailable')
        self.assertTrue(refresh_task_required(task))
        self.assertTrue(refreshed_population_verified(task,state))
        self.assertFalse(refreshed_population_verified(dict(task,key='other'),state))
        self.assertFalse(refreshed_population_verified(dict(task,rebuild_missing_seed=False),state))
        self.assertFalse(refreshed_population_verified(task,dict(state,previous_populations_identical=False)))

    def test_reference_handoff_retains_success_and_requeues_only_missing_seed(self):
        from TROTASR.workflows import run_systematic_campaign as refresh
        from TROTASR.utils.io import write_json
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp); campaign=root/'campaigns/systematics_20260924'
            campaign.mkdir(parents=True); (campaign/'controller.lock').touch()
            name='workflows/run_systematic_campaign.py'
            source=root/name;source.parent.mkdir();source.write_text('new runner')
            backup=campaign/'history'/refresh.REFRESH_ID/refresh.REFERENCE_REPAIR/'sources'/name
            backup.parent.mkdir(parents=True);backup.write_text('old runner')
            tasks=[];previous={};originals=[]
            failed='2025_signal_shard_00568'
            for key in sorted(refresh.MISSING_REFRESH_SEEDS):
                task=dict(key=key,year=2025,input='inputs/2025/'+key[5:]+'.root',
                          fingerprint={'inode':17},refresh_required='_data_' not in key)
                tasks.append(task);old='retired/'+key+'.json.gz'
                originals.append(dict(task,seed_output=old))
                previous[key]=[dict(key=key,output=old,sha256='lost',endpoint=None)]
                s=dict(key=key,status='complete')
                if key==failed:s.update(status='failed',payload_revision=refresh.REFRESH_ID,exception_type='FileNotFoundError')
                write_json(campaign/'state'/(key+'.json'),s)
            key='2025_mc_shard_11111'
            task=dict(key=key,year=2025,input='inputs/2025/mc_shard_11111.root',fingerprint={'inode':17},refresh_required=True)
            tasks.append(task);originals.append(dict(task))
            output=campaign/'outputs'/(key+'.json.gz')
            old=campaign/'history'/refresh.REFRESH_ID/'outputs'/output.name
            write_json(output,{'valid_new':True});write_json(old,{'valid_old':True})
            digest=sha256(output)
            previous[key]=[dict(key=key,output=str(output.relative_to(root)),sha256=sha256(old),endpoint=None)]
            write_json(campaign/'state'/(key+'.json'),dict(status='complete',payload_revision=refresh.REFRESH_ID,
                output=str(output.relative_to(root)),output_sha256=digest,exit_code=0,previous_populations_identical=True))
            spec=dict(id=refresh.REFRESH_ID,year=2025,tasks=tasks,previous_outputs=previous,
                      sources={name:sha256(backup)},contract={},variations=['all_weights'])
            write_json(campaign/'plan.json',dict(tasks=originals,shape_mode='cms_trota_jme',payload_refresh=spec))
            write_json(campaign/'state.json',dict(active=0,status='payload_refresh_needs_attention',failed=[failed]))
            with patch.object(refresh,'ROOT',root), \
                 patch('TROTASR.workflows.nominal_campaign.verify_compiled'), \
                 patch('TROTASR.workflows.nominal_campaign.current_contract',return_value={}), \
                 patch('TROTASR.workflows.nominal_campaign.relocated_seed',side_effect=lambda t:dict(t)):
                result=refresh.repair_refresh_references(campaign)['payload_refresh']
                self.assertEqual(result[refresh.REFERENCE_REPAIR]['retained_completed'],{key:digest})
                self.assertEqual(result[refresh.REFERENCE_REPAIR]['requeue'],[failed])
                self.assertEqual(read_json(campaign/'state'/(failed+'.json'))['status'],'pending')
                self.assertEqual(sha256(output),digest)
                self.assertEqual(sum(refresh.refresh_task_required(t) for t in result['tasks']),7)
                self.assertEqual(refresh.repair_refresh_references(campaign)['payload_refresh'],result)

    def test_refresh_keeps_data_and_updates_background_and_signal(self):
        from TROTASR.workflows.run_systematic_campaign import refresh_required
        for kind in ('data','mc','signal'):
            task=dict(key='2025_'+kind+'_shard_00000',year=2025,
                input='inputs/2025/'+kind+'_shard_00000.root')
            self.assertEqual(refresh_required(task),kind!='data')
            with self.assertRaises(ValueError):refresh_required(dict(task,year=2024))
            with self.assertRaises(ValueError):refresh_required(dict(task,input='inputs/2025/other.root'))

    def test_payload_refresh_changes_weights_not_event_populations(self):
        from TROTASR.workflows.run_systematic_campaign import compare_refresh_populations
        leaf=dict(sumw=[2.],sumw2=[4.],entries=[1],edges=[0.,1.])
        before=dict(histograms={'highdm':{'nominal':{'SR':{'Nb1':{'TT':{'x':leaf}}}}}},
                    input_metadata={'TT':{'is_data':False}},audit={'events_read':1})
        after=copy.deepcopy(before)
        after['histograms']['highdm']['nominal']['SR']['Nb1']['TT']['x']['sumw']=[3.]
        compare_refresh_populations(before,after)
        after['histograms']['highdm']['nominal']['SR']['Nb1']['TT']['x']['entries']=[2]
        with self.assertRaises(ValueError):compare_refresh_populations(before,after)

    def test_payload_refresh_rejects_changed_bookkeeping(self):
        from TROTASR.workflows.run_systematic_campaign import compare_refresh_populations
        before=dict(input_metadata={'TT':{'sumw':3}},audit={'events_read':1})
        after=copy.deepcopy(before);after['input_metadata']['TT']['sumw']=4
        with self.assertRaises(ValueError):compare_refresh_populations(before,after)

    def test_config_relocation_is_explicit_not_a_missing_file_fallback(self):
        from TROTASR.utils.paths import internal_path, JSONS
        self.assertEqual(internal_path(ROOT/'assets.json'),JSONS/'assets.json')
        self.assertEqual(internal_path(ROOT/'unregistered.json'),ROOT/'unregistered.json')
        self.assertEqual(internal_path(ROOT/'campaigns/x/state.json'),ROOT/'campaigns/x/state.json')
        with self.assertRaises(ValueError):internal_path(ROOT/'../assets.json')


    def test_payloads_do_not_escape(self):
        for p in ('../../outside', '/tmp/outside'):
            with self.assertRaises(ValueError): payload_path(p)
        self.assertEqual(payload_path('data/JMESF/2024/jet_jerc.json.gz'), ROOT / 'scales/JME/2024/jet_jerc.json.gz')

    def test_all_five_sr_axes_visible(self):
        label = category_label('highdm', 'SR', '2,1,1,0,0')
        positions = [label.index(token) for token in ('N_b', 'N_{bst}', 'N_{mix}', 'N_{res}', 'N_W')]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('N_W=0', label)

    def test_cr_not_topology_replicated(self):
        self.assertEqual(category_label('highdm', 'LLCR', 'Nb1'), r'$N_b=1$')
        self.assertNotIn('N_{mix}', category_label('lowdm', 'GCR', 'Nb2'))

    def test_adl_companions_present(self):
        for name in ('ids', 'corrections', 'event_selections'):
            self.assertTrue((ROOT / 'utils' / (name + '.adl')).stat().st_size > 100)


if __name__ == '__main__': unittest.main()

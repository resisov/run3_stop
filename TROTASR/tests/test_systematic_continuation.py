import unittest
from TROTASR.workflows.systematic_continuation import approved_retry,retry_keys,prior_contracts,FIELD,ERRORS


class SystematicContinuationTests(unittest.TestCase):
    def test_statistics_adapter_accepts_only_exact_statistics_sources(self):
        from pathlib import Path
        from unittest.mock import patch
        names=('utils/statistics_inputs.py','utils/statistical_templates.py')
        before={str(y):dict(year=y,code={n:'old' for n in names}) for y in (2024,2025)}
        after={str(y):dict(year=y,code={n:'new' for n in names}) for y in (2024,2025)}
        gate=dict(status='passed',unit_tests=71,event_processing_imports_unchanged=True,
            all_event_outputs_equal=True,records=[dict(year=y) for y in (2024,2025)],
            before_contracts=before,after_contracts=after)
        plan=dict(shape_mode='stored_trota_nonjme',contracts=after,statistics_adapter_update=dict(
            previous_plan='old',validation='gate',previous_plan_sha256='digest',validation_sha256='digest'))
        records={'old':dict(contracts=before),'gate':gate}
        prefix='TROTASR.workflows.systematic_continuation.'
        with patch(prefix+'internal_path',side_effect=lambda p:p),patch(prefix+'sha256',return_value='digest'),\
             patch(prefix+'read_json',side_effect=lambda p:records[Path(p).name]):
            self.assertEqual(prior_contracts(plan),{y:[c] for y,c in before.items()})
            gate['all_event_outputs_equal']=False
            with self.assertRaisesRegex(ValueError,'Missing statistics-only'):prior_contracts(plan)
            gate['all_event_outputs_equal']=True
            after['2024']['code']['utils/corrections.py']='unapproved'
            with self.assertRaisesRegex(ValueError,'Unexpected event source'):prior_contracts(plan)

    def test_nonjme_throughput_requires_exact_validated_change(self):
        import copy
        from pathlib import Path
        from unittest.mock import patch
        names=('utils/corrections.py','utils/pog_corrections.py',
               'utils/event_selections.py','utils/histogramming.py')
        before={str(y):dict(year=y,code={n:'old' for n in names},bins=118) for y in (2024,2025)}
        after={str(y):dict(year=y,code={n:'new' for n in names},bins=118) for y in (2024,2025)}
        gate=dict(status='passed',unit_tests=82,three_signal_topologies_tested=True,
                  records=[dict(year=y,all_histograms_equal=True,selection_weight_audits_equal=True)
                           for y in (2024,2025) for _ in range(9)],
                  regions=['SR','LLCR','QCDCR','GCR','DY2E','DY2M'],
                  before_contracts=before,after_contracts=after)
        plan=dict(shape_mode='stored_trota_nonjme',variations=['nominal'],contracts=after,
                  nonjme_throughput_update=dict(previous_plan='old',validation='gate',
                    previous_plan_sha256='digest',validation_sha256='digest'))
        records={'old':dict(contracts=before),'gate':gate}
        prefix='TROTASR.workflows.systematic_continuation.'
        with patch(prefix+'internal_path',side_effect=lambda p:p),patch(prefix+'sha256',return_value='digest'),\
             patch(prefix+'read_json',side_effect=lambda p:records[Path(p).name]):
            self.assertEqual(prior_contracts(plan),{y:[c] for y,c in before.items()})
            gate['records'][0]['all_histograms_equal']=False
            with self.assertRaisesRegex(ValueError,'Missing full-file'):prior_contracts(plan)
            gate['records'][0]['all_histograms_equal']=True
            after['2024']['bins']=117
            with self.assertRaisesRegex(ValueError,'Non-code physics'):prior_contracts(plan)
            after['2024']['bins']=118
            after['2024']['code']['utils/ids.py']='unapproved'
            with self.assertRaisesRegex(ValueError,'Unexpected non-JME'):prior_contracts(plan)

    def test_nonjme_throughput_rejects_other_campaigns(self):
        for mode in (None,'cms_trota_jme'):
            with self.assertRaisesRegex(ValueError,'non-JME nominal-weight'):
                prior_contracts(dict(shape_mode=mode,variations=['nominal'],
                                     nonjme_throughput_update={'validation':'unreached'}))

    def test_response_update_keeps_both_weight_contract_generations(self):
        import copy
        from pathlib import Path
        from unittest.mock import patch
        names=('utils/corrections.py','utils/shape_kinematics.py','utils/histogramming.py')
        original={'2024':dict(code={n:'original' for n in names},year=2024)}
        extended={'2024':dict(code={n:'extended' for n in names},year=2024)}
        current=copy.deepcopy(extended);current['2024']['code']['utils/histogramming.py']='audit'
        extension_gate=dict(status='passed',stored_trota_nonjme_validation=True,unit_tests=46,
            records=[dict(all_weight_histograms_equal=True) for _ in range(16)],
            regions=['SR','LLCR','QCDCR','GCR','DY2E','DY2M'],
            baseline_weight_contracts=original,tested_weight_contracts=extended)
        extension=dict(variations=['all_weights'],contracts=extended,nonjme_code_extension=dict(
            previous_plan='original',validation='extension_gate',previous_plan_sha256='digest',validation_sha256='digest'))
        gate=dict(status='passed',unit_tests=48,multi_dataset_files_passed=True,
            old_success_histograms_equal=True,weight_histograms_equal=True,
            before_contracts={'weights':extended},after_contracts={'weights':current})
        plan=dict(variations=['all_weights'],contracts=current,object_response_audit_update=dict(
            previous_plan='extended',validation='audit_gate',previous_plan_sha256='digest',validation_sha256='digest'))
        records={'original':dict(contracts=original),'extended':extension,
                 'extension_gate':extension_gate,'audit_gate':gate}
        prefix='TROTASR.workflows.systematic_continuation.'
        with patch(prefix+'internal_path',side_effect=lambda p:p),patch(prefix+'sha256',return_value='digest'),\
             patch(prefix+'read_json',side_effect=lambda p:records[Path(p).name]):
            self.assertEqual(prior_contracts(plan),{'2024':[extended['2024'],original['2024']]})
            gate['weight_histograms_equal']=False
            with self.assertRaisesRegex(ValueError,'Missing object-response'):prior_contracts(plan)

    def test_response_audit_failure_has_one_scoped_attempt(self):
        plan={'object_response_audit_update':{'original_attempts':{'a':1}}}
        state=dict(status='failed',attempt=1,exception_type='ValueError',
                   error='operands could not be broadcast together with shapes (215,) (161,)',
                   traceback='utils/histogramming.py selected_recoil_changed')
        self.assertTrue(approved_retry(plan,'a',state))
        old_location='File "/campaign/utils/histogramming.py", line 252, in build\n    else:'
        self.assertTrue(approved_retry(plan,'a',dict(state,traceback=old_location)))
        self.assertFalse(approved_retry(plan,'a',dict(state,traceback=old_location.replace('252','999'))))
        for key,changes in [('a',{'attempt':2}),('b',{}),('a',{'traceback':'different computation'}),
                            ('a',{'error':'Nonfinite weights'})]:
            self.assertFalse(approved_retry(plan,key,dict(state,**changes)))

    def test_nonjme_extension_preserves_exact_old_weight_contract_only(self):
        import copy
        from unittest.mock import patch
        from pathlib import Path
        changed=('utils/corrections.py','utils/shape_kinematics.py','utils/histogramming.py')
        before={'2024':dict(code={n:'old' for n in changed},variations=['all_weights'],year=2024)}
        after={'2024':dict(code={n:'new' for n in changed},variations=['all_weights'],year=2024)}
        old=dict(contracts=before)
        gate=dict(status='passed',stored_trota_nonjme_validation=True,unit_tests=44,
                  records=[dict(all_weight_histograms_equal=True) for _ in range(10)],
                  regions=['SR','LLCR','QCDCR','GCR','DY2E','DY2M'],
                  baseline_weight_contracts=before,tested_weight_contracts=after)
        plan=dict(variations=['all_weights'],contracts=after,nonjme_code_extension=dict(
            previous_plan='old',validation='gate',previous_plan_sha256='digest',validation_sha256='digest'))
        prefix='TROTASR.workflows.systematic_continuation.'
        with patch(prefix+'internal_path',side_effect=lambda p:p),patch(prefix+'sha256',return_value='digest'),\
             patch(prefix+'read_json',side_effect=lambda p:old if Path(p).name=='old' else gate):
            self.assertEqual(prior_contracts(plan),before)
            bad=copy.deepcopy(plan);bad['shape_mode']='stored_trota_nonjme'
            with self.assertRaisesRegex(ValueError,'scoped'):prior_contracts(bad)
            bad=copy.deepcopy(plan);bad['contracts']['2024']['year']=2025
            with self.assertRaisesRegex(ValueError,'contracts differ'):prior_contracts(bad)
            gate['records'][0]['all_weight_histograms_equal']=False
            with self.assertRaisesRegex(ValueError,'Incomplete'):prior_contracts(plan)

    def test_user_stop_has_only_one_exact_resumption(self):
        state=dict(status='failed',attempt=3,exception_type='UserRequestedOptimization',
                   error='User requested worker stop for validated throughput optimization')
        plan={'throughput_optimization':{'resumable_states':{'a':state}}}
        self.assertTrue(approved_retry(plan,'a',state))
        for changed in ({'attempt':4},{'error':'new error'},{'status':'complete'},
                        {'exception_type':'ValueError'}):
            self.assertFalse(approved_retry(plan,'a',dict(state,**changed)))
        self.assertFalse(approved_retry(plan,'b',state))

    def test_throughput_cannot_bypass_nominal_contract(self):
        with self.assertRaisesRegex(ValueError,'current JME'):
            prior_contracts({'throughput_optimization':{'validation':'unreached'}})

    def test_negative_pt_has_one_exact_corrected_attempt(self):
        plan={'negative_pt_selection':{'original_attempts':{'a':2,'b':1}}}
        s=dict(status='failed',attempt=2,exception_type='ValueError',error='Negative TROTA pT: jet_nanoaod_pt')
        self.assertTrue(approved_retry(plan,'a',s))
        self.assertFalse(approved_retry(plan,'b',s))
        self.assertFalse(approved_retry(plan,'a',dict(s,attempt=3)))
        self.assertFalse(approved_retry(plan,'a',dict(s,error='Nonfinite TROTA p4')))

    def test_multiple_explicit_prior_contracts(self):
        from TROTASR.workflows.normalization_recovery import payload_contract
        old,prior,current={'code':1},{'code':2},{'code':3}
        self.assertEqual(payload_contract({'contract':old},current,[old,prior]),old)
        self.assertEqual(payload_contract({'contract':prior},current,[old,prior]),prior)
        self.assertEqual(payload_contract({'contract':{'code':99}},current,[old,prior]),current)

    def test_only_original_first_attempt_known_failure_is_retried(self):
        plan={FIELD:dict(original_attempt_keys=['a'])}
        s=dict(status='failed',attempt=1,exception_type='ValueError',error=sorted(ERRORS)[0])
        self.assertTrue(approved_retry(plan,'a',s))
        self.assertFalse(approved_retry(plan,'b',s))
        self.assertFalse(approved_retry(plan,'a',dict(s,attempt=2)))
        self.assertFalse(approved_retry(plan,'a',dict(s,error='different error')))
        self.assertFalse(approved_retry(plan,'a',dict(s,status='complete')))
        self.assertEqual(retry_keys(plan,{'a':s,'b':s}),['a'])

    def test_other_campaigns_are_unchanged(self):
        self.assertEqual(prior_contracts({}),{})
        self.assertEqual(retry_keys({},{}),[])

    def test_policy_cannot_be_used_for_nominal_or_normalization_changes(self):
        with self.assertRaisesRegex(ValueError,'original JME'):
            prior_contracts({FIELD:{'validation':'unreached'}})


if __name__=='__main__':unittest.main()

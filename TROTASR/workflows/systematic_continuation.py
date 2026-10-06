"""Exact compatibility for the two approved, formerly file-fatal input cases.

Successful old results are immutable: both updates only extend a formerly
raising domain. No arbitrary source/normalization contract bypass is allowed.
"""
import os
import re
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.utils.io import read_json,sha256
from TROTASR.workflows.normalization_recovery import RetainedProcess

FIELD='systematic_input_policy_update'
CHANGED={'utils/systematic_views.py','utils/trota_candidates.py','utils/trota_inference.py'}
ERRORS={'Stored calibration does not match internal JME: jet_corrected_pt',
        'Stored calibration does not match internal JME: jet_corrected_mass',
        'Undefined official Xbb/(Xbb+QCD) input'}
THROUGHPUT_CHANGED={'utils/event_selections.py','utils/pog_corrections.py',
                    'utils/weight_components.py','utils/corrections.py',
                    'utils/private_scales.py','utils/histogramming.py'}


def approved_retry(plan,key,state):
    attempt=plan.get('object_response_audit_update',{}).get('original_attempts',{}).get(key)
    trace=state.get('traceback','')
    # Retained workers execute old bytecode, but traceback linecache can read
    # the newly deployed source. The exact old failure location is line 252;
    # its printed text may therefore be "else:" instead of the old expression.
    response_location=('selected_recoil_changed' in trace or
                       re.search(r'File "[^"\n]*/utils/histogramming\.py", line 252, in build',trace))
    if (attempt is not None and state.get('attempt')==attempt and state.get('status')=='failed'
        and state.get('exception_type')=='ValueError'
        and state.get('error','').startswith('operands could not be broadcast together with shapes')
        and response_location and 'utils/histogramming.py' in trace):
        return True
    paused=plan.get('throughput_optimization',{}).get('resumable_states',{}).get(key)
    if paused and all(state.get(k)==v for k,v in paused.items()):
        return True
    extension=plan.get('negative_pt_selection',{}).get('original_attempts',{}).get(key)
    if extension is not None:
        if (state.get('status')=='failed' and state.get('attempt')==extension
            and state.get('exception_type')=='ValueError'
            and state.get('error') in {'Negative TROTA pT: jet_nanoaod_pt',
                                      'Negative TROTA pT: fatjet_nanoaod_pt'}):
            return True
    spec=plan.get(FIELD,{})
    return (key in spec.get('original_attempt_keys',[]) and state.get('status')=='failed'
            and state.get('attempt')==1 and state.get('exception_type')=='ValueError'
            and state.get('error') in ERRORS)


def retry_keys(plan,states):
    return sorted(k for k,s in states.items() if approved_retry(plan,k,s))


def adopt_worker(plan,campaign,key,state):
    r=plan.get(FIELD,{}).get('retained_workers',{}).get(key)
    if r is None:return None
    if state.get('pid')!=r['pid'] or state.get('attempt')!=r['attempt']:
        raise ValueError('Changed retained systematic worker: '+key)
    p=RetainedProcess(r)
    p.poll()
    return p,open(os.devnull,'w'),r['attempt']


def prior_contracts(plan, _historical=False):
    endpoint = plan.get('endpoint_execution_update')
    if endpoint:
        old_path, gate_path = [internal_path(ROOT/endpoint[k]) for k in ('previous_plan','validation')]
        if (sha256(old_path)!=endpoint['previous_plan_sha256'] or
                sha256(gate_path)!=endpoint['validation_sha256']):
            raise ValueError('Changed independent-endpoint compatibility evidence')
        old, gate = read_json(old_path), read_json(gate_path)
        records = gate.get('records',[])
        if (plan.get('shape_mode')!='stored_trota_nonjme' or plan.get('variations')!=['nominal'] or
                gate.get('status')!='passed' or gate.get('unit_tests',0)<99 or len(records)<18 or
                {r['year'] for r in records}!={2024,2025} or
                not all(r.get('status')=='passed' for r in records) or
                not gate.get('three_signal_topologies_tested') or
                not gate.get('large_batch_validation_passed') or
                not gate.get('strict_endpoint_merge_tested') or
                set(gate.get('regions',[]))!={'SR','LLCR','QCDCR','GCR','DY2E','DY2M'}):
            raise ValueError('Missing independent endpoint physics/performance validation')
        if (old['contracts']!=gate['before_contracts'] or plan['contracts']!=gate['after_contracts'] or
                endpoint.get('chunk_size')!=gate['selected_chunk_size']):
            raise ValueError('Endpoint execution differs from tested contracts/batch size')
        for year,current in plan['contracts'].items():
            before=old['contracts'][year]
            if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in before.items() if k!='code'}:
                raise ValueError('Endpoint scheduling changed physics prescription')
            changed={k for k in set(current['code'])|set(before['code']) if current['code'].get(k)!=before['code'].get(k)}
            if changed!={'utils/histogramming.py'}:
                raise ValueError('Unexpected independent endpoint physics source change')
        ancestors=prior_contracts(old,True)
        return {y:[c]+(ancestors.get(y,[]) if isinstance(ancestors.get(y,[]),list) else [ancestors[y]])
                for y,c in old['contracts'].items()}
    adapter = plan.get('statistics_adapter_update')
    if adapter:
        old_path, gate_path = [internal_path(ROOT/adapter[k]) for k in ('previous_plan','validation')]
        if sha256(old_path)!=adapter['previous_plan_sha256'] or sha256(gate_path)!=adapter['validation_sha256']:
            raise ValueError('Changed statistics-adapter compatibility evidence')
        old, gate = read_json(old_path), read_json(gate_path)
        if (plan.get('shape_mode')!='stored_trota_nonjme' or
                gate.get('status')!='passed' or gate.get('unit_tests',0)<71 or
                not gate.get('event_processing_imports_unchanged') or
                not gate.get('all_event_outputs_equal') or
                {r['year'] for r in gate.get('records',[])}!={2024,2025}):
            raise ValueError('Missing statistics-only deployment validation')
        if old['contracts']!=gate['before_contracts'] or plan['contracts']!=gate['after_contracts']:
            raise ValueError('Statistics-adapter contracts differ from validation')
        for year, current in plan['contracts'].items():
            before=old['contracts'][year]
            if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in before.items() if k!='code'}:
                raise ValueError('Statistics adapter changed event physics')
            changed={k for k in set(current['code'])|set(before['code']) if current['code'].get(k)!=before['code'].get(k)}
            if changed!={'utils/statistics_inputs.py','utils/statistical_templates.py'}:
                raise ValueError('Unexpected event source change in statistics adapter')
        ancestors=prior_contracts(old,True)
        return {y:[c]+(ancestors.get(y,[]) if isinstance(ancestors.get(y,[]),list) else [ancestors[y]])
                for y,c in old['contracts'].items()}
    optimization = plan.get('nonjme_throughput_update')
    if optimization:
        if plan.get('shape_mode') != 'stored_trota_nonjme' or plan.get('variations') != ['nominal']:
            raise ValueError('Throughput compatibility requires the non-JME nominal-weight campaign')
        old_path = internal_path(ROOT/optimization['previous_plan'])
        gate_path = internal_path(ROOT/optimization['validation'])
        if (sha256(old_path) != optimization['previous_plan_sha256'] or
                sha256(gate_path) != optimization['validation_sha256']):
            raise ValueError('Changed non-JME throughput evidence')
        old, gate = read_json(old_path), read_json(gate_path)
        records = gate.get('records', [])
        if (gate.get('status') != 'passed' or gate.get('unit_tests', 0) < 82 or
                len(records) < 18 or {r['year'] for r in records} != {2024,2025} or
                not all(r['all_histograms_equal'] and r['selection_weight_audits_equal'] for r in records) or
                not gate.get('three_signal_topologies_tested') or
                set(gate.get('regions', [])) != {'SR','LLCR','QCDCR','GCR','DY2E','DY2M'}):
            raise ValueError('Missing full-file non-JME throughput validation')
        if gate['before_contracts'] != old['contracts'] or gate['after_contracts'] != plan['contracts']:
            raise ValueError('Non-JME throughput contracts differ from tested sources')
        expected = {'utils/corrections.py','utils/pog_corrections.py',
                    'utils/event_selections.py','utils/histogramming.py'}
        for year, current in plan['contracts'].items():
            before = old['contracts'][year]
            if {k:v for k,v in current.items() if k!='code'} != {k:v for k,v in before.items() if k!='code'}:
                raise ValueError('Non-code physics changed in non-JME optimization')
            changed = {n for n in set(current['code'])|set(before['code'])
                       if current['code'].get(n) != before['code'].get(n)}
            if changed != expected:
                raise ValueError('Unexpected non-JME optimization source change')
        ancestors = prior_contracts(old, True)
        return {y:[c]+(ancestors.get(y,[]) if isinstance(ancestors.get(y,[]),list) else [ancestors[y]])
                for y,c in old['contracts'].items()}
    audit=plan.get('object_response_audit_update')
    if audit:
        old_path, gate_path=[internal_path(ROOT/audit[k]) for k in ('previous_plan','validation')]
        if sha256(old_path)!=audit['previous_plan_sha256'] or sha256(gate_path)!=audit['validation_sha256']:
            raise ValueError('Changed object-response compatibility evidence')
        old,gate=read_json(old_path),read_json(gate_path)
        mode=plan.get('shape_mode') or 'weights'
        if mode not in ('weights','stored_trota_nonjme') or plan.get('variations')!=old.get('variations'):
            raise ValueError('Unexpected object-response audit campaign')
        if (gate.get('status')!='passed' or gate.get('unit_tests',0)<48 or
            not gate.get('multi_dataset_files_passed') or not gate.get('old_success_histograms_equal') or
            not gate.get('weight_histograms_equal')):
            raise ValueError('Missing object-response audit validation')
        if gate['before_contracts'][mode]!=old['contracts'] or gate['after_contracts'][mode]!=plan['contracts']:
            raise ValueError('Object-response contracts differ from validation')
        for year,current in plan['contracts'].items():
            before=old['contracts'][year]
            if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in before.items() if k!='code'}:
                raise ValueError('Object-response update changed non-code physics')
            changed={n for n in set(current['code'])|set(before['code']) if current['code'].get(n)!=before['code'].get(n)}
            if changed!={'utils/histogramming.py'}:
                raise ValueError('Object-response update changed other physics sources')
        prior=prior_contracts(old,True)
        return {y:[c]+(prior.get(y,[]) if isinstance(prior.get(y,[]),list) else [prior[y]])
                for y,c in old['contracts'].items()}
    extension = plan.get('nonjme_code_extension')
    if extension:
        if plan.get('shape_mode') is not None or plan.get('variations') != ['all_weights']:
            raise ValueError('Non-JME code compatibility is scoped to the stored-TROTA weight campaign')
        old_path, gate_path = [internal_path(ROOT/extension[k]) for k in ('previous_plan','validation')]
        if sha256(old_path)!=extension['previous_plan_sha256'] or sha256(gate_path)!=extension['validation_sha256']:
            raise ValueError('Changed weight compatibility evidence')
        old, gate = read_json(old_path), read_json(gate_path)
        if (gate.get('status')!='passed' or not gate.get('stored_trota_nonjme_validation') or
                len(gate.get('records',[]))<10 or gate.get('unit_tests',0)<44 or
                set(gate.get('regions',[]))!={'SR','LLCR','QCDCR','GCR','DY2E','DY2M'} or
                not all(r['all_weight_histograms_equal'] for r in gate['records'])):
            raise ValueError('Incomplete weight-preserving real-file validation')
        if gate['baseline_weight_contracts']!=old['contracts'] or gate['tested_weight_contracts']!=plan['contracts']:
            raise ValueError('Weight compatibility contracts differ from tested implementation')
        for y,new in plan['contracts'].items():
            before=old['contracts'][y]
            if {k:v for k,v in new.items() if k!='code'}!={k:v for k,v in before.items() if k!='code'}:
                raise ValueError('Non-code weight prescription changed')
            changed={k for k in set(new['code'])|set(before['code']) if new['code'].get(k)!=before['code'].get(k)}
            if changed!={'utils/corrections.py','utils/shape_kinematics.py','utils/histogramming.py'}:
                raise ValueError('Unexpected non-JME extension source change')
        return old['contracts']
    optimization=plan.get('throughput_optimization')
    if optimization:
        if plan.get('shape_mode')!='cms_trota_jme':
            raise ValueError('Throughput validation is only for the current JME campaign')
        previous_path=internal_path(ROOT/optimization['previous_plan'])
        validation=internal_path(ROOT/optimization['validation'])
        tests=internal_path(ROOT/optimization['tests'])
        if any(sha256(path)!=optimization[field] for path,field in (
                (previous_path,'previous_plan_sha256'),(validation,'validation_sha256'),
                (tests,'tests_sha256'))):
            raise ValueError('Changed throughput validation or previous plan')
        previous=read_json(previous_path); evidence=read_json(validation); test_result=read_json(tests)
        if (evidence.get('status')!='passed' or len(evidence.get('full_files',[]))!=12 or
            {r['year'] for r in evidence['full_files']}!={2024,2025} or
            not all(r['histograms_audits_equal'] for r in evidence['full_files']) or
            test_result.get('status')!='passed' or test_result.get('tests_run',0)<64):
            raise ValueError('Missing exact full-file throughput validation')
        for year,current in plan['contracts'].items():
            old=previous['contracts'][year]
            if (evidence['baseline_contracts'][year]!=old or
                evidence['tested_contracts'][year]!=current):
                raise ValueError('Throughput contracts differ from tested files')
            if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in old.items() if k!='code'}:
                raise ValueError('Non-code prescription changed in throughput optimization')
            changed={n for n in set(current['code'])|set(old['code']) if current['code'].get(n)!=old['code'].get(n)}
            if changed!=THROUGHPUT_CHANGED:
                raise ValueError('Unexpected throughput source change')
        for name,digest in optimization['workflow_sha256'].items():
            if not _historical and sha256(ROOT/name)!=digest:
                raise ValueError('Changed throughput continuation workflow: '+name)
        ancestors=prior_contracts(previous,_historical=True)
        return {y:(ancestors[y] if isinstance(ancestors.get(y),list) else
                   [ancestors[y]] if y in ancestors else [])+[c]
                for y,c in previous['contracts'].items()}
    extension=plan.get('negative_pt_selection')
    if extension:
        if plan.get('shape_mode')!='cms_trota_jme':
            raise ValueError('Negative-pT selection bridge is only for JME')
        previous_path=internal_path(ROOT/extension['previous_plan'])
        validation=internal_path(ROOT/extension['validation'])
        if (sha256(previous_path)!=extension['previous_plan_sha256'] or
            sha256(validation)!=extension['validation_sha256']):
            raise ValueError('Changed negative-pT validation/previous plan')
        previous=read_json(previous_path); evidence=read_json(validation)
        if (evidence.get('status')!='passed' or evidence.get('unit_tests',0)<25 or
            len(evidence.get('affected_files',[]))!=4 or
            {r['year'] for r in evidence.get('full_pilot_files',[])}!={2024,2025} or
            not all(r['histograms_audits_equal'] for r in evidence['full_pilot_files'])):
            raise ValueError('Missing negative-pT successful-domain validation')
        for year,current in plan['contracts'].items():
            old=previous['contracts'][year]
            if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in old.items() if k!='code'}:
                raise ValueError('Non-code prescription changed in negative-pT selection')
            changed={n for n in set(current['code'])|set(old['code']) if current['code'].get(n)!=old['code'].get(n)}
            if changed!={'utils/trota_inference.py'}:
                raise ValueError('Unexpected negative-pT source change')
            for name in changed:
                if (evidence['previous_sources'][name]!=old['code'][name] or
                    evidence['tested_sources'][name]!=current['code'][name]):
                    raise ValueError('Negative-pT code differs from validated source')
        for name,digest in extension['workflow_sha256'].items():
            if not _historical and sha256(ROOT/name)!=digest:
                raise ValueError('Changed negative-pT continuation workflow: '+name)
        ancestors=prior_contracts(previous, _historical=True)
        return {y: ([ancestors[y]] if y in ancestors else [])+[c]
                for y,c in previous['contracts'].items()}
    spec=plan.get(FIELD)
    if not spec:return {}
    if plan.get('shape_mode')!='cms_trota_jme' or plan.get('normalization_extension_recovery'):
        raise ValueError('Policy continuation only applies to the original JME campaign')
    path=internal_path(ROOT/spec['validation'])
    if sha256(path)!=spec['validation_sha256']:
        raise ValueError('Changed input-policy validation')
    evidence=read_json(path)
    if (evidence.get('status')!='passed' or evidence.get('unit_tests',0)<22
        or {r['year'] for r in evidence['unchanged_complete_files']}!={2024,2025}
        or not all(r['all_five_endpoints_byte_equal'] for r in evidence['unchanged_complete_files'])):
        raise ValueError('Incomplete successful-domain closure')
    previous=read_json(ROOT/spec['previous_plan'])
    if sha256(ROOT/spec['previous_plan'])!=spec['previous_plan_sha256']:
        raise ValueError('Changed original frozen plan')
    if previous['contracts']!=spec['prior_contracts']:
        raise ValueError('Prior contracts not bound to original plan')
    for year,current in plan['contracts'].items():
        old=previous['contracts'][year]
        if {k:v for k,v in current.items() if k!='code'}!={k:v for k,v in old.items() if k!='code'}:
            raise ValueError('Non-code analysis prescription changed')
        changed={n for n in set(current['code'])|set(old['code']) if current['code'].get(n)!=old['code'].get(n)}
        if changed!=CHANGED:raise ValueError('Unaudited source changes in input-policy continuation')
        for name in changed:
            if (evidence['previous_sources'].get(name)!=old['code'][name]
                or evidence['tested_sources'].get(name)!=current['code'][name]):
                raise ValueError('Source differs from tested domain extension: '+name)
    for name,digest in spec['workflow_sha256'].items():
        if not _historical and sha256(ROOT/name)!=digest:raise ValueError('Changed continuation workflow: '+name)
    return spec['prior_contracts']

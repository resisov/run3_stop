"""Strict deterministic per-year merge; preserve every histogram axis.

Large per-file correction audits stay on hep2 with content hashes. Compact
summed acceptance and normalization bookkeeping accompany merged histograms.
No partial campaign can masquerade as complete or feed a production fit.
"""
import argparse
import copy
import math
import time
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.nominal_campaign import (frozen_plan, validate_payload, fingerprint, relative,
                                               execution_tasks, task_contract)


def add_numbers(a, b):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('Histogram/covariance dimensions differ')
        return [add_numbers(x, y) for x, y in zip(a, b)]
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            raise ValueError('Inconsistent boolean invariant')
        return a
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError('Nonnumeric additive field')
    if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(a+b):
        raise ValueError('Nonfinite additive field')
    return a+b


def merge_histograms(target, source):
    if 'sumw' in source:
        if not {'sumw', 'sumw2', 'entries', 'edges'} <= set(source):
            raise ValueError('Incomplete histogram leaf')
        if set(target) != set(source):
            raise ValueError('Histogram leaf schema changed')
        for key, value in source.items():
            if key in ('sumw', 'sumw2', 'entries'):
                target[key] = add_numbers(target[key], value)
            elif target[key] != value:
                raise ValueError('Histogram binning/label changed: ' + key)
        return
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_histograms(target[key], value)
        else:
            raise ValueError('Unexpected histogram node: ' + key)


def merge_counters(target, source):
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_counters(target[key], value)
        else:
            target[key] = add_numbers(target[key], value)


DATASET_SUM_FIELDS = {'events_read', 'events_written', 'files_attempted', 'files_processed',
    'sumw', 'sumw2', 'signal_event_genweight_sum_by_genmodel', 'signal_runs_sumw_source_counts',
    'signal_sumw_by_genmodel', 'sumw_source_counts'}


def merge_datasets(target, source):
    for key, record in source.items():
        if key not in target:
            target[key] = copy.deepcopy(record)
            continue
        previous = target[key]
        if set(previous) != set(record):
            raise ValueError('Dataset metadata schema differs: ' + key)
        for field, value in record.items():
            if field in DATASET_SUM_FIELDS:
                if isinstance(value, dict):
                    merge_counters(previous[field], value)
                else:
                    previous[field] = add_numbers(previous[field], value)
            elif previous[field] != value:
                raise ValueError('Dataset identity/normalization metadata differs: ' + key+'/'+field)


def assemble_endpoints(payloads, endpoints, compatible_codes=()):
    """Join disjoint variations, counting metadata and central traversal once.

    The caller supplies nominal first, and validates every shard's contract,
    input identity and content hash before yielding it. No event arrays or
    intermediate ROOT are written, and only one shard is read at a time.
    """
    result, seen = None, set()
    required = {'nominal', *endpoints}
    for payload in payloads:
        endpoint = payload['contract'].get('object_task_endpoint')
        if endpoint not in required or endpoint in seen:
            raise ValueError('Duplicate or unknown endpoint shard')
        seen.add(endpoint)
        if result is None:
            if endpoint != 'nominal':
                raise ValueError('Nominal endpoint must be assembled first')
            result = payload
            continue
        if (payload['input_metadata'] != result['input_metadata'] or
                payload['audit']['events_read'] != result['audit']['events_read']):
            raise ValueError('Endpoint input/normalization accounting mismatch')
        for field in ('schema','sr_data_blinded','bins_merged','highdm_sr_source_axes',
                      'mixed_sf','category_order'):
            if payload.get(field) != result.get(field):
                raise ValueError('Endpoint invariant differs: '+field)
        for field in ('input','fingerprint','year','code','normalization_sha256','assets_sha256',
                      'modes','variations','configuration_sha256','sr_binning_sha256','object_payloads'):
            if field=='code' and compatible_codes and all(c in compatible_codes for c in
                    (payload['contract'].get('code'),result['contract'].get('code'))):
                continue
            if payload['contract'].get(field) != result['contract'].get(field):
                raise ValueError('Endpoint contract differs: '+field)
        if payload.get('dy_rz') or payload['audit'].get('migrations'):
            raise ValueError('Shift shard repeats nominal accounting')
        for section in ('histograms','physical_histograms'):
            for mode, variations in payload.get(section, {}).items():
                if set(variations) - {endpoint}:
                    raise ValueError('Endpoint histogram contamination')
                result.setdefault(section, {}).setdefault(mode, {}).update(variations)
        if set(payload.get('dy_rz_variations',{})) - {endpoint}:
            raise ValueError('Endpoint DY contamination')
        result.setdefault('dy_rz_variations',{}).update(payload.get('dy_rz_variations',{}))
        for section in ('shape_migrations','object_endpoints','object_response','object_acceptance'):
            values = payload['audit'].get(section,{})
            if set(values) - {endpoint}:
                raise ValueError('Endpoint audit contamination')
            result['audit'].setdefault(section,{}).update(values)
        for section in ('chunks','weights'):
            values = payload['audit'].get(section,[])
            if any(v.get('endpoint','nominal') != endpoint for v in values):
                raise ValueError('Endpoint chunk/weight contamination')
            result['audit'][section].extend(values)
    if seen != required:
        raise ValueError('Incomplete independent endpoint coverage')
    return result


def merge(campaign, year):
    campaign = internal_path(campaign)
    plan = frozen_plan(campaign)
    from TROTASR.workflows.normalization_recovery import prior_contracts, payload_contract
    previous_contracts = {} if '_execution' in plan else prior_contracts(plan)
    from TROTASR.workflows.systematic_continuation import prior_contracts as policy_contracts
    if '_execution' not in plan:
        previous_contracts.update(policy_contracts(plan))
    tasks = [t for t in plan['tasks'] if t['year'] == year]
    if not tasks:
        raise ValueError('No inputs for requested year')
    results, groups = [], {}
    for task in execution_tasks(plan):
        if task['year'] != year:
            continue
        s = read_json(campaign/'state'/(task['key']+'.json'))
        if s['status'] != 'complete':
            raise ValueError('Incomplete task: ' + task['key'])
        p = internal_path(ROOT/s['output'])
        if sha256(p) != s['output_sha256'] or fingerprint(ROOT/task['input']) != task['fingerprint']:
            raise ValueError('Completed input/output changed: ' + task['key'])
        if task.get('metadata') and fingerprint(ROOT/task['metadata']) != task['metadata_fingerprint']:
            raise ValueError('Completed metadata changed: '+task['key'])
        r = dict(task=task, output=s['output'], sha256=s['output_sha256'], state=s)
        results.append(r)
        groups.setdefault(task.get('parent_key',task['key']),[]).append(r)
    provenance = dict(plan_sha256=sha256(campaign/'plan.json'), merge_code_sha256=sha256(Path(__file__)),
                      input_hashes={r['task']['key']:r['sha256'] for r in results})
    prefix = 'systematics' if plan.get('shape_mode') or plan.get('variations') == ['all_weights'] else 'nominal'
    output = campaign/'merged'/('%s_%s.json.gz' % (prefix, year))
    if output.exists():
        old = read_json(output)
        if old.get('merge_provenance') == provenance and old.get('status') == 'complete':
            return output
        raise FileExistsError('Preserve divergent existing merge')
    merged = dict(schema='trotasr_histograms_v1', status='complete', scope=plan['scope'],
        contract=plan['contracts'][str(year)], sr_data_blinded=True, bins_merged=False,
        highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2', mixed_sf='not_provided',
        histograms={}, physical_histograms={}, dy_rz={}, dy_rz_variations={}, input_metadata={},
        category_order=['Nb', 'Nbst', 'Nmix', 'Nres', 'Nw'],
        audit=dict(events_read=0, migrations={}, selection_totals={},
                   per_file_weight_audit='Retained in hashed input outputs; not discarded'),
        merge_provenance=provenance,
        coverage=dict(expected_files=len(tasks), completed_files=len(groups), failed_files=0,
            completed_execution_tasks=len(results),
            expected_events=sum(t['entries'] for t in tasks),
            full_input_inventory=plan['scope'] in ('full_nominal_production','full_cms_trota_jme_production','full_weight_systematic_production','full_nonjme_systematic_production')))
    def validated_records(records):
        for r in records:
            raw = read_json(ROOT/r['output'])
            if '_execution' in plan:
                from TROTASR.workflows.nominal_campaign import integration_payload
                yield integration_payload(raw,r['task'],plan,r['state'])
                continue
            independent = 'object_endpoint' in r['task']
            expected = payload_contract(raw, task_contract(plan,r['task']),
                None if independent else previous_contracts.get(str(year)))
            yield validate_payload(raw, r['task'], expected)
    for task in tasks:
        records = groups[task['key']]
        payloads = validated_records(records)
        if 'object_endpoint' in records[0]['task']:
            if '_execution' in plan:
                # These exact per-file payloads have each been checked above
                # against the immutable result registry and current physics.
                validated=list(payloads)
                codes=[p['contract']['code'] for p in validated]
                payload=assemble_endpoints(validated,plan['contracts'][str(year)]['object_endpoints'],codes)
            else:
                payload = assemble_endpoints(payloads, plan['contracts'][str(year)]['object_endpoints'])
        else:
            if len(records) != 1:
                raise ValueError('Duplicate full-file output')
            payload = next(payloads)
        for section in ('histograms', 'physical_histograms', 'dy_rz', 'dy_rz_variations'):
            merge_histograms(merged[section], payload.get(section, {}))
        merge_datasets(merged['input_metadata'], payload['input_metadata'])
        merged['audit']['events_read'] += payload['audit']['events_read']
        merge_counters(merged['audit']['migrations'], payload['audit']['migrations'])
        for section in ('trota_inference','shape_migrations','object_endpoints','object_response','object_acceptance'):
            if section in payload['audit']:
                merge_counters(merged['audit'].setdefault(section, {}),payload['audit'][section])
        for chunk in payload['audit']['chunks']:
            # Data-only SR is deliberately absent from this aggregation; a
            # MC SR count must never be confused with an unblinded data count.
            endpoint = chunk.get('endpoint','nominal')
            chunk = {k:v for k,v in chunk.items() if k not in ('SR','endpoint')}
            target = (merged['audit']['selection_totals'] if endpoint=='nominal' else
                      merged['audit'].setdefault('selection_totals_by_endpoint',{}).setdefault(endpoint,{}))
            merge_counters(target, chunk)
    if merged['audit']['events_read'] != merged['coverage']['expected_events']:
        raise ValueError('Merged traversal mismatch')
    write_json(output, merged)
    # Readback ensures compression/JSON integrity; hashes bind downstream.
    check = read_json(output)
    if check['merge_provenance'] != provenance:
        raise RuntimeError('Merge readback mismatch')
    write_json(output.with_name('%s_%s.summary.json' % (prefix,year)), dict(status='complete',
        scope=plan['scope'], output=relative(output), sha256=sha256(output),
        coverage=merged['coverage'], migrations=merged['audit']['migrations'],
        physics_contract=plan['contracts'][str(year)], full_workflow_complete=False))
    return output


def merge_refreshed_year(campaign):
    """Strict 2025 remerge at the original path, with unchanged data reused."""
    from TROTASR.workflows.run_systematic_campaign import (refresh_plan, REFRESH_ID,
        refresh_task_required, refreshed_population_verified, refresh_result_contract)
    campaign=internal_path(campaign);plan=refresh_plan(campaign);spec=plan['payload_refresh']
    if spec['validation'].get('status') != 'passed':
        raise ValueError('Every refreshed/rebuilt file must pass its registered validation')
    records=[]
    for task in spec['tasks']:
        if refresh_task_required(task):
            s=read_json(campaign/'state'/(task['key']+'.json'))
            if (s.get('status')!='complete' or s.get('exit_code')!=0 or
                    s.get('payload_revision')!=REFRESH_ID or not refreshed_population_verified(task,s)):
                raise ValueError('Unverified refreshed input: '+task['key'])
            record=dict(task=task,output=s['output'],sha256=s['output_sha256'])
        else:
            old=spec['previous_outputs'][task['key']]
            if len(old)!=1 or old[0]['endpoint'] is not None:
                raise ValueError('Unexpected data endpoint layout')
            record=dict(task=task,output=old[0]['output'],sha256=old[0]['sha256'])
        if sha256(ROOT/record['output'])!=record['sha256'] or fingerprint(ROOT/task['input'])!=task['fingerprint']:
            raise ValueError('Refreshed or retained input/output changed')
        records.append(record)
    merged=dict(schema='trotasr_histograms_v1',status='complete',scope=plan['scope'],
        contract=spec['contract'],sr_data_blinded=True,bins_merged=False,
        highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',mixed_sf='not_provided',
        histograms={},physical_histograms={},dy_rz={},dy_rz_variations={},input_metadata={},
        category_order=['Nb','Nbst','Nmix','Nres','Nw'],
        audit=dict(events_read=0,migrations={},selection_totals={},
            per_file_weight_audit='Retained in the canonical hashed per-file outputs'),
        merge_provenance=dict(plan_sha256=sha256(campaign/'plan.json'),
            merge_code_sha256=sha256(Path(__file__)),payload_revision=REFRESH_ID,
            input_hashes={r['task']['key']:r['sha256'] for r in records}),
        coverage=dict(expected_files=len(records),completed_files=len(records),failed_files=0,
            expected_events=sum(r['task']['entries'] for r in records),full_input_inventory=True,
            refreshed_mc_signal_files=sum(r['task']['refresh_required'] for r in records),
            rebuilt_missing_data_files=sum(not r['task']['refresh_required'] and
                r['task'].get('rebuild_missing_seed',False) for r in records),
            reused_data_files=sum(not refresh_task_required(r['task']) for r in records)))
    for r in records:
        payload=read_json(ROOT/r['output']);task=r['task']
        if refresh_task_required(task):
            validate_payload(payload,task,refresh_result_contract(spec,task,r['sha256']))
        elif (payload.get('status')!='complete_one_file_test' or
                not payload.get('sr_data_blinded') or
                not all(x.get('is_data') for x in payload['input_metadata'].values()) or
                payload['audit']['events_read']!=task['entries']):
            raise ValueError('Retained file is not validated collision data')
        for section in ('histograms','physical_histograms','dy_rz','dy_rz_variations'):
            merge_histograms(merged[section],payload.get(section,{}))
        merge_datasets(merged['input_metadata'],payload['input_metadata'])
        merged['audit']['events_read']+=payload['audit']['events_read']
        merge_counters(merged['audit']['migrations'],payload['audit']['migrations'])
        for section in ('trota_inference','shape_migrations','object_endpoints','object_response','object_acceptance'):
            if section in payload['audit']:
                merge_counters(merged['audit'].setdefault(section,{}),payload['audit'][section])
        for chunk in payload['audit']['chunks']:
            endpoint=chunk.get('endpoint','nominal')
            target=(merged['audit']['selection_totals'] if endpoint=='nominal' else
                merged['audit'].setdefault('selection_totals_by_endpoint',{}).setdefault(endpoint,{}))
            merge_counters(target,{k:v for k,v in chunk.items() if k not in ('SR','endpoint')})
    if merged['audit']['events_read']!=merged['coverage']['expected_events']:
        raise ValueError('Incomplete refreshed-year traversal')
    output=campaign/'merged/systematics_2025.json.gz'
    if output.exists():raise FileExistsError('Existing merge must be preserved before replacement')
    write_json(output,merged)
    check=read_json(output)
    if check['merge_provenance']!=merged['merge_provenance']:
        raise ValueError('Refreshed merge readback failed')
    write_json(campaign/'merged/systematics_2025.summary.json',dict(status='complete',scope=plan['scope'],
        output=relative(output),sha256=sha256(output),coverage=check['coverage'],
        physics_contract=spec['contract'],payload_revision=REFRESH_ID,full_workflow_complete=False))
    return output


COMBINATION_POLICY = dict(id='jme_with_stored_nonjme_as_is', user_approved=True,
    central='corrected_central_jme', nonjme_endpoints='stored_absolute_histograms',
    nominal_mismatch_acknowledged=True, reanchor=False, recompute_events=False)


def combine_saved_payloads(jme, nonjme, policy):
    """Explicit user-approved *as-is* union, never a hidden nominal rebasing.

    Nominal, weights, JME, data and normalization come from JME unchanged.
    The sixteen non-JME absolute endpoint populations retain their own stored
    central convention. This is preliminary and is not central equivalence.
    Inputs may be consumed in memory; persisted source files stay untouched.
    """
    from TROTASR.utils.corrections import NONJME_ENDPOINTS
    if policy != COMBINATION_POLICY:
        raise ValueError('Explicit acknowledged as-is combination policy required')
    endpoints = set(NONJME_ENDPOINTS)
    for payload, scope, mode in ((jme, 'full_cms_trota_jme_production', 'cms_trota_jme'),
            (nonjme, 'full_nonjme_systematic_production', 'stored_trota_nonjme')):
        coverage = payload.get('coverage', {})
        if (payload.get('status') != 'complete' or payload.get('scope') != scope
                or payload.get('contract', {}).get('shape_mode') != mode
                or payload.get('sr_data_blinded') is not True
                or not coverage.get('full_input_inventory') or coverage.get('failed_files') != 0
                or coverage.get('expected_files') != coverage.get('completed_files')):
            raise ValueError('Incomplete or incompatible source campaign')
    for field in ('year', 'normalization_sha256', 'assets_sha256', 'modes',
                  'configuration_sha256', 'sr_binning_sha256'):
        if jme['contract'].get(field) != nonjme['contract'].get(field):
            raise ValueError('Systematic input identity differs: '+field)
    for field in ('schema', 'bins_merged', 'highdm_sr_source_axes', 'category_order', 'mixed_sf'):
        if jme.get(field) != nonjme.get(field):
            raise ValueError('Histogram representation differs: '+field)
    # Dataset inventory and bookkeeping are not added a second time.
    mc = {k:v for k,v in jme['input_metadata'].items() if not v.get('is_data')}
    if set(mc) != set(nonjme['input_metadata']):
        raise ValueError('JME/non-JME MC dataset inventories differ')
    from TROTASR.workflows.run_systematic_campaign import compare_values
    compare_values(mc, nonjme['input_metadata'], 'MC bookkeeping')
    if set(nonjme['contract'].get('object_endpoints', [])) != endpoints:
        raise ValueError('All sixteen non-JME endpoints are required')
    for section in ('histograms', 'physical_histograms'):
        if set(jme[section]) != {'highdm', 'lowdm'} or set(nonjme[section]) != {'highdm', 'lowdm'}:
            raise ValueError('Both histogram regimes required')
        for mode in ('highdm', 'lowdm'):
            source = nonjme[section][mode]
            target = jme[section][mode]
            if set(source) != {'nominal', *endpoints} or endpoints & set(target):
                raise ValueError('Missing or overlapping systematic endpoints')
            if not {'nominal', 'jesTotalUp', 'jesTotalDown', 'jerUp', 'jerDown'} <= set(target):
                raise ValueError('Missing original JME/nominal endpoints')
            for endpoint in sorted(endpoints):
                for categories in source[endpoint].values():
                    for samples in categories.values():
                        if {'data', 'data_obs'} & set(samples):
                            raise ValueError('Non-JME variation contains collision data')
                target[endpoint] = source[endpoint]
    # On/off-Z endpoint records use endpoint as their outer key.
    varied = nonjme.get('dy_rz_variations', {})
    if set(varied) - endpoints or set(varied) & set(jme.get('dy_rz_variations', {})):
        raise ValueError('Overlapping or unknown DY endpoint records')
    jme.setdefault('dy_rz_variations', {}).update(varied)
    original_contract = jme['contract']
    jme['contract'] = dict(original_contract, shape_mode='jme_with_stored_nonjme',
        object_endpoints=list(NONJME_ENDPOINTS), combination_policy=dict(policy),
        component_contracts=dict(jme=original_contract, nonjme=nonjme['contract']))
    jme.update(scope='full_combined_systematic_production', preliminary=True,
        systematic_scope='JME central/weights/JES/JER plus stored-TROTA non-JME16 absolute endpoints; unequal nominal conventions acknowledged by user',
        central_conventions_consistent=False)
    jme['audit']['nonjme_source_events_read'] = nonjme['audit']['events_read']
    jme['audit']['nonjme_source_coverage'] = nonjme['coverage']
    return jme


def combine(campaign, other_campaign, year, output, policy):
    """Bind strict completed year merges and preserve both original campaigns."""
    campaign, other_campaign, output = map(internal_path, (campaign, other_campaign, output))
    sources = {}
    for name, folder in (('jme', campaign), ('nonjme', other_campaign)):
        path = folder/'merged'/('systematics_%s.json.gz' % year)
        summary = read_json(path.with_name('systematics_%s.summary.json' % year))
        digest = sha256(path)
        if summary.get('status') != 'complete' or summary.get('sha256') != digest:
            raise ValueError('Strict year merge is absent or changed: '+name)
        sources[name] = dict(path=relative(path), sha256=digest,
                             summary_sha256=sha256(path.with_name('systematics_%s.summary.json' % year)))
    provenance = dict(sources=sources, policy=policy, merge_code_sha256=sha256(Path(__file__)))
    if output.exists():
        old = read_json(output)
        if old.get('combination_provenance') != provenance or old.get('status') != 'complete':
            raise FileExistsError('Preserve existing different combined product')
        return output
    joined = combine_saved_payloads(*(read_json(ROOT/sources[n]['path']) for n in ('jme', 'nonjme')), policy)
    joined['combination_provenance'] = provenance
    write_json(output, joined)
    del joined
    check = read_json(output)
    if check.get('combination_provenance') != provenance:
        raise RuntimeError('Combined histogram readback failed')
    write_json(output.with_name('systematics_%s.summary.json' % year),
        dict(status='complete', scope=check['scope'], output=relative(output), sha256=sha256(output),
             coverage=check['coverage'], combination_provenance=provenance, completed=time.time(),
             preliminary=True, central_conventions_consistent=False, full_workflow_complete=False))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--year', type=int, required=True, choices=(2024, 2025))
    p.add_argument('--combine-with', type=Path, help='Complete non-JME campaign for an explicitly approved as-is union')
    p.add_argument('--config', type=Path, help='Configuration containing the exact systematic_combination policy')
    p.add_argument('--output', type=Path, help='New combined year histogram file; original results are preserved')
    a = p.parse_args()
    if a.combine_with:
        if not a.config or not a.output:
            p.error('--combine-with requires --config and --output')
        print(combine(a.campaign, a.combine_with, a.year, a.output,
                      read_json(internal_path(a.config))['systematic_combination']))
    else:
        if a.config or a.output:
            p.error('--config and --output are only for explicit combination')
        print(merge(a.campaign, a.year))

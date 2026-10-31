"""Mandatory common Top sample policy applied to saved histograms.

Frozen event workers and their source contracts are never changed. Only mixed
files need the existing histogram producer again, restricted to the explicitly
excluded dataset IDs after the normal Events/TopWTruth alignment has run.
"""
import copy
import fcntl
import math
import os
from pathlib import Path
import time

from TROTASR.utils.paths import ROOT, ERAS, EARLY_ERAS, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

LEGACY_POLICY = dict(id='rare_top_20261022', user_approved=True,
    excluded_primaries=['TTWH', 'TTZH'],
    top_primaries=['TTW-WtoQQ-1Jets', 'TTZ-ZtoQQ-1Jets', 'TTWW', 'TTWZ', 'TTZZ'],
    tttt='unchanged', years=list(EARLY_ERAS))
POLICY = dict(LEGACY_POLICY, id='common_top_20261027',
    top_primaries=LEGACY_POLICY['top_primaries']+['TTTT'], tttt='Top', years=list(ERAS))
SECTIONS = ('histograms', 'physical_histograms', 'dy_rz', 'dy_rz_variations')


def primary(record):
    return record['dataset'].lstrip('/').split('/')[0].split('_Tune')[0]


def excluded_ids(metadata):
    result = {str(k) for k, r in metadata.items() if primary(r) in POLICY['excluded_primaries']}
    if any(metadata[k].get('is_data') or metadata[k].get('is_signal') for k in result):
        raise ValueError('Excluded primary is not background MC')
    for r in metadata.values():
        if r.get('process') == 'TTV' and primary(r) not in POLICY['excluded_primaries'] + POLICY['top_primaries']:
            raise ValueError('Unknown TTV primary; explicit sample classification required')
    return result


def require_clean(payload):
    """No excluded primary or unnormalized rare-Top label reaches consumers."""
    metadata = payload.get('input_metadata', {})
    selected = excluded_ids(metadata)
    # Frozen, already-approved nominal products retain original inventory in
    # provenance. Accept only their explicit exact-subtraction evidence.
    evidence = payload.get('background_policy', {})
    if selected:
        recorded = {str(k) for row in evidence.get('excluded_files', [])
                    for k in row.get('datasets', {})}
        if evidence.get('policy') not in (POLICY, LEGACY_POLICY) or selected != recorded:
            raise ValueError('TTWH/TTZH require canonical sample exclusion before analysis')
    for section in ('histograms', 'physical_histograms'):
        for modes in payload.get(section, {}).values():
            for regions in modes.values():
                for categories in regions.values():
                    for samples in categories.values():
                        if {'TTV', 'TTTT', 'TTWH', 'TTZH'} & set(samples):
                            raise ValueError('Common Top policy not applied to histogram samples')
    return payload


def needed(campaign, year):
    """Policy is default, never dependent on an optional campaign sidecar."""
    plan = read_json(campaign/'plan.json')
    output = merged_output(campaign, year, plan)
    if not output.exists():
        return True
    payload = read_json(output)
    if (campaign/'background_policy'/str(year)/'receipt.json').exists():
        return True
    return bool(excluded_ids(payload.get('input_metadata', {}))) or any(
        {'TTV', 'TTTT'} & set(samples)
        for section in ('histograms', 'physical_histograms')
        for modes in payload.get(section, {}).values()
        for regions in modes.values() for categories in regions.values()
        for samples in categories.values())


def canonical_merge(campaign, year, merge_function):
    campaign = internal_path(campaign)
    if needed(campaign, year):
        return run(campaign, year, merge_function)
    # A clean preserved late-era merge is not recalculated for a code change.
    output = merged_output(campaign, year, read_json(campaign/'plan.json'))
    summary = read_json(output.with_name(output.name.replace('.json.gz', '.summary.json')))
    if summary.get('status') != 'complete' or summary.get('sha256') != sha256(output):
        raise ValueError('Clean saved merge has no matching completion receipt')
    require_clean(read_json(output))
    return output


def subtract_values(a, b, field):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError('Excluded histogram dimension differs')
        return [subtract_values(x, y, field) for x, y in zip(a, b)]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)):
        raise ValueError('Invalid excluded histogram value')
    value = a - b
    if field == 'entries':
        if value < 0 or value != int(value):
            raise ValueError('Excluded entries exceed saved entries')
        return int(value)
    if field == 'sumw2' and value < 0:
        if abs(value) > 1e-10 * max(1., abs(a), abs(b)):
            raise ValueError('Excluded variance exceeds saved variance')
        return 0.
    return value


def subtract_histograms(target, source):
    if 'sumw' in source:
        if set(target) != set(source):
            raise ValueError('Excluded histogram schema differs')
        for key, value in source.items():
            if key in ('sumw', 'sumw2', 'entries'):
                target[key] = subtract_values(target[key], value, key)
            elif target[key] != value:
                raise ValueError('Excluded histogram axes differ')
        return
    for key, value in source.items():
        if key not in target:
            raise ValueError('Excluded component absent from original merge: '+key)
        subtract_histograms(target[key], value)


def regroup(payload):
    from TROTASR.workflows.merge_nominal import merge_histograms
    for section in ('histograms', 'physical_histograms'):
        for modes in payload[section].values():
            for regions in modes.values():
                for categories in regions.values():
                    for samples in categories.values():
                        for name in ('TTV', 'TTTT'):
                            rare = samples.pop(name, None)
                            if rare is not None:
                                if 'TT' in samples:
                                    merge_histograms(samples['TT'], rare)
                                else:
                                    samples['TT'] = rare


def reader_for_ids(base_reader, selected, expected_entries):
    """Filter only after the original aligned reader, never alter input ROOT."""
    import numpy as np

    class DatasetReader(base_reader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.source_entries = self.num_entries
            self.num_entries = expected_entries

        def iterate(self, branches, *, step_size, library='ak'):
            observed = 0
            for start in range(0, self.source_entries, step_size):
                chunk = self.arrays(branches, entry_start=start,
                    entry_stop=min(start + step_size, self.source_entries), library=library)
                chosen = np.isin(np.asarray(chunk['dataset_id']), list(map(int, selected)))
                observed += int(chosen.sum())
                if chosen.any():
                    yield chunk[chosen]
            if observed != expected_entries:
                raise ValueError('Excluded dataset Events count differs from retained metadata')

    return DatasetReader


def projection_options(contract, plan=None):
    """Replay only excluded rows with the saved production endpoint contract."""
    shape = contract.get('shape_mode')
    if shape is None:
        return dict(chunk_size=2000)
    if (plan is None or shape not in ('cms_trota_jme', 'stored_trota_nonjme')
            or plan.get('shape_mode') != shape or 'object_task_endpoint' in contract):
        raise ValueError('Full saved systematic-file contract required')
    execution = contract.get('execution', {})
    options = dict(modes=tuple(contract['modes']), variations=tuple(contract['variations']),
        shape_mode=shape, chunk_size=execution.get('chunk_size', plan['chunk_size']),
        candidate_budget=plan.get('candidate_budget', 150000))
    if shape == 'stored_trota_nonjme':
        endpoints = execution.get('endpoints')
        if (not endpoints or endpoints[0] != 'nominal'
                or list(endpoints[1:]) != contract.get('object_endpoints')):
            raise ValueError('Saved non-JME endpoint inventory differs')
        options['object_endpoints'] = tuple(endpoints)
        options['reference_execution'] = execution.get('branch_selection') == 'all_reference'
    elif execution:
        if execution != dict(inference_support='all_region_pretopology_union', physics_definitions_changed=False):
            raise ValueError('Unknown saved JME execution contract')
        options['support_only'] = True
    return options


def merged_output(campaign, year, plan):
    shape = plan.get('shape_mode')
    if shape not in (None, 'cms_trota_jme', 'stored_trota_nonjme'):
        raise ValueError('Unsupported campaign shape mode')
    stem = 'systematics_' if shape else 'nominal_'
    return campaign/'merged'/(stem+str(year)+'.json.gz')


def recover_excluded(task, original, selected, folder, campaign=None):
    from TROTASR.utils import histogramming
    from TROTASR.workflows.nominal_campaign import fingerprint
    source = ROOT/task['input']
    metadata = original['input_metadata']
    expected = sum(metadata[k]['events_written'] for k in selected)
    output = folder/(task['key']+'.json.gz')
    receipt = folder/(task['key']+'.receipt.json')
    contract = dict(policy=POLICY, task=task['key'], selected_dataset_ids=sorted(selected),
        original_contract=original['contract'], expected_entries=expected,
        adapter_sha256=sha256(Path(__file__)))
    if receipt.exists():
        old = read_json(receipt)
        if old.get('contract') == contract and old.get('status') == 'complete' and sha256(output) == old['sha256']:
            return read_json(output), old
        raise ValueError('Divergent excluded-dataset recovery')
    if output.exists():
        raise FileExistsError('Interrupted recovery needs inspection; do not repeat silently')
    if fingerprint(source) != task['fingerprint']:
        raise ValueError('Mixed input changed')
    plan = read_json(campaign/'plan.json') if campaign is not None else None
    options = projection_options(original['contract'], plan)
    before = histogramming.TopWEvents
    histogramming.TopWEvents = reader_for_ids(before, selected, expected)
    started = time.time()
    print('projecting excluded datasets only:', task['key'], sorted(selected), expected, flush=True)
    try:
        if campaign is None:
            result = histogramming.build(source, task['year'], output, **options)
            expected_contract, historical = original['contract'], None
        else:
            from TROTASR.workflows.btag_zero_recovery import historical_btag_projection
            with historical_btag_projection(campaign, original['contract']) as (expected_contract, historical):
                result = histogramming.build(source, task['year'], output, **options)
    finally:
        histogramming.TopWEvents = before
    if result['contract'] != expected_contract or result['audit']['events_read'] != expected:
        raise ValueError('Mixed recovery altered physics contract or event coverage')
    result['input_metadata'] = {k: metadata[k] for k in sorted(selected)}
    result['dataset_projection'] = contract
    if historical:
        result['dataset_projection_execution'] = historical
    write_json(output, result)
    proof = dict(status='complete', contract=contract, output=str(output.relative_to(ROOT)),
        sha256=sha256(output), started=started, finished=time.time(),
        input_root_unchanged=fingerprint(source) == task['fingerprint'])
    write_json(receipt, proof)
    return result, proof


def run(campaign, year, merge_function):
    """Record actual process identity and terminal exit, with shared admission."""
    import sys
    from TROTASR.workflows.diagnostic_common import live_resources
    from TROTASR.workflows.normalization_recovery import process_identity
    campaign = internal_path(campaign)
    folder = campaign/'background_policy'/str(year)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        slots, memory = live_resources()
        if slots > 120 or memory < 36 * 1024**3:
            raise RuntimeError('Shared 120-slot/32-GiB reserve admission unavailable')
        record = dict(status='running', identity=process_identity(os.getpid()),
            command=sys.argv, started=time.time(), slots_including_self=slots,
            memory_available_bytes=memory, policy=POLICY)
        path = folder/'execution.json'
        if path.exists():
            previous = read_json(path)
            write_json(folder/('execution_history_'+str(time.time_ns())+'.json'), previous)
        write_json(path, record)
        try:
            output = apply(campaign, year, merge_function)
            record.update(status='complete', exit_code=0, output=str(output.relative_to(ROOT)), sha256=sha256(output))
            return output
        except BaseException as error:
            record.update(status='failed', exit_code=1, error=repr(error))
            raise
        finally:
            record['finished'] = time.time()
            write_json(path, record)


def apply(campaign, year, merge_function):
    """One canonical revised merge; original merge and measurements retained."""
    from TROTASR.workflows.nominal_campaign import frozen_plan
    campaign = internal_path(campaign)
    policy_path = campaign/'background_policy.json'
    if str(year) not in ERAS:
        raise ValueError('Unsupported era')
    if policy_path.exists() and read_json(policy_path) not in (POLICY, LEGACY_POLICY):
        raise ValueError('Divergent background sample policy')
    if not policy_path.exists():
        write_json(policy_path, POLICY)
    folder = campaign/'background_policy'/str(year)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = frozen_plan(campaign)  # all active event code remains unchanged
        output = merged_output(campaign, year, plan)
        summary = output.with_name(output.name.replace('.json.gz', '.summary.json'))
        receipt = folder/'receipt.json'
        policy_contract = dict(policy=POLICY, adapter_sha256=sha256(Path(__file__)),
            plan_sha256=sha256(campaign/'plan.json'))
        if receipt.exists():
            if 'btag_zero_denominator_recovery' in plan:
                from TROTASR.workflows.btag_zero_recovery import reuse_unchanged_merge
                reused = reuse_unchanged_merge(campaign, plan, year, receipt)
                if reused is not None:
                    return reused
            old = read_json(receipt)
            if (old.get('status') == 'complete' and old.get('contract') == policy_contract
                    and sha256(output) == old['output_sha256']
                    and sha256(ROOT/old['original_output']) == old['original_sha256']):
                return output
            raise ValueError('Revised rare-top merge changed')
        if not output.exists():
            merge_function(campaign, year)
        raw_sha = sha256(output)
        original = read_json(output)
        if (read_json(summary)['sha256'] != raw_sha or original['status'] != 'complete'
                or original['merge_provenance']['plan_sha256'] != policy_contract['plan_sha256']
                or original.get('background_policy')):
            raise ValueError('Original canonical merge is not verified')
        # Recheck the full saved-output inventory; queue disappearance is irrelevant.
        for key, digest in original['merge_provenance']['input_hashes'].items():
            state = read_json(campaign/'state'/(key+'.json'))
            if state['status'] != 'complete' or state['output_sha256'] != digest or sha256(ROOT/state['output']) != digest:
                raise ValueError('Original per-file result changed: '+key)
        # Systematic payloads are large. The original file remains immutable;
        # modify its private in-memory copy and publish only after readback.
        revised = original if plan.get('shape_mode') else copy.deepcopy(original)
        records, removed_events = [], 0
        removed_ids = excluded_ids(original['input_metadata'])
        affected_groups = {'background_'+original['input_metadata'][k]['process'] for k in removed_ids}
        for task in plan['tasks']:
            if str(task['year']) != str(year) or not affected_groups.intersection(task.get('groups', [])):
                continue
            state = read_json(campaign/'state'/(task['key']+'.json'))
            part = read_json(ROOT/state['output'])
            selected = excluded_ids(part['input_metadata'])
            if not selected:
                continue
            if selected == set(part['input_metadata']):
                excluded = part
                proof = dict(mode='reuse_saved_file', output=state['output'], sha256=state['output_sha256'])
            else:
                excluded, proof = recover_excluded(task, part, selected, folder/'mixed', campaign)
                proof = dict(proof, mode='exact_excluded_dataset_projection')
            for section in SECTIONS:
                subtract_histograms(revised.setdefault(section, {}), excluded.get(section, {}))
            removed_events += excluded['audit']['events_read']
            records.append(dict(task=task['key'], datasets={k:part['input_metadata'][k]['dataset'] for k in selected},
                removed_events=excluded['audit']['events_read'], proof=proof))
        excluded_ids(original['input_metadata'])  # reject unexpected TTV before mapping
        regroup(revised)
        revised['input_metadata'] = {k: dict(v, process='TT') if primary(v) in POLICY['top_primaries'] else v
            for k,v in original['input_metadata'].items() if k not in removed_ids}
        original_path = folder/('original_systematics.json.gz' if plan.get('shape_mode') else 'original_nominal.json.gz')
        revised['background_policy'] = dict(policy_contract, original_output=str(original_path.relative_to(ROOT)),
            original_sha256=raw_sha, excluded_events=removed_events, excluded_files=records,
            retained_events=original['audit']['events_read']-removed_events,
            normalization_denominators_unchanged=True, tttt_group='Top',
            production_coverage_is_original_inventory=True)
        require_clean(revised)
        candidate = folder/('revised_systematics.json.gz' if plan.get('shape_mode') else 'revised_nominal.json.gz')
        write_json(candidate, revised)
        policy_evidence = revised['background_policy']
        if plan.get('shape_mode'):
            del revised, original
            import gc
            gc.collect()
        if read_json(candidate)['background_policy'] != policy_evidence:
            raise ValueError('Revised histogram readback failed')
        # Measurements tied to the original content cannot be reused after exclusion.
        measured = ROOT/'estimations'/campaign.name/str(year)
        if measured.exists():
            manifest = read_json(measured/'measurement_manifest.json')
            if manifest['provenance']['hist_input_sha256'] != raw_sha:
                raise ValueError('Divergent background measurements; preserve without moving')
            measured.rename(folder/'original_measurements')
        output.rename(original_path)
        summary.rename(folder/'original_summary.json')
        candidate.rename(output)
        final_sha = sha256(output)
        old_summary = read_json(folder/'original_summary.json')
        write_json(summary, dict(old_summary, sha256=final_sha, background_policy=policy_evidence))
        write_json(receipt, dict(status='complete', contract=policy_contract, output=str(output.relative_to(ROOT)),
            output_sha256=final_sha, original_output=str(original_path.relative_to(ROOT)), original_sha256=raw_sha,
            excluded_events=removed_events, files=records, finished=time.time(), tttt_group='Top'))
        return output

"""Quantitative representative-input gate, not a sensitivity/limit claim."""
import argparse
import math
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.signal_models import signal_topology
from TROTASR.utils.sr_binning import project
from TROTASR.workflows.nominal_campaign import frozen_plan, validate_payload, relative
from TROTASR.workflows.merge_nominal import merge_counters


def sum_sr(payload, field):
    return sum(sum(rec['recoil'][field])
        for samples in payload['histograms'].get('highdm', {}).get('nominal', {}).get('SR', {}).values()
        for rec in samples.values())


def validate(campaign, output):
    campaign, output = internal_path(campaign), internal_path(output)
    plan = frozen_plan(campaign)
    definition = read_json(ROOT/'jsons/highdm_sr_binning.json')
    expected = {'data:EGamma', 'data:JetMET', 'data:Muon', 'mc:ST', 'mc:DY',
                'signal:T2tt', 'signal:T2tb', 'signal:T2bW'}
    coverage = {str(y): set() for y in (2024,2025)}
    migrations, records, issues = {}, [], []
    for t in plan['tasks']:
        state = read_json(campaign/'state'/(t['key']+'.json'))
        if state['status'] != 'complete':
            raise ValueError('Representative input not complete: ' + t['key'])
        p = ROOT/state['output']
        if sha256(p) != state['output_sha256']:
            raise ValueError('Changed validation output: ' + t['key'])
        payload = validate_payload(read_json(p), t, plan['contracts'][str(t['year'])])
        observed = set()
        for d in payload['input_metadata'].values():
            tag = ('data:'+d['process'] if d['is_data'] else
                   'signal:'+signal_topology(d['dataset']) if d['is_signal'] else 'mc:'+d['process'])
            observed.add(tag)
        coverage[str(t['year'])].update(observed)
        projected, layout = project(payload, definition)
        if sum(len(c['edges'])-1 for c in layout) != 162:
            raise ValueError('Representative test reduced full SR map')
        for field in ('sumw', 'sumw2', 'entries'):
            if not math.isclose(sum_sr(payload,field), sum_sr(projected,field), rel_tol=1e-11, abs_tol=1e-9):
                raise ValueError('SR162 projection lost yield/variance: '+t['key']+'/'+field)
        year_migrations = migrations.setdefault(str(t['year']), {})
        for sample, by_region in payload['audit']['migrations'].items():
            topology = 'T2bW' if sample.startswith('T2bW_') else 'T2tb' if sample.startswith('T2tb_') else 'T2tt' if sample.startswith('mStop') else sample
            merge_counters(year_migrations.setdefault(topology, {}), by_region)
        records.append(dict(key=t['key'], coverage=sorted(observed), entries=t['entries'],
            seconds=payload['seconds'], input_output_sha256=state['output_sha256'],
            full_162_projection_preserved=True, high_low_intersection=0))
    for year, found in coverage.items():
        if expected-found:
            issues.append(dict(year=year, missing_representative_processes=sorted(expected-found)))
    for year, by_sample in migrations.items():
        for sample, regions in by_sample.items():
            for region, regimes in regions.items():
                for regime, row in regimes.items():
                    if row['after_entries'] != row['before_entries'] + row['gained'] - row['lost']:
                        issues.append(dict(year=year, sample=sample, region=region, regime=regime,
                                           problem='migration_count_identity_failed'))
    result = dict(schema='trotasr_representative_gate_v1', status='passed' if not issues else 'failed',
        campaign=relative(campaign), plan_sha256=sha256(campaign/'plan.json'),
        contracts=plan['contracts'], coverage={y:sorted(v) for y,v in coverage.items()},
        three_signal_topologies_tested=all({'signal:T2tt','signal:T2tb','signal:T2bW'}<=v for v in coverage.values()),
        records=records, migrations=migrations, issues=issues,
        interpretation='Validated implemented user-requested nominal selections and full bin map; '
                       'acceptance changes are recorded, not a sensitivity or baseline-equivalence claim.',
        original_inputs_modified=False, sr_data_blinded=True, mixed_sf='not_provided',
        full_campaign_complete=False, limits_executed=False)
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=validate(a.campaign,a.output)
    print(result['status'],result['coverage'])
    raise SystemExit(0 if result['status']=='passed' else 2)

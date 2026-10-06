"""Build native TH1 templates from internal histograms and measured factors."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble, write_templates, verify_templates, json_ready


def prepare(hists, sgamma, rz_high, rz_low, double_ratio, topology, mstop, mlsp):
    paths = {name: internal_path(path) for name, path in
             zip(('hists','sgamma','rz_high','rz_low','double_ratio'),(hists,sgamma,rz_high,rz_low,double_ratio))}
    digests = {name: sha256(path) for name,path in paths.items()}
    inputs = {name: read_json(path) for name,path in paths.items()}
    for name in ('sgamma','rz_high','rz_low','double_ratio'):
        if inputs[name].get('provenance', {}).get('hist_input_sha256') != digests['hists']:
            raise ValueError('Background measurement does not match the supplied histograms: ' + name)
    contract = inputs['hists'].get('contract', {})
    for key,path in [('sr_binning_sha256',ROOT/'jsons/highdm_sr_binning.json'),
                     ('configuration_sha256',ROOT/'gnn4lowdm/config.json')]:
        if contract.get(key) != sha256(path):
            raise ValueError('Histogram binning/configuration contract mismatch: ' + key)
    payload = build_inputs(*(inputs[k] for k in ('hists','sgamma','rz_high','rz_low','double_ratio')),
                           topology, mstop, mlsp)
    channels, data, clipping, issues = assemble([payload])
    issues = list(payload['issues']) + issues
    for channel in channels:
        if not any(c == channel for c, _ in data):
            issues.append(dict(problem='channel_without_processes', channel=channel))
    if not any(r['family'] == 'signal' and r['nominal'].sum() > 0 for r in data.values()):
        issues.append(dict(problem='no_positive_signal_template', topology=topology, mStop=mstop, mLSP=mlsp))
    provenance = {name:dict(path=str(path.relative_to(ROOT)),sha256=digests[name]) for name,path in paths.items()}
    return payload, channels, data, clipping, issues, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('hists','sgamma','rz-high','rz-low','double-ratio','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--topology', choices=('T2tt','T2bW','T2tb'), required=True)
    p.add_argument('--mstop', type=int, required=True)
    p.add_argument('--mlsp', type=int, required=True)
    p.add_argument('--validate-only', action='store_true', help='Validate model inputs; never report ROOT templates as complete')
    a = p.parse_args()
    output = internal_path(a.output)
    if not a.validate_only and sys.platform == 'darwin':
        p.error('ROOT products must stay on approved remote scratch, never on the local laptop')
    payload, channels, data, clipping, issues, inputs = prepare(a.hists,a.sgamma,a.rz_high,a.rz_low,a.double_ratio,
                                                             a.topology,a.mstop,a.mlsp)
    source_files = ('utils/statistics_inputs.py','utils/statistics_model.py','utils/statistical_templates.py',
                    'utils/sr_binning.py','utils/renderers/signal_theory.py','workflows/build_template.py')
    contract = dict(inputs=inputs, topology=a.topology,mStop=a.mstop,mLSP=a.mlsp,
                    code={n:sha256(ROOT/n) for n in source_files}, validate_only=a.validate_only)
    manifest_path = output / 'model_manifest.json'
    if output.exists():
        if manifest_path.is_file():
            previous=read_json(manifest_path)
            reusable = ('validated_inputs_not_root_templates' if a.validate_only else 'templates_ready')
            if previous.get('contract')==contract and previous.get('status')==reusable:
                for name,digest in previous['files'].items():
                    if sha256(output/name)!=digest:
                        raise ValueError('Completed template file changed: '+name)
                if not a.validate_only:
                    verify_templates(output,channels,data)
                print(reusable+' (validated existing outputs)')
                return
        raise FileExistsError('Different/incomplete existing output; preserve it and select a new directory: '+str(output))
    output.mkdir(parents=True)
    write_json(output/'statistics_inputs.json.gz',json_ready(payload))
    write_json(output/'negative_to_zero_audit.json',clipping)
    write_json(output/'template_issues.json',issues)
    status='blocked' if issues else 'validated_inputs_not_root_templates' if a.validate_only else 'templates_ready'
    if not issues and not a.validate_only:
        write_json(manifest_path,dict(schema='trotasr_templates_v1',status='writing_templates',contract=contract,
                                     root_integrity_checked=False,full_workflow_complete=False))
        try:
            write_templates(output,channels,data)
            verify_templates(output,channels,data)
        except Exception as exc:
            write_json(manifest_path,dict(schema='trotasr_templates_v1',status='failed',contract=contract,
                error=dict(type=type(exc).__name__,message=str(exc)),root_integrity_checked=False,
                full_workflow_complete=False))
            raise
    files={path.name:sha256(path) for path in output.iterdir() if path.is_file() and path != manifest_path}
    write_json(manifest_path,dict(schema='trotasr_templates_v1',status=status,contract=contract,
        year=payload['year'],scope=payload['scope'],bins=payload['bin_counts'],
        channels={c:[b['name'] for b in bins] for c,bins in channels.items()},
        auto_mc_stats=[10,1,1],sr_data_blinded=True,files=files,
        full_workflow_complete=False,root_integrity_checked=status=='templates_ready'))
    print(status, 'analysis bins=',sum(payload['bin_counts'].values()), 'issues=',len(issues))
    if issues:
        raise SystemExit(2)


if __name__=='__main__': main()

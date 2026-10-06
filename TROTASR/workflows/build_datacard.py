"""Create one- or two-year cards from verified internal native TH1 templates."""
import argparse
import os
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.statistical_templates import (assemble, card_text, template_name,
    verify_templates, load_theory, profiled_theory, add_signal_xsec_lnN)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--templates',nargs='+',type=Path,required=True,help='One completed per-year template directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cr-only',action='store_true')
    a=p.parse_args()
    output=internal_path(a.output)
    payloads, directories, origins = [], {}, []
    for folder in a.templates:
        folder=internal_path(folder)
        manifest=read_json(folder/'model_manifest.json')
        if manifest.get('status')!='templates_ready' or not manifest.get('root_integrity_checked'):
            raise ValueError('Real ROOT templates have not passed integrity checks: '+str(folder))
        for name,digest in manifest['files'].items():
            if sha256(folder/name)!=digest:
                raise ValueError('Template input changed: '+name)
        payload=read_json(folder/'statistics_inputs.json.gz')
        if payload['status']!='complete' or payload['year'] in directories:
            raise ValueError('Blocked input or duplicated year')
        channels,data,_,issues=assemble([payload])
        if issues:
            raise ValueError('Unsupported TH1 component')
        verify_templates(folder,channels,data)
        directories[payload['year']]=folder
        payloads.append(payload)
        origins.append(dict(path=str(folder.relative_to(ROOT)),sha256=sha256(folder/'model_manifest.json')))
    if not 1<=len(payloads)<=2 or len({(p['topology'],p['mStop'],p['mLSP']) for p in payloads})!=1:
        raise ValueError('Combine only matching mass points from distinct 2024/2025 years')
    channels,data,_,issues=assemble(payloads)
    if issues:
        raise ValueError('Combined template model is not supported')
    text=card_text(channels,data,cr_only=a.cr_only)
    lines=[]
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0]=='shapes':
            channel=fields[2]
            year=channel.rsplit('_',1)[-1]
            path=internal_path(directories[year]/template_name(channel))
            relative=os.path.relpath(path,output.parent)
            if any(c.isspace() for c in relative):
                raise ValueError('Combine template paths must not contain whitespace')
            fields[3]=relative
            line=' '.join(fields)
        lines.append(line)
    text='\n'.join(lines)+'\n'
    theory_path=ROOT/'stats/stop_xsec_13p6TeV.json'
    if not a.cr_only:
        theory=profiled_theory(load_theory(theory_path))
        text=add_signal_xsec_lnN(text,payloads[0]['mStop'],theory)
    if output.exists():
        if output.read_text()!=text:
            raise FileExistsError('Refusing to overwrite a divergent card: '+str(output))
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            stream.write(text)
    manifest=dict(status='card_written_not_fitted',inputs=origins,
        card_sha256=sha256(output),years=sorted(directories),topology=payloads[0]['topology'],
        mStop=payloads[0]['mStop'],mLSP=payloads[0]['mLSP'],cr_only=a.cr_only,
        theory_sha256=None if a.cr_only else sha256(theory_path),auto_mc_stats=[10,1,1],
        sr_data_blinded=True,full_workflow_complete=False)
    manifest_path=internal_path(output.with_suffix('.manifest.json'))
    if manifest_path.exists() and read_json(manifest_path)!=manifest:
        raise FileExistsError('Refusing to overwrite divergent card provenance: '+str(manifest_path))
    if not manifest_path.exists():
        write_json(manifest_path,manifest)
    print('card_written_not_fitted',output)


if __name__=='__main__': main()

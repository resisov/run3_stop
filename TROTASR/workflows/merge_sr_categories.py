"""Apply an audited, lossless SR category union to completed native templates.

No event reread, correction evaluation, clipping change, or fitted value reuse.
The source grid is immutable. All924 signal shapes and every nuisance endpoint
are projected with the same map; CR and low-dM objects are unchanged.
"""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.sr_merge import load_merge, bin_groups, project
from TROTASR.workflows.combine_limits import IntegrityCache, verify_card, runtime
from TROTASR.workflows.build_nominal_grid import write_card_once


def transformed(values, channel, groups):
    return project(values, groups) if channel.startswith('SR_highdm_') else np.asarray(values, dtype=float)


def rewrite_card(text, old_dir, source_templates, target_templates, target_dir):
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'shapes':
            source = internal_path(old_dir / fields[3])
            if source.parent != source_templates:
                raise ValueError('Unexpected template source')
            fields[3] = os.path.relpath(target_templates/source.name, target_dir)
            line = ' '.join(fields)
        lines.append(line)
    return '\n'.join(lines)+'\n'


def merge_root(source, target, old_channels, new_channels, groups):
    import ROOT as R
    src, dst = R.TFile.Open(str(source)), R.TFile.Open(str(target), 'CREATE')
    if not src or src.IsZombie() or not dst or dst.IsZombie():
        raise OSError('Cannot open native merge files')
    count = 0
    try:
        for ck in src.GetListOfKeys():
            c = ck.GetName()
            if c not in old_channels or not ck.GetClassName().startswith('TDirectory'):
                raise ValueError('Unexpected native ROOT directory')
            directory = src.GetDirectory(c)
            dest = dst.mkdir(c)
            for key in directory.GetListOfKeys():
                h = key.ReadObj()
                if not h.InheritsFrom('TH1') or h.GetDimension()!=1 or h.GetNbinsX()!=len(old_channels[c]):
                    raise ValueError('Unexpected native histogram')
                labels = [h.GetXaxis().GetBinLabel(i+1) for i in range(h.GetNbinsX())]
                if labels != old_channels[c] or not h.GetSumw2N():
                    raise ValueError('Native label/variance mismatch')
                if any(h.GetBinContent(i) or h.GetBinError(i) for i in (0,h.GetNbinsX()+1)):
                    raise ValueError('Unaccounted flow content')
                values = transformed([h.GetBinContent(i+1) for i in range(h.GetNbinsX())], c, groups)
                variances = transformed([h.GetBinError(i+1)**2 for i in range(h.GetNbinsX())], c, groups)
                dest.cd()
                if c.startswith('SR_highdm_'):
                    out = R.TH1D(h.GetName(),h.GetTitle(),len(values),0.,float(len(values)))
                    out.Sumw2()
                    for i,(v,w) in enumerate(zip(values,variances),1):
                        out.SetBinContent(i,float(v));out.SetBinError(i,math.sqrt(float(w)))
                        out.GetXaxis().SetBinLabel(i,new_channels[c][i-1])
                    out.SetEntries(h.GetEntries())
                else:
                    out = h.Clone()
                if out.Write() <= 0:
                    raise OSError('Native histogram write failed')
                out.SetDirectory(0); h.SetDirectory(0)
                count += 1
    finally:
        dst.Close(); src.Close()
    return count


def verify_root(source, target, old_channels, new_channels, groups):
    import uproot
    count = 0
    with uproot.open(source,object_cache=None,array_cache=None) as a, uproot.open(target,object_cache=None,array_cache=None) as b:
        if a.classnames(cycle=False) != b.classnames(cycle=False):
            raise ValueError('Native histogram objects changed')
        for name, kind in a.classnames(cycle=False).items():
            if kind.startswith('TDirectory'): continue
            c = name.split('/')[0]
            before,after = a[name], b[name]
            if before.axis().labels()!=old_channels[c] or after.axis().labels()!=new_channels[c]:
                raise ValueError('Native merge labels changed')
            for field in ('values','variances'):
                old = getattr(before,field)();new = getattr(after,field)()
                np.testing.assert_allclose(new,transformed(old,c,groups),rtol=2e-12,atol=1e-12)
                np.testing.assert_allclose(new.sum(),old.sum(),rtol=2e-12,atol=1e-9)
            count += 1
    return count


def build(source_manifest, merge_path, output):
    if sys.platform!='linux':raise RuntimeError('ROOT templates remain on hep2')
    source_manifest,merge_path,output = map(internal_path,(source_manifest,merge_path,output))
    if output==source_manifest.parent:raise ValueError('Source grid is immutable')
    source = read_json(source_manifest)
    if (source['status']!='cards_ready' or source['scope']!='full_nominal_production'
            or source['bins']!=540 or source.get('sr_merge') or not source['root_integrity_checked']):
        raise ValueError('Expected the verified original162+30 SR grid')
    reference = dict(path=str(merge_path.relative_to(ROOT)),sha256=sha256(merge_path))
    config = load_merge(reference); groups = bin_groups(config)
    decision = ROOT/config['decision_payload']
    if sha256(decision)!=config['decision_payload_sha256']:raise ValueError('Merge decision input changed')
    decision_data=read_json(decision)
    if decision_data['provenance']['manifest_sha256']!=sha256(source_manifest):
        raise ValueError('Merge decision from different grid')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'builder.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=dict(source_grid=dict(path=str(source_manifest.relative_to(ROOT)),sha256=sha256(source_manifest)),
            sr_merge=reference,code={n:sha256(ROOT/n) for n in (
                'workflows/merge_sr_categories.py','utils/sr_merge.py')})
        if (output/'manifest.json').exists():
            final=read_json(output/'manifest.json')
            if final.get('merge_contract')!=contract:raise ValueError('Divergent completed grid')
            cache=IntegrityCache()
            for point in final['points']:verify_card(point,cache)
            print('Existing merged grid verified');return
        if (output/'contract.json').exists():raise ValueError('Interrupted merge retained; inspect before continuation')
        write_json(output/'contract.json',contract)
        write_json(output/'state.json',dict(status='merging_native_templates',pid=os.getpid(),started=time.time()))
        cache=IntegrityCache()
        for point in source['points']:verify_card(point,cache)
        old_path=internal_path(ROOT/source['templates_manifest']);old=read_json(old_path)
        source_templates=old_path.parent;templates=output/'templates';templates.mkdir()
        channels=copy.deepcopy(old['channels'])
        for c in channels:
            if c.startswith('SR_highdm_'):channels[c]=['SR_highdm_bin'+str(i) for i in range(len(groups))]
        audit={}
        for name,digest in old['files'].items():
            if sha256(source_templates/name)!=digest:raise ValueError('Source template changed')
            written=merge_root(source_templates/name,templates/name,old['channels'],channels,groups)
            verified=verify_root(source_templates/name,templates/name,old['channels'],channels,groups)
            if written!=verified:raise ValueError('Incomplete native readback')
            audit[name]=dict(histograms_verified=verified,source_sha256=digest,target_sha256=sha256(templates/name))
            print(name,verified,'histograms projected and verified',flush=True)
        manifest=copy.deepcopy(old)
        manifest.update(channels=channels,bins=sum(map(len,channels.values())),sr_merge=reference,
            files={n:a['target_sha256'] for n,a in audit.items()},source_templates_manifest=dict(
                path=str(old_path.relative_to(ROOT)),sha256=sha256(old_path)))
        manifest['contract']=dict(old['contract'],sr_merge=reference,merge_contract=contract)
        write_json(templates/'model_manifest.json',manifest)
        origins=[dict(path=str(templates.relative_to(ROOT)),sha256=sha256(templates/'model_manifest.json'))]
        new_points=[];cr_only=None
        for point in source['points']+[source['cr_only_card']]:
            old_card=internal_path(ROOT/point['card']);meta=read_json(old_card.with_suffix('.manifest.json'))
            if sha256(old_card)!=point['card_sha256']:raise ValueError('Source card changed')
            target=output/'datacards'/old_card.name
            text=rewrite_card(old_card.read_text(),old_card.parent,source_templates,templates,target.parent)
            meta.pop('card_sha256',None);meta.update(inputs=origins,sr_merge=reference,source_card_sha256=point['card_sha256'])
            digest=write_card_once(target,text,meta)
            record=dict(point,card=str(target.relative_to(ROOT)),card_sha256=digest)
            if meta['cr_only']:cr_only=record
            else:new_points.append(record)
        final=dict(source,points=new_points,cr_only_card=cr_only,bins=manifest['bins'],sr_merge=reference,
            merge_contract=contract,templates_manifest=str((templates/'model_manifest.json').relative_to(ROOT)),finished=time.time())
        new_cache=IntegrityCache()
        for point in new_points:verify_card(point,new_cache)
        write_json(output/'merge_audit.json',dict(status='passed',root_histograms=audit,
            all_native_objects_readback_verified=True,all_process_and_signal_integrals_preserved=True,
            all_sumw2_and_endpoints_preserved=True,cr_and_lowdm_arrays_unchanged=True,
            sr_data_blinded=True,old_bins=540,new_bins=manifest['bins'],full_workflow_complete=False))
        write_json(output/'manifest.json',final)
        write_json(output/'state.json',{k:v for k,v in final.items() if k!='points'})
        print('cards_ready',len(new_points),'signals',manifest['bins'],'analysis bins',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--merge',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--native-runtime',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.native_runtime:
        env=runtime(internal_path(a.output)/'software_runtime')
        os.chdir(ROOT)
        os.execvpe('python3',['python3',str(Path(__file__).resolve()),'--manifest',str(internal_path(a.manifest)),
            '--merge',str(internal_path(a.merge)),'--output',str(internal_path(a.output)),'--native-runtime'],env)
    build(a.manifest,a.merge,a.output)

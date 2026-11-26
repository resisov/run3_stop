"""Use the current TT slot in an otherwise unchanged blinded background model."""
import copy
from pathlib import Path
import numpy as np
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json
from TROTASR.utils.top_pt import PRESCRIPTION
from TROTASR.workflows.top_pt_inventory import RELATIVE


def replace_tt(original, replay):
    """Object variations are absolute sparse populations, including lost cells."""
    for mode in ('highdm','lowdm'):
        dest=original[mode];src=replay[mode]
        for name in dest.keys()|src.keys():
            target=dest.setdefault(name,{})
            for cats in target.values():
                for samples in cats.values():samples.pop('TT',None)
            for region,cats in src.get(name,{}).items():
                for key,samples in cats.items():
                    if set(samples)-{'TT'}:raise ValueError('Non-TT scoped sample')
                    if 'TT' in samples:
                        target.setdefault(region,{}).setdefault(key,{})['TT']=samples['TT']


def update_dy(original,previous,current):
    from TROTASR.workflows.tt_dy_subtraction import leaves
    out=copy.deepcopy(original)
    before,after=dict(leaves(previous)),dict(leaves(current))
    for key in before.keys()|after.keys():
        if key[-1]!='other':raise ValueError('Inclusive ttbar must be non-DY')
        node=out
        for k in key:node=node[k]
        old,new=before.get(key),after.get(key)
        if old is None or new is None or old['edges']!=node['edges'] or new['edges']!=node['edges']:
            raise ValueError('Inconsistent on/off-Z subtraction axes: '+str(key))
        if old['entries']!=new['entries']:raise ValueError('Changed TT population across weights')
        for field in ('entries','sumw2','sumw'):
            v=np.asarray(node[field])-np.asarray(old[field])+np.asarray(new[field])
            if not np.isfinite(v).all() or (field!='sumw' and (v<0).any()):
                raise ValueError('Unsupported DY subtraction: '+str(key))
            node[field]=v.tolist()
    return out


def shifted_controls(payload,endpoint):
    """Override only TT in the recoil controls used by the background measurement."""
    h={}
    for mode,tree in payload['histograms'].items():
        nom={}
        for region in ('GCR','DY2E','DY2M'):
            nom[region]={}
            for k,samples in tree['nominal'][region].items():
                row=dict(samples)
                if 'TT' in row:row['TT']=tree[endpoint][region][k]['TT']
                nom[region][k]=row
        h[mode]={'nominal':nom}
    return dict(payload,histograms=h)


def measure_era(p):
    from TROTASR.utils.background_estimation import measure
    from TROTASR.workflows.build_background_estimation import merged_zgamma_measurement,ZGamma_BPIX_MERGE
    m=measure(p)
    if str(p['contract']['year'])==ZGamma_BPIX_MERGE['year'] and m['double_ratio']['status']=='blocked':
        m['double_ratio']=merged_zgamma_measurement(p,m['double_ratio'])
    for k,record in m.items():
        if record.get('status')!='complete':
            raise ValueError('Measured factor blocked: '+str(p['contract']['year'])+'/'+k+' '+str(record.get('issues',[]))[:600])
    return m


def transfer_diagnostic(era,current,baseline):
    """Keep the display TF status; it is not an input to the card likelihood."""
    allowed={('highdm','qcd_qcdcr','Nb1','unsupported_cells'),
             ('highdm','qcd_qcdcr','Nb2plus','unsupported_cells'),
             ('lowdm','w_llcr','Nb2plus_NISR0','signed_mc_bins')}
    def issues(v):
        return {(i['mode'],i['route'],i['category'],i['status']) for i in v['issues']}
    found=issues(current)
    status='unsupported_cells' if found else 'complete'
    if (str(current['year'])!=era or str(baseline['year'])!=era or current['schema']!=baseline['schema']
            or current['status']!=status or baseline['status']!=status
            or found!=issues(baseline) or not found<=allowed):
        raise ValueError('Changed native transfer-factor support in '+era+': '+str(sorted(found)))
    for mode,route,category,why in found:
        new=current[mode]['records'][route][category];old=baseline[mode]['records'][route][category]
        if (new['status']!=why or new['valid']!=old['valid'] or
                new['process_coverage']!=old['process_coverage']):
            raise ValueError('Changed native transfer-factor bins: '+str((era,mode,route,category)))
        for field in ('numerator','denominator','numerator_sumw2','denominator_sumw2'):
            np.testing.assert_allclose(new[field],old[field],rtol=1e-12,atol=1e-14,
                                       err_msg=str((era,mode,route,category,field)))
    return dict(status=status,issues=current['issues'],card_input=False)


def build_year(era):
    from TROTASR.utils.paths import ERAS
    from TROTASR.utils.top_pt import ENDPOINTS
    from TROTASR.gnn4lowdm.configuration import configuration_path
    from TROTASR.utils.transfer_factors import build_transfer_factors
    base=ROOT/RELATIVE;dest=ROOT/'stats/run3/top_pt/inputs'/era
    if era not in ERAS:raise ValueError('Unknown analysis era')
    prior=ROOT/'results/merged'/('systematics_'+era+'.json.gz')
    slot=read_json(base/'card_inputs'/(era+'.json.gz')); dy=read_json(base/'dy_subtraction'/(era+'.json.gz'))
    if (slot['scope']!='TT_slot_only' or slot['top_pt_prescription']!=PRESCRIPTION
            or slot['status']!='complete' or dy['status']!='complete' or dy['regions']!=['DY2E','DY2M']
            or dy['observations_used'] or dy['year']!=era or slot['year']!=era):
        raise ValueError('Incomplete TT inputs: '+era)
    p=read_json(prior)
    if p.get('sr_data_blinded') is not True or p['status']!='complete' or str(p['contract']['year'])!=era:
        raise ValueError('Wrong baseline histogram population: '+era)
    for section in ('histograms','physical_histograms'):replace_tt(p[section],slot[section])
    old=p['dy_rz']; current=dy['dy_rz']; p['dy_rz']=update_dy(old,current['previous'],current['nominal'])
    # The card measurements consume the nominal on/off-Z matrix; do not publish
    # inherited detector-only auxiliary matrices as new TT-corrected products.
    p.pop('dy_rz_variations',None)
    p['tt_card_input']=dict(year=era,prescription=PRESCRIPTION,coverage=slot['coverage'],
                            dy_files=dy['files'],nominal_pass_only_for_jme=True)
    factors=measure_era(p)
    pairs={}
    for endpoint in ENDPOINTS:
        shifted=shifted_controls(p,endpoint);shifted['dy_rz']=update_dy(old,current['previous'],current[endpoint])
        pairs[endpoint]=measure_era(shifted)
    factors['transfer_factors']=build_transfer_factors(p,read_json(configuration_path(era)))
    diagnostic=transfer_diagnostic(era,factors['transfer_factors'],
                                  read_json(ROOT/'results/measurements'/era/'transfer_factors.json.gz'))
    for name,record in factors.items():
        record['provenance']=dict(campaign_year=era,top_pt_prescription=PRESCRIPTION)
        if name!='transfer_factors':
            record['inclusive_ttbar_endpoints']={k:v[name] for k,v in pairs.items()}
        if name!='transfer_factors' and record.get('status')!='complete':
            raise ValueError('Incomplete '+name+': '+era)
        write_json(dest/(name+'.json.gz'),record)
    write_json(dest/'histograms.json.gz',p)
    write_json(dest/'receipt.json',dict(status='complete',year=era,tt_card_input=p['tt_card_input'],
                                       sr_data_blinded=True,measurements=sorted(factors),
                                       transfer_factor_diagnostic=diagnostic))


def infer():
    """One run of the canonical stages. Never launch fits on a partial grid."""
    import fcntl
    import os
    import shutil
    import subprocess
    import time
    from TROTASR.utils.paths import ERAS
    from TROTASR.workflows.resources import resources
    dest=ROOT/'stats/run3/top_pt';dest.mkdir(parents=True,exist_ok=True)
    dy=ROOT/RELATIVE/'dy_subtraction'; manifest=dest/'grid/manifest.json'
    state=dict(controller_pid=os.getpid(),status='running',stages={})
    def publish():
        state['updated']=time.time();write_json(dest/'state.json',state)
    def stage(name,status,**metadata):
        state['stages'][name]=dict(status=status,**metadata);publish()
    def launch(name,cmd):
        with (dest/(name+'.log')).open('ab') as log:
            p=subprocess.Popen(['bash','TROTASR/workflows/run_inference.sh','-m','TROTASR',name]+list(map(str,cmd)),
                  cwd=str(ROOT.parent),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        stage(name,'started',pid=p.pid);return p
    def coverage():
        src=read_json(ROOT/RELATIVE/'nominal.json')['tasks']
        by={era:sum(str(t['year'])==era for t in src) for era in ERAS}
        if not all(by.values()) or sum(by.values())!=len(src):raise ValueError('Invalid DY input plan')
        return by
    with (dest/'inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            by=coverage();stage('dy','waiting',files=sum(by.values()))
            while True:
                s=read_json(dy/'state.json')
                if s.get('failed') or s.get('total')!=sum(by.values()):
                    raise ValueError('DY subtraction failure or coverage mismatch, see '+str(dy/'state.json'))
                if s['status']=='complete' and s['complete']==s['total']:break
                with (dy/'controller.lock').open('rb') as owner:
                    try:fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:
                        if read_json(dy/'state.json')['status']=='complete':continue
                        raise RuntimeError('DY subtraction controller is no longer active; no fit submitted')
                time.sleep(15)
            stage('dy','complete',files=s['complete']);stage('inputs','running',eras=[])
            done=[]
            for era in ERAS:
                source=read_json(dy/(era+'.json.gz'))
                if source.get('files')!=by[era] or source.get('status')!='complete':
                    raise ValueError('Missing or partial DY subtraction: '+era)
                if not (dest/'inputs'/era/'receipt.json').exists():
                    while True:
                        jobs,mem=resources(all_account=True)
                        if jobs<116 and mem>40*2**30 and shutil.disk_usage(dest).free>80*2**30:break
                        time.sleep(15)
                    build_year(era)
                r=read_json(dest/'inputs'/era/'receipt.json')
                if (r.get('year')!=era or r.get('status')!='complete' or not r.get('sr_data_blinded')
                        or r['tt_card_input']['dy_files']!=by[era] or r['tt_card_input']['prescription']!=PRESCRIPTION):
                    raise ValueError('Invalid updated model inputs: '+era)
                done.append(era);stage('inputs','running',eras=done)
            stage('inputs','complete',eras=done)
            p=launch('cards',['--config',ROOT/'jsons/analysis.json','--output',manifest.parent,
                              '--preserve-unsupported-templates','--card-workers',8])
            rc=p.wait()
            if rc:stage('cards','failed',exit_code=rc);raise RuntimeError('Card construction failed; see cards.log')
            m=read_json(manifest); ready=[r for r in m['points'] if r.get('fit_ready')]
            if (m.get('status')!='cards_ready' or m.get('tt_card_input')!=PRESCRIPTION or
                m.get('bins')!=208 or m.get('floating_normalization_parameters')!=44 or
                any(sum(r['model']==model for r in ready)!=n for model,n in [('T2tt',205),('T2bW',345),('T2tb',345)])
                or len(ready)!=895 or m.get('signal_templates_nominal_only') is not True):
                raise ValueError('New card grid is not the requested 895-point 208-bin/44-rate model')
            stage('cards','complete',files=len(ready))
            children={}; common=['--manifest',manifest]
            for name,extra in [('limits',['--workers','56']),('cronly',[])]:
                children[name]=launch(name,common+['--output',dest/name]+extra)
            stage('impacts','waiting_for_benchmark')
            benchmark=dest/'limits/T2tt/mStop1200_mLSP500/state.json'
            while children:
                if state['stages']['impacts']['status']=='waiting_for_benchmark':
                    if benchmark.exists() and read_json(benchmark).get('workspace_valid') is True:
                        children['impacts']=launch('impacts',common+['--output',dest/'impacts','--workers',24,
                                      '--workspace-state',benchmark])
                    elif 'limits' not in children:
                        stage('impacts','blocked',reason='Limit benchmark workspace not available')
                for name,p in list(children.items()):
                    rc=p.poll()
                    if rc is not None:
                        stage(name,'complete' if rc==0 else 'failed',exit_code=rc);del children[name]
                if children:time.sleep(15)
            if any(s['status']!='complete' for s in state['stages'].values()):
                raise RuntimeError('One or more inference stages did not complete; inspect stage receipts')
            state['status']='complete';publish()
        except Exception as exc:
            state.update(status='needs_attention',error=repr(exc));publish();raise


def restart_limits():
    """Replace the recalled limit stage; adopt the existing independent impact fit."""
    import fcntl,os,subprocess,time
    from TROTASR.workflows.resources import process_identity
    d=ROOT/'stats/run3/top_pt'; manifest=d/'grid/manifest.json'
    with (d/'inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read_json(d/'state.json')
        stages=state['stages']
        last=stages['limits']
        saved=read_json(d/'limits/workflow_state.json')
        before_admission=(last.get('prior_ranges') is True and last.get('exit_code')==1
            and saved.get('status')=='draining' and saved.get('pid')!=last.get('pid')
            and not saved.get('results') and not (d/'limits/range_plan.json').exists()
            and not process_identity(last.get('pid')))
        if (stages['cards']!={'files':895,'status':'complete'} or stages['cronly'].get('exit_code')!=0
                or last.get('status')!='failed' or (last.get('exit_code')!=-15 and not before_admission)):
            raise ValueError('Expected an explicitly recalled limit stage and completed CR-only fit')
        def publish():
            state['updated']=time.time();write_json(d/'state.json',state)
        state.update(controller_pid=os.getpid(),status='running');state.pop('error',None)
        impact=read_json(d/'impacts/impact_status.json')
        ipid=impact['controller_pid'];iticks=impact['controller_pid_start_ticks']
        assert stages['impacts'].get('pid')==ipid or stages['impacts'].get('status')=='complete'
        with (d/'limits.log').open('ab') as log:
            child=subprocess.Popen(['bash','TROTASR/workflows/run_inference.sh','-m','TROTASR','limits',
                '--manifest',str(manifest),'--output',str(d/'limits'),'--workers','56','--prior-ranges'],
                cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        stages['limits']=dict(status='started',pid=child.pid,prior_ranges=True);publish()
        while True:
            rc=child.poll()
            if rc is not None and stages['limits']['status']=='started':
                stages['limits'].update(status='complete' if rc==0 else 'failed',exit_code=rc);publish()
            if stages['impacts']['status']=='started':
                live=process_identity(ipid)
                if not live or live['start_ticks']!=iticks:
                    saved=read_json(d/'impacts/impact_status.json')
                    good=saved.get('status') in ('generated_pending_visual_QA','fits_complete')
                    stages['impacts'].update(status='complete' if good else 'failed',receipt_status=saved.get('status'))
                    publish()
            if rc is not None and stages['impacts']['status']!='started':break
            time.sleep(10)
        state['status']='complete' if all(row['status']=='complete' for row in stages.values()) else 'needs_attention'
        publish()
        return 0 if state['status']=='complete' else 1


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--restart-limits',action='store_true')
    args=parser.parse_args()
    if args.restart_limits:raise SystemExit(restart_limits())
    infer()

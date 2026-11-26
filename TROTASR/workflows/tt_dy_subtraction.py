"""Measure the inclusive ttbar part of the existing on/off-Z subtraction.

Only the control measurement uses the pre-replacement weight. It never
selects an alternative central prediction for cards, or uses observed events.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
from functools import lru_cache
from pathlib import Path
import os
import shutil
import time
import numpy as np
import awkward as ak
import uproot
from TROTASR.utils import histogramming as h, top_pt
from TROTASR.utils.io import read_json, write_json, validate_root
from TROTASR.utils.paths import ROOT, era_year
from TROTASR.utils.systematic_views import event_views
from TROTASR.utils.trota_friend import find_friend_task, open_friend
from TROTASR.utils.region_kinematics import build_region_blocks
from TROTASR.utils.event_selections import highdm_core, topology_buffers, topology_counts
from TROTASR.workflows.top_pt_inventory import RELATIVE
from TROTASR.workflows.resources import resources, holds_flock, process_identity
from TROTASR.workflows.merge_nominal import merge_histograms

BASE = ROOT/RELATIVE
DEST = BASE/'dy_subtraction'
CHANNELS = ('DY2E','DY2M')
WEIGHTS = ('previous','nominal',*top_pt.ENDPOINTS)


def previous_factor(pts):
    # Frozen input used offset 1.051 before the approved replacement by 0.973.
    pair = ak.pad_none(pts,2,axis=1,clip=True)
    both = np.asarray(ak.num(pts,axis=1)) >= 2
    out = np.ones(len(pts))
    def one(pt):
        return (.103*np.exp(-.0118*pt)-.000134*pt+1.051)*(.991+.000075*pt)
    for_pts = [np.asarray(ak.to_numpy(pair[both,i]),dtype=float) for i in (0,1)]
    out[both] = np.sqrt(one(for_pts[0])*one(for_pts[1]))
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite pre-replacement DY subtraction')
    return out


@lru_cache(maxsize=6)
def normalization(era):
    return read_json(ROOT/'estimations'/('normalization_'+era+'.json.gz'))


def support(a,era):
    blocks,_=build_region_blocks(a,dy_mass_window=None,year=era)
    return np.logical_or.reduce([highdm_core(a,blocks,r,None,era) | (blocks[r].core & (blocks[r].nb>=1))
                                 for r in CHANNELS])


def fill(a,res,meta,era,policy,trees):
    blocks,_=build_region_blocks(a,dy_mass_window=None,year=era)
    buffers=topology_buffers(a)
    for region in CHANNELS:
        b=blocks[region]
        if not (highdm_core(a,blocks,region,None,era) | (b.core & (b.nb>=1))).any():
            continue
        counts,clean,_,_=topology_counts(a,res,region,buffers,era)
        high=highdm_core(a,blocks,region,None,era) & (counts.sum(axis=1)>=1)
        low=b.core & (b.nb>=1) & (counts.sum(axis=1)==0)
        mass=np.asarray(a['mee' if region=='DY2E' else 'mmm'],dtype=float)
        ids=np.asarray(a['dataset_id'])
        for key in np.unique(ids[(high|low)&(mass>50)]):
            use=(ids==key)&(high|low)&(mass>50)
            dataset,process,is_data,is_signal=h.dataset_label(meta,int(key))
            if is_data or is_signal or not top_pt.inclusive_ttbar(dataset):
                raise ValueError('DY subtraction accepts inclusive ttbar simulation only')
            sub={k:a[k][use] for k in ak.fields(a)}
            masks={r:np.full(use.sum(),r==region,dtype=bool) for r in h.REGIONS}
            pts=sub['gen_top_pt']; cols=dict(sub,gen_top_pt=pts[:,:0])
            common,_=h.event_weights(cols,dataset,process,era_year(era),masks,policy,apply_topw=False,include_variations=False)
            tw,_=h.topw_event_variations(ROOT,str(era),dataset,process,sub,policy,cleaned=clean[use],include_variations=False)
            norm=h.norm_vector(normalization(era),{k:sub[k][:1] for k in ('dataset_id','mStop','mLSP')},
                               int(key),dataset,False,False,require_normalization=True)
            central=top_pt.pair_weight(pts)
            for weight,factor in dict(previous=previous_factor(pts),nominal=central,**top_pt.variation_components(central)).items():
                w=(common['nominal']*factor)*tw['nominal']*norm
                m=mass[use]
                for mode,keep in (('highdm',high[use]),('lowdm',low[use])):
                    for nb,nmask in (('Nb1',b.nb[use]==1),('Nb2plus',b.nb[use]!=1)):
                        take=keep&nmask
                        if not take.any(): continue
                        h.add(trees[weight],(mode,'mll',region,nb,'other'),m[take],w[take],h.MLL_EDGES)
                        for window,window_mask in (('on',(m>71)&(m<111)),('off',((m>50)&(m<71))|(m>111))):
                            good=take&window_mask
                            h.add(trees[weight],(mode,'yields',region,nb,window,'other'),
                                  np.full(good.sum(),.5),w[good],np.array([0.,1.]))


def leaves(node,prefix=()):
    if 'sumw' in node:
        yield prefix,node
    else:
        for key,v in node.items():yield from leaves(v,prefix+(key,))


def check_central(actual,reference):
    a,b=dict(leaves(actual)),dict(leaves(reference))
    for key in a.keys()|b.keys():
        x,y=a.get(key),b.get(key)
        if x is None or y is None:
            if any(np.any((x or y)[k]) for k in ('entries','sumw','sumw2')):
                raise ValueError('Missing populated DY cell: '+str(key))
            continue
        if x['edges']!=y['edges'] or x['entries']!=y['entries']:
            raise ValueError('Changed DY population or bins: '+str(key))
        for k in ('sumw','sumw2'):
            np.testing.assert_allclose(x[k],y[k],rtol=1e-8,atol=1e-10,err_msg=str(key))


def shard_binding(t):
    return (DEST/t['year']/(Path(t['input']).stem+'.json.gz'),
            dict(input=t['input'],year=t['year'],target_entries=t['target_entries'],weights=list(WEIGHTS),
                 regions=list(CHANNELS),top_pt_prescription=top_pt.PRESCRIPTION,observations_used=False))


def reusable(out,header):
    if out.exists():
        row=read_json(out)
        if row.get('status')!='complete' or not all(row.get(k)==v for k,v in header.items()):
            raise ValueError('Different existing DY-only receipt: '+str(out))
        return True
    return False


def shard(t):
    era=t['year'];p=ROOT/t['input'];out,header=shard_binding(t)
    if reusable(out,header):return str(out)
    meta=validate_root(p,era);_,seen,gate=h._dataset_preselection(meta,list(top_pt.PRIMARIES))
    friend_task=find_friend_task(p,era)
    if not friend_task:raise ValueError('No corrected nominal friend: '+str(p))
    trees={w:{} for w in WEIGHTS};audit=dict(events_read=0,central_object_corrections={})
    with uproot.open(p,object_cache=None,array_cache=None) as f, open_friend(friend_task,p) as friend:
        policy=h.topw_file_input_policy(f); reader=h.TopWEvents(f,policy)
        if reader.num_entries!=t['entries'] or friend.num_entries!=reader.num_entries:
            raise ValueError('Events/friend population differs: '+str(p))
        columns=[k for k in reader.events.keys() if k!='nTopMixed1pct' and not k.startswith(('TopMixed1pct_','nTopMixed1pct_'))]
        light=('pass_electron_trigger','pass_muon_trigger','nelectron_eta_all','nmuon_eta_all')
        def accept(a):
            return gate.accept(a)&((np.asarray(a[light[0]],bool)&(np.asarray(a[light[2]])>=2))|
                                   (np.asarray(a[light[1]],bool)&(np.asarray(a[light[3]])>=2)))
        chunks=h._gated_event_chunks(reader,columns,1000000,(*gate.columns,*light),accept,audit)
        chunks=h._gated_systematic_chunks(chunks,era,audit,friend=friend)
        for a,res,_ in event_views(chunks,era,None,{},nominal_only=True,friend_chunks=True,
                                   support_mask=lambda a:support(a,era)):
            fill(a,res,meta,era,policy,trees)
    if sum(seen.values())!=t['target_entries'] or audit['events_read']!=t['entries']:
        raise ValueError('Incomplete inclusive TT coverage: '+str(p))
    check_central(trees['nominal'],read_json(ROOT/t['output'])['dy_rz'])
    write_json(out,dict(header,status='complete',dy_rz=trees))
    return str(out)


def run(workers,index=None):
    inv=read_json(BASE/'nominal.json')
    tasks=inv['tasks']
    if inv.get('top_pt_prescription')!=top_pt.PRESCRIPTION:raise ValueError('Wrong input population')
    if index is not None:
        print(shard(tasks[index]),flush=True);return
    DEST.mkdir(parents=True,exist_ok=True)
    lock_path=DEST/'controller.lock'
    inherited=os.environ.pop('TROTASR_DY_HANDOFF_FD',None)
    with (lock_path.open('a') if inherited is None else os.fdopen(int(inherited),'rb')) as lock:
        if inherited is not None and (not os.path.samestat(os.fstat(lock.fileno()),lock_path.stat())
                                      or not holds_flock(os.getpid(),lock_path)):
            raise RuntimeError('The inherited lock does not belong to this DY controller')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[];missing=[];failed={};remain=True;active={}
        for task in tasks:
            out,header=shard_binding(task)
            if reusable(out,header):results.append(str(out))
            else:missing.append(task)
        reused=len(results);queue=iter(missing);ticks=process_identity(os.getpid())['start_ticks']
        def snapshot(state):
            write_json(DEST/'state.json',dict(status=state,pid=os.getpid(),total=len(tasks),complete=len(results),
                       workers=workers,start_ticks=ticks,reused=reused,
                       failed=len(failed),errors=failed,active=len(active),updated=time.time()))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            snapshot('running')
            while remain or active:
                ext,mem=resources(all_account=True)
                room=min(workers-len(active),116-ext-len(active)-1,(mem-32*2**30)//(5*2**30))
                if shutil.disk_usage(DEST).free<80*2**30:room=0
                while remain and room>0 and not failed:
                    t=next(queue,None)
                    if t is None:remain=False;break
                    from TROTASR.workflows.tt_dy_subtraction import shard as worker
                    active[pool.submit(worker,t)]=t;room-=1
                if failed:remain=False
                if not active:
                    if remain:time.sleep(5)
                    continue
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    t=active.pop(future)
                    try:results.append(future.result())
                    except Exception as exc:failed[t['input']]=repr(exc)
                snapshot('running')
        if failed:
            snapshot('failed');raise RuntimeError(str(next(iter(failed.items()))))
        from TROTASR.utils.paths import ERAS
        for era in ERAS:
            data={w:{} for w in WEIGHTS};by_era=[Path(x) for x in results if Path(x).parent.name==era]
            for p in sorted(by_era):
                for w,v in read_json(p)['dy_rz'].items():merge_histograms(data[w],v)
            write_json(DEST/(era+'.json.gz'),dict(year=era,status='complete',files=len(by_era),weights=list(WEIGHTS),
                       dy_rz=data,regions=list(CHANNELS),observations_used=False))
        snapshot('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,choices=range(1,49),default=24)
    p.add_argument('--task-index',type=int);a=p.parse_args();run(a.workers,a.task_index)

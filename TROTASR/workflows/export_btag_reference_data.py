"""Export actual adopted b-tag efficiency cells for read-only figure QA."""
import argparse
from pathlib import Path
import numpy as np
from coffea.util import load
from hist import loc
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT,internal_path
from TROTASR.utils.io import read_json,write_json,sha256

SAMPLES={'TTto2L2Nu':'TTto2L2Nu_','TTtoLNu2Q':'TTtoLNu2Q_','TTto4Q':'TTto4Q_',
         'WtoLNu':'WtoLNu-2Jets_Bin-1J-PTLNu-200to400_',
         'Zto2Nu':'Zto2Nu-2Jets_Bin-1J-PTNuNu-200to400_'}


def run(output):
    assets=read_json(ROOT/'jsons/assets.json')
    sources=[ROOT/'estimations'/('btageff%s.merged'%y) for y in (2024,2025)]
    for p in sources:
        expected=next(r['sha256'] for r in assets['files'] if r['target']==str(p.relative_to(ROOT)))
        if sha256(p)!=expected:raise ValueError('Adopted efficiency payload changed')
    if sha256(sources[0])!=sha256(sources[1]):raise ValueError('Years no longer share this reference')
    histograms=load(str(sources[0]))['UParTAK4'];panels=[]
    for sample,prefix in SAMPLES.items():
        matches=[k for k in histograms if k.startswith(prefix)]
        if len(matches)!=1:raise ValueError('Ambiguous sample: '+sample)
        for flavor,label in ((5,'b'),(4,'c'),(0,'lf')):
            h=histograms[matches[0]]
            passed=h[{'wp':loc('medium'),'btag':loc('pass'),'flavor':loc(flavor)}]
            failed=h[{'wp':loc('medium'),'btag':loc('fail'),'flavor':loc(flavor)}]
            if [a.name for a in passed.axes]!=['pt','abseta']:raise ValueError('Unexpected axes')
            num=passed.values(flow=False);fail=failed.values(flow=False);den=num+fail
            if not np.all(np.isfinite(den)) or np.any(num<0) or np.any(fail<0):raise ValueError('Invalid counts')
            eff=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den!=0)
            panels.append(dict(sample=sample,dataset=matches[0],flavor=label,wp='medium',
                an_path='figures/btageff2024/%s_%s_medium.png'%(sample,label),
                pt_edges=passed.axes['pt'].edges.tolist(),abseta_edges=passed.axes['abseta'].edges.tolist(),
                passed=num.tolist(),failed=fail.tolist(),
                efficiency=[[float(v) if np.isfinite(v) else None for v in row] for row in eff]))
    result=dict(schema='trotasr_btag_reference_cells_v1',tagger='UParTAK4',years=[2024,2025],
        source_payloads=[str(p.relative_to(ROOT)) for p in sources],source_payload_sha256=sha256(sources[0]),
        exporter_sha256=sha256(Path(__file__)),source_modified=False,panels=panels)
    write_json(internal_path(output),result)
    print(__import__('json').dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)

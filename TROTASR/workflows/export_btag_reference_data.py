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
from TROTASR.utils.io import write_json

SAMPLES={'TTto2L2Nu':'TTto2L2Nu_','TTtoLNu2Q':'TTtoLNu2Q_','TTto4Q':'TTto4Q_',
         'WtoLNu':'WtoLNu-2Jets_Bin-1J-PTLNu-200to400_',
         'Zto2Nu':'Zto2Nu-2Jets_Bin-1J-PTNuNu-200to400_'}


def run(output):
    sources=[ROOT/'estimations'/('btageff%s.merged'%y) for y in (2024,2025)]
    histograms=load(str(sources[0]))['UParTAK4'];panels=[]
    histograms_2025=load(str(sources[1]))['UParTAK4']
    for sample,prefix in SAMPLES.items():
        matches=[k for k in histograms if k.startswith(prefix)]
        matches_2025=[k for k in histograms_2025 if k.startswith(prefix)]
        if len(matches)!=1 or len(matches_2025)!=1:raise ValueError('Ambiguous sample: '+sample)
        for flavor,label in ((5,'b'),(4,'c'),(0,'lf')):
            h=histograms[matches[0]]
            passed=h[{'wp':loc('medium'),'btag':loc('pass'),'flavor':loc(flavor)}]
            failed=h[{'wp':loc('medium'),'btag':loc('fail'),'flavor':loc(flavor)}]
            if [a.name for a in passed.axes]!=['pt','abseta']:raise ValueError('Unexpected axes')
            num=passed.values(flow=False);fail=failed.values(flow=False);den=num+fail
            for label_key, counts in (('pass', passed), ('fail', failed)):
                other=histograms_2025[matches_2025[0]][{'wp':loc('medium'),'btag':loc(label_key),'flavor':loc(flavor)}]
                if (other.axes != counts.axes
                        or not np.array_equal(other.values(flow=False), counts.values(flow=False))):
                    raise ValueError('The two eras have different displayed b-tag efficiency inputs: '+sample)
            if not np.all(np.isfinite(den)) or np.any(num<0) or np.any(fail<0):raise ValueError('Invalid counts')
            eff=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den!=0)
            panels.append(dict(sample=sample,dataset=matches[0],flavor=label,wp='medium',
                an_path='figures/btageff2024/%s_%s_medium.png'%(sample,label),
                pt_edges=passed.axes['pt'].edges.tolist(),abseta_edges=passed.axes['abseta'].edges.tolist(),
                passed=num.tolist(),failed=fail.tolist(),
                efficiency=[[float(v) if np.isfinite(v) else None for v in row] for row in eff]))
    result=dict(schema='trotasr_btag_reference_cells_v1',tagger='UParTAK4',years=[2024,2025],
        source_payloads=[str(p.relative_to(ROOT)) for p in sources],source_modified=False,panels=panels)
    write_json(internal_path(output),result)
    print(__import__('json').dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)

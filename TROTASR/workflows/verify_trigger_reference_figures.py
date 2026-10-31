"""Read-only numeric binding of preserved trigger PDFs to measured JSON.

Recover linear axes from the PDF's own labeled major ticks, then verify every
marker, horizontal bin span and vertical uncertainty endpoint. No PDF authoring,
replotting, payload changes or external analysis imports are performed.
"""
import argparse
import math
import re
from pathlib import Path
import numpy as np
import pdfplumber
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

TOLERANCE_POINTS = 2e-5


def affine(ticks):
    if len(ticks) < 3:
        raise ValueError('At least three independently labeled ticks required')
    values, positions = np.array(ticks, dtype=float).T
    slope, intercept = np.polyfit(values, positions, 1)
    residual = float(np.max(np.abs(slope*values+intercept-positions)))
    if slope <= 0 or residual > TOLERANCE_POINTS:
        raise ValueError('Nonlinear/ambiguous PDF axis')
    return dict(slope=float(slope), intercept=float(intercept),
                ticks=[list(t) for t in ticks], residual_points=residual)


def axis_contract(page):
    frames = [r for r in page.rects if .5*page.width < r['width'] < .99*page.width
              and .5*page.height < r['height'] < .99*page.height]
    if len(frames) != 1:
        raise ValueError('Ambiguous PDF plotting frame')
    frame = frames[0]
    xticks = sorted(set(l['x0'] for l in page.lines
        if abs(l['x1']-l['x0']) < 1e-6 and abs(l['y0']-frame['y0']) < 1e-6
        and 10 < l['y1']-l['y0'] < 15))
    yticks = sorted(set(l['y0'] for l in page.lines
        if abs(l['y1']-l['y0']) < 1e-6 and abs(l['x0']-frame['x0']) < 1e-6
        and 10 < l['x1']-l['x0'] < 15))
    xlabels, ylabels = [], []
    for w in page.extract_words():
        if not re.fullmatch(r'-?\d+(?:\.\d+)?', w['text']):
            continue
        value = float(w['text'])
        x = (w['x0']+w['x1'])/2
        y = page.height-(w['top']+w['bottom'])/2
        if frame['bottom'] < w['top'] < frame['bottom']+15:
            nearest = min(xticks, key=lambda t:abs(t-x))
            if abs(nearest-x) < 4:
                xlabels.append((value, nearest))
        if frame['x0']-60 < w['x0'] and w['x1'] < frame['x0'] and frame['top'] <= w['top'] <= frame['bottom']:
            nearest = min(yticks, key=lambda t:abs(t-y))
            if abs(nearest-y) < 4:
                ylabels.append((value, nearest))
    return frame, affine(xlabels), affine(ylabels)


def position(axis, value):
    return axis['slope']*value+axis['intercept']


def expected_points(result, kind, view):
    edges = result['bin_edges_gev'] if kind == 'met' else result['pt_edges_gev']
    npt = len(edges)-1
    for i, b in enumerate(result['bins']):
        j = i % npt
        offset = 0 if kind == 'met' else (i//npt-1.5)*3.0
        x = (edges[j]+edges[j+1])/2+offset
        series = ('scale_factor',) if view == 'scale_factor' else ('data', 'mc')
        for tag in series:
            if tag == 'scale_factor':
                y = b['scale_factor']; lo=y-b['scale_factor_uncertainty']; hi=y+b['scale_factor_uncertainty']
            else:
                y = b[tag+'_efficiency']; lo,hi=b[tag+'_interval']
                lo=min(lo,y);hi=max(hi,y)
            if not all(math.isfinite(float(v)) for v in (x,y,lo,hi)):
                raise ValueError('Non-finite measurement')
            yield dict(bin=i,series=tag,x=x,y=y,lo=lo,hi=hi,
                       xlo=edges[j]+offset,xhi=edges[j+1]+offset,
                       marker='rect' if tag=='mc' else 'curve',
                       marker_size=6 if kind=='met' else 5.5)


def verify_pdf(path, result, kind, view):
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError('Expected one-page preserved trigger figure')
        page=pdf.pages[0];frame,xaxis,yaxis=axis_contract(page)
        records=[];used=set()
        for e in expected_points(result,kind,view):
            x,y=position(xaxis,e['x']),position(yaxis,e['y'])
            markers=[(i,o) for i,o in enumerate(page.curves+page.rects)
                if o['object_type']==e['marker']
                and abs(o['width']-e['marker_size'])<1e-6
                and abs(o['height']-e['marker_size'])<1e-6
                and abs((o['x0']+o['x1'])/2-x)<TOLERANCE_POINTS]
            if not markers:raise ValueError('Missing PDF marker: '+str(e))
            marker_index,marker=min(markers,key=lambda io:abs((io[1]['y0']+io[1]['y1'])/2-y))
            if marker_index in used:raise ValueError('PDF marker matched twice')
            used.add(marker_index)
            marker_error=max(abs((marker['x0']+marker['x1'])/2-x),abs((marker['y0']+marker['y1'])/2-y))
            vertical=[l for l in page.lines if abs(l['x0']-x)<TOLERANCE_POINTS and abs(l['x1']-x)<TOLERANCE_POINTS]
            horizontal=[l for l in page.lines if abs(l['y0']-y)<TOLERANCE_POINTS and abs(l['y1']-y)<TOLERANCE_POINTS]
            verr=min((max(abs(l['y0']-position(yaxis,e['lo'])),abs(l['y1']-position(yaxis,e['hi']))) for l in vertical),default=float('inf'))
            herr=min((max(abs(l['x0']-position(xaxis,e['xlo'])),abs(l['x1']-position(xaxis,e['xhi']))) for l in horizontal),default=float('inf'))
            if max(marker_error,verr,herr)>TOLERANCE_POINTS:
                raise ValueError('PDF numeric binding mismatch: '+str(dict(expected=e,marker_error=marker_error,vertical_error=verr,horizontal_error=herr)))
            records.append(dict(bin=e['bin'],series=e['series'],marker_error_points=marker_error,
                                uncertainty_error_points=verr,bin_extent_error_points=herr))
        return dict(status='verified_all_markers_bins_and_uncertainty_endpoints',
                    points_checked=len(records),xaxis=xaxis,yaxis=yaxis,
                    tolerance_pdf_points=TOLERANCE_POINTS,
                    max_residual_pdf_points=max(max(r[k] for k in ('marker_error_points','uncertainty_error_points','bin_extent_error_points')) for r in records),
                    records=records)


def run(output):
    provenance=ROOT/'validation/reference_assets_20260923/provenance/triggers'
    previous=read_json(provenance/'verification.json')
    results=[]
    for item in previous['records']:
        measurement=internal_path(ROOT/item['measurement'])
        payload=internal_path(ROOT/item['payload'])
        if sha256(measurement)!=item['measurement_sha256'] or sha256(payload)!=item['payload_sha256']:
            raise ValueError('Previously verified measurement/payload changed')
        result=read_json(measurement)
        for fig in item['figures']:
            path=internal_path(ROOT/'plots/reference_assets_20260923'/fig['an_path'])
            if sha256(path)!=fig['sha256']:raise ValueError('Preserved reference figure changed')
            view='scale_factor' if 'scale_factor' in path.stem else 'efficiency'
            check=verify_pdf(path,result,item['kind'],view)
            check.update(an_path=fig['an_path'],pdf_sha256=fig['sha256'],
                         measurement=item['measurement'],measurement_sha256=item['measurement_sha256'],
                         payload=item['payload'],payload_sha256=item['payload_sha256'])
            results.append(check)
    summary=dict(schema='trotasr_trigger_pdf_numeric_binding_v1',status='verified',
                 source_sha256=sha256(Path(__file__)),pdfs_verified=len(results),
                 measurement_to_payload_verification=str((provenance/'verification.json').relative_to(ROOT)),
                 no_pdf_authored_or_edited=True,no_payload_changed=True,
                 visual_qa='previous Poppler rendering and full-page inspection retained; independent numeric binding only',
                 full_workflow_complete=False,records=results)
    write_json(internal_path(output),summary)
    print(__import__('json').dumps({k:v for k,v in summary.items() if k!='records'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)

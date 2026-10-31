"""Bind preserved b-tag PNG annotations to adopted efficiency cells, read-only.

The legacy display uses geometric pT centers on a log axis, arithmetic eta
centers, and three-decimal annotations. Recover the frame from its black
spines, verify annotation positions and values, and check unsupported cells
remain white/unannotated. This verifies displayed precision, not hidden digits.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256

POSITION_TOLERANCE_PIXELS = 5.0


def contiguous_groups(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(indices) else []


def find_frame(rgb):
    height, width = rgb.shape[:2]
    black = np.all(rgb < 30, axis=2)
    columns = np.flatnonzero(black[int(.12*height):int(.82*height)].sum(axis=0) > .60*height)
    groups = [g for g in contiguous_groups(columns) if len(g)]
    if len(groups) != 4:
        raise ValueError('Expected plot and colorbar vertical spines')
    left, right = [float(np.mean(g)) for g in groups[:2]]
    rows = np.flatnonzero(black[:, int(left):int(right)+1].sum(axis=1) > .98*(right-left))
    ygroups = [g for g in contiguous_groups(rows) if len(g)]
    if len(ygroups) != 2:
        raise ValueError('Ambiguous horizontal plot spines')
    top, bottom = [float(np.mean(g)) for g in ygroups]
    if not (.5*width < right-left < .8*width and .5*height < bottom-top < .8*height):
        raise ValueError('Unexpected plot dimensions')
    return dict(left=left, right=right, top=top, bottom=bottom)


def verify_panel(path, panel, observations):
    rgb = np.array(Image.open(path).convert('RGB'))
    height, width = rgb.shape[:2]
    frame = find_frame(rgb)
    pt, eta = panel['pt_edges'], panel['abseta_edges']
    if pt != [20,30,50,70,100,140,200,300,600,1000] or eta != [0,1.4,2,2.5]:
        raise ValueError('Unexpected reference bin boundaries')
    used, records = set(), []
    for i in range(1, len(pt)-1):
        p = math.sqrt(pt[i]*pt[i+1])
        y = frame['bottom']-(math.log(p/30)/math.log(1000/30))*(frame['bottom']-frame['top'])
        for j in range(len(eta)-1):
            x = frame['left']+((eta[j]+eta[j+1])/2/2.5)*(frame['right']-frame['left'])
            value = panel['efficiency'][i][j]
            matches=[]
            for index, observation in enumerate(observations):
                ox=(observation['x']+observation['width']/2)*width
                oy=(1-observation['y']-observation['height']/2)*height
                residual=max(abs(ox-x),abs(oy-y))
                if residual <= POSITION_TOLERANCE_PIXELS:
                    matches.append((index,observation,residual))
            entry=dict(pt_bin=[pt[i],pt[i+1]],abseta_bin=[eta[j],eta[j+1]],efficiency=value)
            if value is None:
                if panel['passed'][i][j]+panel['failed'][i][j] != 0:
                    raise ValueError('Unsupported efficiency without zero denominator')
                if matches:raise ValueError('Unsupported cell has a numeric annotation')
                patch=rgb[int(y)-12:int(y)+13,int(x)-12:int(x)+13]
                if not np.all(patch == 255):raise ValueError('Unsupported cell is not blank white')
                entry.update(status='verified_zero_denominator_blank',white_patch_pixels=int(patch.shape[0]*patch.shape[1]))
            else:
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid efficiency')
                if len(matches)!=1:raise ValueError('Missing/ambiguous numeric annotation at '+str(entry))
                index,observation,residual=matches[0]
                if index in used:raise ValueError('Annotation reused')
                used.add(index)
                if observation['confidence'] < .9:raise ValueError('Low OCR confidence')
                expected=format(value,'.3f')
                if observation['text'] != expected:
                    raise ValueError('Numeric annotation mismatch: '+str(dict(expected=expected,actual=observation['text'],cell=entry)))
                entry.update(status='verified_at_displayed_three_decimal_precision',text=observation['text'],position_residual_pixels=residual)
            records.append(entry)
    if len(used)!=len(observations):raise ValueError('Unmatched numeric annotation')
    return dict(an_path=panel['an_path'],image_sha256=sha256(path),frame_pixels=frame,
        status='verified_displayed_cells_and_empty_support',cells=len(records),
        numeric_annotations=len(used),blank_zero_denominator_cells=len(records)-len(used),
        max_position_residual_pixels=max((r.get('position_residual_pixels',0) for r in records)),
        records=records)


def run(output):
    base=ROOT/'validation/reference_assets_20260923/provenance/btag_efficiencies'
    data=read_json(base/'plot_data.json');ocr=read_json(base/'png_numeric_ocr.json')
    if data['tagger']!='UParTAK4' or data['years']!=[2024,2025]:raise ValueError('Unexpected payload')
    for rel in data['source_payloads']:
        if sha256(internal_path(ROOT/rel))!=data['source_payload_sha256']:raise ValueError('Adopted payload changed')
    if sha256(ROOT/'workflows/export_btag_reference_data.py')!=data['exporter_sha256']:
        raise ValueError('Exporter changed')
    figures={str(Path(p['path']).relative_to('TROTASR/plots/reference_assets_20260923')):p for p in ocr['panels']}
    rows=read_json(ROOT/'validation/reference_assets_20260923/visual_qa.json')['records']
    results=[]
    for panel in data['panels']:
        rel=panel['an_path'];path=ROOT/'plots/reference_assets_20260923'/rel
        original=next(r for r in rows if r['an_path']==rel)
        expected=original['sha256']
        if sha256(path)!=expected:raise ValueError('Preserved reference image changed')
        results.append(verify_panel(path,panel,figures[rel]['observations']))
    result=dict(schema='trotasr_btag_png_numeric_binding_v1',
        status='all_15_panels_bound_to_adopted_payload',source_modified=False,
        years=data['years'],tagger=data['tagger'],source_payload_sha256=data['source_payload_sha256'],
        source_plot_data_sha256=sha256(base/'plot_data.json'),ocr_sha256=sha256(base/'png_numeric_ocr.json'),
        verifier_sha256=sha256(Path(__file__)),precision='three decimal places, exactly as displayed',
        position_tolerance_pixels=POSITION_TOLERANCE_PIXELS,
        displayed_pt_range_gev=[30,1000],source_pt_20_to_30_not_displayed=True,
        panels=results,figures_checked=len(results),
        numeric_annotations_checked=sum(r['numeric_annotations'] for r in results),
        blank_zero_denominator_cells_checked=sum(r['blank_zero_denominator_cells'] for r in results),
        source_method_note='Legacy annotate() uses geometric pT and arithmetic eta centers; no renderer or payload changes.')
    write_json(internal_path(output),result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='panels'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)

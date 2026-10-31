"""Read-only binding of existing low-pT lepton figures to archived measurements.

Consumes an inward-migrated compact provenance record, never an external runtime
analysis path or Git checkout. Does not render, author, change or fit anything.
Missing applied-policy points are reported, not treated as a successful check.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pdfplumber
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.verify_trigger_reference_figures import axis_contract, position

TOLERANCE_POINTS = 2e-5
COLORS = [0, (.8901960784, .1019607843, .1098039216),
          (.2156862745, .4941176471, .7215686275)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def color_equal(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(abs(x-y) < 1e-9 for x, y in zip(a, b))
    return a == b


def summary_points(page, kind, view):
    """Read actual centers and full error-bar endpoints using labeled axes."""
    frame, xa, ya = axis_contract(page)
    inverse = lambda a, x: (x-a['intercept'])/a['slope']
    points = []
    for i in range(6 if kind == 'electron' else 3):
        eta, pt = divmod(i, 2) if kind == 'electron' else (i, 0)
        for series in (('sf',) if view == 'scale_factor' else ('data', 'mc')):
            x = [6., 8.5][pt] if kind == 'electron' else [.45, 1.05, 1.8][eta]
            if kind == 'electron' and series != 'sf':
                x += (eta-1)*.08 + (-.018 if series == 'data' else .018)
            shape = 'rect' if series == 'mc' or (series == 'sf' and kind == 'electron' and eta == 1) else 'curve'
            xp = position(xa, x)
            markers = [o for o in page.curves+page.rects
                       if o['object_type'] == shape
                       and abs(o['width']-6) < 1e-5 and abs(o['height']-6) < 1e-5
                       and abs((o['x0']+o['x1'])/2-xp) < TOLERANCE_POINTS
                       and (series != 'sf' or kind != 'electron'
                            or color_equal(o['stroking_color'], COLORS[eta]))]
            rec = dict(bin=i, series=series, present=bool(markers))
            if not markers:
                points.append(rec)
                continue
            if len(markers) != 1:
                raise ValueError('Ambiguous summary marker: '+str(rec))
            marker = markers[0]
            mx, my = (marker['x0']+marker['x1'])/2, (marker['y0']+marker['y1'])/2
            vertical = [o for o in page.lines if abs(o['x0']-mx) < TOLERANCE_POINTS
                        and abs(o['x1']-mx) < TOLERANCE_POINTS and o['y0'] <= my <= o['y1']]
            horizontal = [o for o in page.lines if abs(o['y0']-my) < TOLERANCE_POINTS
                          and abs(o['y1']-my) < TOLERANCE_POINTS and o['x0'] <= mx <= o['x1']]
            if not vertical or not horizontal:
                raise ValueError('Missing error bars: '+str(rec))
            v = min(vertical, key=lambda o: abs((o['y0']+o['y1'])/2-my))
            h = min(horizontal, key=lambda o: abs((o['x0']+o['x1'])/2-mx))
            vlo, vhi = v['y0'], v['y1']
            clipped = None
            # Matplotlib's PDF path simplifier clips a very long error bar at
            # the page edge + 1 pt before the axes clip is applied. Retain that
            # fact; recover only a symmetric endpoint from the opposite side.
            if abs((vlo+vhi)/2-my) > TOLERANCE_POINTS and abs(vhi-page.height-1) < TOLERANCE_POINTS:
                clipped = dict(endpoint='hi', visible_pdf_coordinate=vhi)
                vhi = 2*my-vlo
            if max(abs((vlo+vhi)/2-my), abs((h['x0']+h['x1'])/2-mx)) > TOLERANCE_POINTS:
                raise ValueError('Error bars not centered: '+str(rec))
            rec.update(x=inverse(xa, mx), y=inverse(ya, my), lo=inverse(ya, vlo),
                       hi=inverse(ya, vhi), xlo=inverse(xa, h['x0']), xhi=inverse(xa, h['x1']))
            if clipped:
                rec['symmetric_endpoint_reconstructed_after_page_clipping'] = clipped
            points.append(rec)
    return dict(points=points, xaxis=xa, yaxis=ya,
                frame={k: frame[k] for k in ('x0', 'y0', 'x1', 'y1')})


def mass_panel_fingerprint(page):
    """Hash all panel curves/lines including axes; exclude outside legend only.

The four panels are located from rectangle geometry. Full path coordinates,
styles and colors participate, so a changed mass point, fit curve or error bar
fails even when the visible bounding rectangle remains unchanged.
"""
    frames = [r for r in page.rects if .2*page.width < r['width'] < .4*page.width
              and .2*page.height < r['height'] < .4*page.height]
    if len(frames) != 4:
        raise ValueError('Expected four mass-fit panels')
    edges = sorted([{k:r[k] for k in ('x0', 'y0', 'x1', 'y1')} for r in frames],
                   key=lambda r:(r['y0'], r['x0']))
    right = max(r['x1'] for r in frames)
    keys = ('x0', 'y0', 'x1', 'y1', 'pts', 'path', 'linewidth', 'stroke', 'fill',
            'stroking_color', 'non_stroking_color', 'dash')
    geometry = {typ:[{k:o.get(k) for k in keys} for o in getattr(page, typ)
                    if o['x0'] <= right+3] for typ in ('curves', 'lines')}
    texts = []
    for f in edges:
        # Include tick labels and scientific-notation multipliers: those may
        # straddle the frame by font-metric fractions even with identical data.
        cropped = page.crop((f['x0']-38, page.height-f['y1']-18,
                             f['x1']+2, page.height-f['y0']+20))
        texts.append(' '.join(cropped.extract_text().split()))
    return dict(frames=edges, curves=len(geometry['curves']), lines=len(geometry['lines']),
                geometry_sha256=digest(geometry), panel_text=texts)


def compare_summary(actual, expected, rounded_digits=None):
    source = {(p['bin'], p['series']):p for p in expected['points']}
    target = {(p['bin'], p['series']):p for p in actual['points']}
    if len(source) != len(expected['points']) or set(source) != set(target):
        raise ValueError('Summary point identity mismatch')
    checks, problems, clipped = [], [], []
    for key, original in source.items():
        a, e = target[key], dict(original)
        for label, point in [('actual', a), ('canonical', e)]:
            if 'symmetric_endpoint_reconstructed_after_page_clipping' in point:
                clipped.append(dict(bin=key[0], series=key[1], source=label,
                                    detail=point['symmetric_endpoint_reconstructed_after_page_clipping']))
        if a['present'] != e['present']:
            problems.append(dict(bin=key[0], series=key[1], issue='missing_policy_point' if e['present'] else 'unexpected_point'))
            continue
        if not e['present']:
            continue
        if rounded_digits is not None:
            err = round((e['hi']-e['lo'])/2, rounded_digits)
            e['y'] = round(e['y'], rounded_digits)
            e['lo'], e['hi'] = e['y']-err, e['y']+err
        residual = max(abs(a[k]-e[k])*actual['xaxis' if k.startswith('x') else 'yaxis']['slope']
                       for k in ('x', 'xlo', 'xhi', 'y', 'lo', 'hi'))
        if not math.isfinite(residual) or residual > TOLERANCE_POINTS:
            problems.append(dict(bin=key[0], series=key[1], issue='numeric_mismatch',
                                 actual=a, expected=e, residual_pdf_points=residual))
        else:
            checks.append(dict(bin=key[0], series=key[1], residual_pdf_points=residual))
    return dict(status='verified' if not problems else 'reference_numeric_review_required',
                checked_points=len(checks), expected_points=sum(p['present'] for p in source.values()),
                tolerance_pdf_points=TOLERANCE_POINTS, rounded_display_digits=rounded_digits,
                max_passing_residual_pdf_points=max((c['residual_pdf_points'] for c in checks), default=None),
                checks=checks, problems=problems, clipped_endpoint_audit=clipped)


def verify_summary_measurement(summary, measurement, view):
    """Cross-check migrated canonical PDF values with fit result, not the AN PDF."""
    checked = 0
    for p in summary['points']:
        if not p['present']:
            continue
        b = measurement['bins'][p['bin']]
        if view == 'scale_factor':
            if not b['valid']:
                continue  # Applied forward unity policy is bound to payload below.
            y, err = b['scale_factor'], b['scale_factor_uncertainty']
        else:
            fit = b['nominal'][p['series']]
            y, err = fit['efficiency'], fit['efficiency_stat_uncertainty']
        if max(abs(p['y']-y), abs(p['lo']-(y-err)), abs(p['hi']-(y+err))) > 5e-8:
            raise ValueError('Archived canonical PDF differs from measurement')
        checked += 1
    return checked


def verify_canonical_calibration(calibration, figures, payload):
    selected = [f for f in figures if f['year'] == calibration['year']
                and f['kind'] == calibration['kind'] and f['view'] != 'mass_fit']
    summaries = {f['view']:f['canonical_summary']['points'] for f in selected}
    correction = payload['corrections'][0]
    values = {c['key']:c['value']['content'] for c in correction['data']['content']}
    eff = {(p['bin'], p['series']):p for p in summaries['efficiency']}
    policy_cells = [4, 5] if calibration['kind'] == 'electron' else []
    if policy_cells and ('unity' not in correction['description']
                         or any(values['nominal'][i] != 1 for i in policy_cells)):
        raise ValueError('Missing explicit adopted electron forward policy')
    checked, omitted = [], []
    for p in summaries['scale_factor']:
        i = p['bin']
        if not p['present']:
            if i not in policy_cells:
                raise ValueError('Non-policy SF point absent from canonical figure')
            omitted.append(i)
            continue
        if max(abs(p[k]-values[v][i]) for k, v in [('y','nominal'), ('lo','down'), ('hi','up')]) > 5e-8:
            raise ValueError('Canonical figure and adopted payload differ')
        if i not in policy_cells:
            ratio = eff[(i,'data')]['y']/eff[(i,'mc')]['y']
            if abs(ratio-values['nominal'][i]) > 5e-8:
                raise ValueError('Efficiency ratio and scale factor differ')
        checked.append(i)
    return dict(year=calibration['year'], kind=calibration['kind'],
                canonical_sf_points_bound_to_payload=checked,
                adopted_unity_policy_cells=policy_cells,
                canonical_measurement_only_omitted_policy_cells=omitted,
                archived_and_adopted_payload_decoded_sha256=calibration['decoded_payload_sha256'])


def reference_summary_contract(record, calibration):
    """Do not confuse measured SF points with the applied forward-unity policy.

Omissions require an explicit, hash-bound AN definition and matching invalid
fit-result bins. Missing valid measurements still fail the ordinary comparison.
"""
    expected = copy.deepcopy(record['canonical_summary'])
    scope = record.get('reference_scope')
    if scope is None:
        return expected, {}
    if scope['kind'] != 'valid_measurement_only' or record['view'] != 'scale_factor':
        raise ValueError('Unsupported reference scope')
    evidence = internal_path(ROOT/scope['definition_audit'])
    if sha256(evidence) != scope['definition_audit_sha256']:
        raise ValueError('Changed reference-scope evidence')
    definition = read_json(evidence)
    if (record['an_path'] not in definition['measurement_only_figures']
            or definition['scope'] != 'valid_measurement_only'):
        raise ValueError('Reference scope not established by AN definition')
    invalid = [i for i, b in enumerate(calibration['measurement']['bins']) if b['valid'] is False]
    if sorted(scope['omitted_invalid_bins']) != invalid:
        raise ValueError('Omitted bins differ from actual invalid measurement bins')
    excluded = []
    for p in expected['points']:
        if p['bin'] in invalid:
            if p['series'] != 'sf': raise ValueError('Non-SF omission')
            if p['present']:
                excluded.append(dict(p))
            p['present'] = False
    return expected, dict(scope='valid_measurement_only',
        applied_policy_not_a_measurement=True, omitted_invalid_fit_bins=invalid,
        archived_applied_policy_points_not_shown=excluded,
        definition_audit=scope['definition_audit'], definition_audit_sha256=scope['definition_audit_sha256'])


def run(provenance, output):
    source = read_json(internal_path(provenance))
    payloads, calibration_checks = {}, []
    for calibration in source['calibrations']:
        p = internal_path(ROOT/calibration['payload'])
        data = read_json(p)
        if sha256(p) != calibration['payload_sha256'] or digest(data) != calibration['decoded_payload_sha256']:
            raise ValueError('Adopted calibration payload changed: '+str(p))
        calibration_checks.append(verify_canonical_calibration(calibration, source['figures'], data))
        payloads[(calibration['year'], calibration['kind'])] = calibration
    results = []
    for record in source['figures']:
        path = internal_path(ROOT/'plots/reference_assets_20260923'/record['an_path'])
        if sha256(path) != record['reference_sha256']:
            raise ValueError('Preserved reference PDF changed: '+str(path))
        calibration = payloads[(record['year'], record['kind'])]
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) != 1:
                raise ValueError('Unexpected page count')
            page = pdf.pages[0]
            if record['view'] == 'mass_fit':
                check = mass_panel_fingerprint(page)
                if check != record['canonical_panel_fingerprint']:
                    raise ValueError('Mass-fit panel mismatch: '+str(path))
                check.update(status='verified', method='exact_mass_panel_geometry_and_text')
            else:
                n = 0
                if 'measurement' in calibration:
                    n = verify_summary_measurement(record['canonical_summary'], calibration['measurement'], record['view'])
                actual = summary_points(page, record['kind'], record['view'])
                digits = 4 if record['year'] == 2024 and record['kind'] == 'muon' and record['view'] == 'scale_factor' else None
                expected, scope_audit = reference_summary_contract(record, calibration)
                check = compare_summary(actual, expected, digits)
                check.update(method='independent_labeled_axes_and_all_error_endpoints',
                             canonical_points_bound_to_fit_result=n)
                if scope_audit:
                    check['reference_scope_audit'] = scope_audit
        check.update(an_path=record['an_path'], pdf_sha256=record['reference_sha256'],
                     archived_pdf_sha256=record['source_sha256'], payload=calibration['payload'],
                     payload_sha256=calibration['payload_sha256'])
        results.append(check)
    verified = sum(r['status'] == 'verified' for r in results)
    report = dict(schema='trotasr_lowpt_reference_numeric_binding_v1',
                  status='verified' if verified == len(results) else 'reference_numeric_review_required',
                  figures_checked=len(results), figures_verified=verified,
                  figures_requiring_review=len(results)-verified, records=results,
                  calibration_checks=calibration_checks,
                  provenance=str(internal_path(provenance).relative_to(ROOT)),
                  provenance_sha256=sha256(provenance), verifier_sha256=sha256(Path(__file__)),
                  no_pdf_authored_or_edited=True, no_physics_payload_changed=True,
                  no_AN_changed=True, full_workflow_complete=False)
    write_json(internal_path(output), report)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.provenance, args.output)

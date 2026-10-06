#!/usr/bin/env python3
"""Signal production cross-section uncertainty, separate from fit nuisances."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

POLICY = ('Signal production cross-section uncertainty changes the exclusion reference; '
          'it is not a profiled signal-rate nuisance and is not combined with expected quantile bands. '
          'Signal acceptance uncertainties are separate. No observed SR data are used.')
SIGNAL_XSEC_NUISANCE = 'CMS_NPS26012_cross_section_stop_13p6TeV'
LEGACY_SIGNAL_XSEC_NUISANCE = 'theory_xsec_stop_13p6TeV'
PROFILED_POLICY = ('User-adopted 2026-09-18: signal production cross-section uncertainty is '
                  'profiled as one asymmetric lnN nuisance, correlated across years and regions. '
                  'Do not apply a second external theory shift to these limits. '
                  'Acceptance uncertainties are separate. No observed SR data are used.')


def profiled_theory(theory):
    return dict(theory, policy=PROFILED_POLICY, profiled_in_likelihood=True,
                nuisance=SIGNAL_XSEC_NUISANCE)


def add_signal_xsec_lnN(card, mass, theory):
    """Add the AN xsec uncertainty to signal columns only; retain all shapes."""
    record = theory['records'][str(int(mass))]
    down = 1. - float(record['uncertainty_down_relative'])
    up = 1. + float(record['uncertainty_up_relative'])
    if not (0. < down <= 1. <= up) or not all(math.isfinite(x) for x in (down, up)):
        raise ValueError('invalid lnN factors at mass '+str(mass))
    lines = card.splitlines()
    if any(line.split() and line.split()[0] in
           (SIGNAL_XSEC_NUISANCE, LEGACY_SIGNAL_XSEC_NUISANCE) for line in lines):
        raise ValueError('signal xsec nuisance already present')
    process_lines = [line.split()[1:] for line in lines if line.startswith('process ')]
    names = next((row for row in process_lines if 'signal' in row), None)
    if names is None:
        raise ValueError('card has no signal columns')
    token = '{:.8g}/{:.8g}'.format(down, up)
    nuisance = SIGNAL_XSEC_NUISANCE+' lnN '+' '.join(token if name == 'signal' else '-' for name in names)
    for i, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == 'kmax' and fields[1] != '*':
            fields[1] = str(int(fields[1])+1)
            lines[i] = ' '.join(fields)
    lines.append(nuisance)
    return '\n'.join(lines)+'\n'


def load_theory(table, an_source=None):
    payload = json.loads(Path(table).read_text())
    records = {}
    for r in payload['records']:
        mass = int(r['mStop'])
        sigma = float(r['xsec_pb'])
        up = float(r['uncertainty_up_relative'])
        down = float(r['uncertainty_down_relative'])
        if not all(math.isfinite(v) for v in (sigma, up, down)) or sigma <= 0 or min(up, down) < 0:
            raise ValueError('invalid theory input at mass ' + str(mass))
        if str(mass) in records:
            raise ValueError('duplicate theory mass')
        records[str(mass)] = dict(xsec_pb=sigma, uncertainty_up_relative=up, uncertainty_down_relative=down,
                                  xsec_up_pb=sigma*(1+up), xsec_down_pb=sigma*(1-down) if down < 1 else None)
    result = dict(policy=POLICY, records=records, table_sha256=hashlib.sha256(Path(table).read_bytes()).hexdigest(),
                  reference='CMS AN-26-090, signal sample cross-section table; SUS-19-010 exclusion convention')
    if an_source:
        text = Path(an_source).read_text()
        section = text.split('\\label{tab:signal_sample_paths}', 1)[1].split('\\end{longtable}', 1)[0]
        count = 0
        for line in section.splitlines():
            if not re.match(r'^\d+\s*&', line):
                continue
            cols = line.split('&')
            if len(cols) != 6:
                raise ValueError('unrecognized signal xsec table row')
            mass, sigma, percent = int(cols[0]), float(cols[4]), float(cols[5].split('\\')[0])
            r = records[str(mass)]
            if not math.isclose(r['xsec_pb'], sigma, rel_tol=1e-9) or not math.isclose(r['uncertainty_up_relative']*100, percent, abs_tol=1e-9):
                raise ValueError('AN / xsec payload disagreement at ' + str(mass))
            count += 1
        if count < 10:
            raise ValueError('signal theory table not found')
        result.update(an_table_checked_rows=count, an_sha256=hashlib.sha256(Path(an_source).read_bytes()).hexdigest())
    return result


def interpret_limit(mu95, mass, theory):
    """Fixed experimental sigma95; only its ratio to sigma_theory changes."""
    if not math.isfinite(mu95) or mu95 <= 0:
        raise ValueError('invalid expected upper limit')
    r = theory['records'][str(int(mass))]
    if r['uncertainty_down_relative'] >= 1:
        raise ValueError('no positive theory -1 sigma reference at mass ' + str(mass))
    if theory.get('profiled_in_likelihood'):
        return dict(mStop=int(mass), **r, mu95_nominal=mu95, sigma95_pb=mu95*r['xsec_pb'],
                    profiled_in_likelihood=True, nuisance=SIGNAL_XSEC_NUISANCE,
                    external_theory_shift_applied=False, policy=theory['policy'])
    return dict(mStop=int(mass), **r, mu95_nominal=mu95, sigma95_pb=mu95*r['xsec_pb'],
                mu95_theory_up=mu95/(1+r['uncertainty_up_relative']),
                mu95_theory_down=mu95/(1-r['uncertainty_down_relative']),
                policy=theory['policy'])


def annotate_expected_grid(payload, theory):
    """Enrich exported limits without changing expected quantiles or plot style."""
    points = {}
    for key, original in payload['points'].items():
        record = dict(original)
        interpretation = interpret_limit(float(record['expected']), int(record['mStop']), theory)
        record['signal_theory'] = interpretation
        points[key] = record
    return dict(payload, points=points, signal_theory_policy=theory['policy'],
                signal_theory_table_sha256=theory['table_sha256'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', type=Path, required=True)
    ap.add_argument('--an-source', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    result = load_theory(a.table, a.an_source)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(mass_points=len(result['records']), an_rows=result.get('an_table_checked_rows'),
                          mStop1200=result['records']['1200'])))

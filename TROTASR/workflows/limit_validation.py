"""Existing Combine expected-limit numerical validation, internal copy.

The crossing check retains the adopted retry_failed_limits_hep2.py policy.
The user explicitly removed the Asimov-warning veto on 2026-09-29; warnings
and fitted r are diagnostics, not a numerical failure criterion.
No five-quantile ROOT file is accepted without measured crossing evidence.
"""
import math
import re
import statistics

QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
REL_ACC, ABS_ACC = 0.0025, 0.00001
SCAN = re.compile(r'At r = ([\d.eE+\-]+):\s*delta\(nll\) = ([\d.eE+\-]+)')


def asimov_diagnostics(text):
    warnings = [line.strip() for line in text.splitlines()
                if 'Best fit of asimov dataset is at' in line]
    blocks = text.split('Fit to asimov dataset:')
    fit = blocks[1].split('Will search for NLL crossing by bisection')[0] if len(blocks) == 2 else ''
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    rows = re.findall(r'^\s*r\s+('+number+r')\s+('+number+r')\s+\+/-\s+('+number+r')', fit, re.M)
    return dict(warning_lines=warnings, fitted_r=float(rows[0][1]) if len(rows) == 1 else None,
                diagnostic_only=True, acceptance_threshold=None)


def crossing_check(text, values, tolerance):
    """Require a measured target crossing, not merely five finite outputs.

    Combine logs the five bisection searches in median,-2,-1,+1,+2 order.
    Accept its documented NLL stopping tolerance or a sufficiently narrow
    bracketing interval containing the returned limit. An interrupted early
    scan retaining a stale success flag therefore cannot pass this check.
    """
    blocks = text.split('Will search for NLL crossing by bisection')[1:]
    if len(blocks) != 5:
        return False, {'reason': 'expected_five_crossing_searches', 'count': len(blocks)}
    evidence = []
    normal = statistics.NormalDist()
    for block, quantile in zip(blocks, (0.5, 0.025, 0.16, 0.84, 0.975)):
        scans = [(float(r), float(n)) for r, n in SCAN.findall(block)]
        target = 0.5 * (normal.inv_cdf(quantile) + normal.inv_cdf(1 - quantile * 0.05)) ** 2
        result = values[quantile]
        rtol = max(REL_ACC * result, ABS_ACC) + 0.000002
        close = any(abs(r - result) <= 0.000002 and abs(n - target) <= 0.05 * tolerance + 0.00001
                    for r, n in scans)
        below = [r for r, n in scans if n <= target and r <= result + 0.000002]
        above = [r for r, n in scans if n >= target and r >= result - 0.000002]
        width = min(above) - max(below) if below and above else None
        bracketed = width is not None and 0 <= width <= 2 * rtol
        evidence.append(dict(quantile=quantile, result=result, target_delta_nll=target,
                             scan_evaluations=len(scans), bracket_width=width,
                             passed=bool(close or bracketed)))
    return all(x['passed'] for x in evidence), {'crossings': evidence}


def validate(path, log, tolerance):
    import uproot
    try:
        with uproot.open(path) as root:
            rows = root['limit'].arrays(['quantileExpected', 'limit'], library='np')
        if len(rows['limit']) != 5:
            raise ValueError('not five expected quantiles')
        values = {}
        for q in QUANTILES:
            matched = [float(v) for a, v in zip(rows['quantileExpected'], rows['limit'])
                       if abs(float(a) - q) < 0.00001]
            if len(matched) != 1 or not math.isfinite(matched[0]) or matched[0] <= 0:
                raise ValueError('invalid quantile')
            values[q] = matched[0]
        if any(values[a] >= values[b] for a, b in zip(QUANTILES, QUANTILES[1:])):
            raise ValueError('non-increasing quantiles')
        text = log.read_text(errors='replace')
        passed, evidence = crossing_check(text, values, tolerance)
        return passed, dict(evidence, quantiles=values, asimov=asimov_diagnostics(text))
    except Exception as exc:
        return False, {'reason': str(exc)}

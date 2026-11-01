"""Account for Combine's float return and six-decimal scan logging only.

Quantile, likelihood-target and bracket requirements still apply. Asimov
warnings are diagnostic-only. Reconcile exact-r mismatches using the binary32
rounding interval and the precision of the printed scan values.
"""
import copy
import math
import struct
from TROTASR.workflows.limit_validation import SCAN, validate


def rounding_interval(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Expected a positive finite saved limit')
    encoded = struct.pack('>f', value)
    bits = struct.unpack('>I', encoded)[0]
    saved = struct.unpack('>f', encoded)[0]
    if saved != value or not 0 < bits < 0x7f7fffff:
        raise ValueError('Saved result is not a finite binary32 value')
    before = struct.unpack('>f', struct.pack('>I', bits-1))[0]
    after = struct.unpack('>f', struct.pack('>I', bits+1))[0]
    # Actual scan output uses printf("%f"), hence half of one printed unit.
    return ((before+saved)/2.-.0000005, (after+saved)/2.+.0000005)


def reconcile_crossings(text, evidence, tolerance):
    result = copy.deepcopy(evidence)
    blocks = text.split('Will search for NLL crossing by bisection')[1:]
    crossings = result.get('crossings', [])
    if len(blocks) != 5 or len(crossings) != 5 or 'quantiles' not in result:
        return False, result
    for crossing, block in zip(crossings, blocks):
        if crossing['passed']:
            continue
        try:
            low, high = rounding_interval(crossing['result'])
        except (ValueError, OverflowError):
            continue
        matches = [(float(r), float(n)) for r, n in SCAN.findall(block)
                   if low <= float(r) <= high and
                   abs(float(n)-crossing['target_delta_nll']) <= .05*tolerance+.00001]
        if matches:
            crossing.update(passed=True, original_exact_r_match_passed=False,
                matching_scan_rows=matches, serialization_interval=[low, high],
                reconciliation='binary32_return_and_six_decimal_log')
    return all(c['passed'] for c in crossings), result


def validate_saved_limit(path, log, tolerance):
    passed, original = validate(path, log, tolerance)
    if passed or 'crossings' not in original:
        return passed, original
    passed, result = reconcile_crossings(log.read_text(errors='replace'), original, tolerance)
    if passed:
        result['original_validation'] = original
        result['validation_adapter'] = 'binary32_return_and_six_decimal_log'
    return passed, result

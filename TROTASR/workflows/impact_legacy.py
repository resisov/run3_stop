"""Preserved impact numerical kernels; all file hashes use internal paths."""
import math
import threading
from types import SimpleNamespace
from TROTASR.utils.io import sha256
MASS = '120'
ROOT_LOCK = threading.Lock()
post = SimpleNamespace(overlay=SimpleNamespace(sha=sha256))


def check_endpoints(values, bounds):
    if len(values) != 3 or not all(math.isfinite(x) for row in values for x in row):
        raise ValueError('expected three finite fit entries')
    nominal, lower, upper = [row[1] for row in values]
    tol = 1e-6 * max(1., abs(nominal), abs(lower), abs(upper))
    if nominal == lower == upper:
        raise ValueError('identical parameter endpoints')
    if not lower <= nominal + tol or not nominal <= upper + tol:
        raise ValueError('endpoints do not bracket nominal')
    if lower == nominal and not math.isclose(nominal, bounds[0], rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('zero-width lower endpoint away from boundary')
    if upper == nominal and not math.isclose(nominal, bounds[1], rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('zero-width upper endpoint away from boundary')


def fit_command(workspace, name, expect_signal, strategy):
    initial = name == 'r'
    tag = '_initialFit_Test' if initial else '_paramFit_Test_' + name
    command = ['combine', '-M', 'MultiDimFit', '-n', tag, '--algo',
               'singles' if initial else 'impact', '--redefineSignalPOIs', 'r']
    if not initial:
        command += ['-P', name, '--floatOtherPOIs', '1', '--saveInactivePOI', '1']
    command += ['-m', MASS, '-d', str(workspace), '--robustFit', '1',
                '--cminDefaultMinimizerStrategy', str(strategy), '-t', '-1',
                '--expectSignal', str(expect_signal), '--setParameterRanges',
                'r=%s,20' % ('-20' if expect_signal == 0 else '0'), '--saveFitResult']
    return command, tag


def validate_fit(folder, name, tag, bounds):
    import ROOT
    result = dict(name=name, valid=False)
    source = fit_source = None
    with ROOT_LOCK:
        try:
            path = folder / ('higgsCombine%s.MultiDimFit.mH%s.root' % (tag, MASS))
            source = ROOT.TFile.Open(str(path))
            if not source or source.IsZombie():
                raise ValueError('unreadable fit ROOT')
            tree = source.Get('limit')
            if not tree:
                raise ValueError('missing limit tree')
            values = [[float(row.r), float(getattr(row, name)), float(row.deltaNLL)] for row in tree]
            check_endpoints(values, bounds)
            if name == 'r':
                fit_source = ROOT.TFile.Open(str(folder / ('multidimfit%s.root' % tag)))
                fit = fit_source.Get('fit_mdf') if fit_source and not fit_source.IsZombie() else None
                if not fit:
                    raise ValueError('missing saved RooFitResult')
                result.update(fit_status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()))
                if result['fit_status'] != 0 or result['covariance_quality'] < 2 or not math.isfinite(result['edm']):
                    raise ValueError('nominal fit convergence/covariance validation failed')
            else:
                # Combine v10.5.1 does not call saveResult for algo=impact.
                # doImpact commits each endpoint only if minim.minimize succeeds.
                result['validation_scope'] = 'three finite bracketed entries; both endpoint minimizations succeeded'
            result.update(valid=True, entries=values, bounds=list(bounds), sha256=post.overlay.sha(path))
        except Exception as error:
            result['error'] = str(error)
        finally:
            if source:
                source.Close()
            if fit_source:
                fit_source.Close()
    return result

"""Run preserved extraction/rendering bodies through the internal boundary."""
import argparse
import runpy
import sys
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('extract_cronly', 'plot_cronly'), required=True)
    a, remaining = p.parse_known_args()
    source = {'extract_cronly': 'extract_cronly_covariance.py',
              'plot_cronly': 'renderers/cronly_pulls.py'}[a.operation]
    sys.argv = [str(ROOT / 'workflows' / source), *remaining]
    runpy.run_path(sys.argv[0], run_name='__main__')

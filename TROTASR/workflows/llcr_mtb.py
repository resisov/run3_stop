"""Render the saved high-dM control-region cut comparison."""

import argparse
import json
from pathlib import Path

from TROTASR.utils.paths import ROOT


OUTPUT = ROOT / 'stats/run3/diagnostics/cr_mtb_2024'


def main():
    from TROTASR.utils.selection_contract import RETIRED_LLCR_MESSAGE
    parser = argparse.ArgumentParser(prog='python -m TROTASR cr-mtb',
        description=RETIRED_LLCR_MESSAGE + ' Only the saved CR diagnostic can be rendered.')
    parser.add_argument('--plot-only', type=Path, required=True, metavar='RESULT_JSON',
                        help='Render the saved diagnostic without opening event inputs')
    parser.add_argument('--plot-output', type=Path, default=OUTPUT, metavar='DIR')
    args = parser.parse_args()
    from TROTASR.utils.renderers.llcr_mtb import render
    with args.plot_only.open() as source:
        render(json.load(source), args.plot_output)


if __name__ == '__main__':
    main()

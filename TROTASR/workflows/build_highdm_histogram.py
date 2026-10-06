"""Histogram an integrated nominal ROOT input with the trial topology policy."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.histogramming import build


def main(default='highdm'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--year', type=int, choices=(2024, 2025))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mode', choices=('highdm', 'lowdm', 'both'), default=default)
    p.add_argument('--chunk-size', type=int, default=2000)
    p.add_argument('--cutflow-config', type=Path, help='Read-only SR cutflows; no histograms or inference rerun')
    p.add_argument('--cutflow-worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.cutflow_config:
        from TROTASR.utils.cutflow import run, worker
        (run if args.cutflow_worker is None else worker)(args.cutflow_config, args.output,
            *(() if args.cutflow_worker is None else (args.cutflow_worker,)))
        return
    if args.input is None or args.year is None:
        p.error('--input and --year are required for histogram production')
    modes = ('highdm', 'lowdm') if args.mode == 'both' else (args.mode,)
    result = build(args.input, args.year, args.output, modes=modes, chunk_size=args.chunk_size)
    print(result['status'], 'events=', result['audit']['events_read'], 'seconds=', round(result['seconds'], 2))


if __name__ == '__main__':
    main()

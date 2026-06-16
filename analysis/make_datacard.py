#!/usr/bin/env python
from __future__ import annotations

from optparse import OptionParser
from pathlib import Path


def sanitize_name(name):
    return name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace('-', '_')


def make_datacard(template_root, output, signal, regions, backgrounds, auto_mc_stats=10):
    template_root = Path(template_root)
    output = Path(output)
    if not template_root.is_file():
        raise RuntimeError('template ROOT does not exist: %s' % template_root)
    if not signal:
        raise RuntimeError('signal must be specified explicitly')
    if not regions:
        raise RuntimeError('at least one region must be specified')
    if not backgrounds:
        raise RuntimeError('at least one background must be specified')
    processes = [signal] + list(backgrounds)
    lines = [
        'imax * number of channels',
        'jmax * number of backgrounds',
        'kmax * number of nuisance parameters',
        '------------',
        'shapes * * %s $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC' % template_root,
        '------------',
        'bin ' + ' '.join(regions),
        'observation ' + ' '.join(['-1'] * len(regions)),
        '------------',
    ]
    bins = []
    procs = []
    ids = []
    rates = []
    for region in regions:
        for i, process in enumerate(processes):
            bins.append(region)
            procs.append(sanitize_name(process))
            ids.append(str(0 if i == 0 else i))
            rates.append('-1')
    lines.append('bin ' + ' '.join(bins))
    lines.append('process ' + ' '.join(procs))
    lines.append('process ' + ' '.join(ids))
    lines.append('rate ' + ' '.join(rates))
    lines.append('------------')
    lines.append('* autoMCStats %s' % auto_mc_stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--template-root', dest='template_root')
    parser.add_option('--output', dest='output')
    parser.add_option('--signal', dest='signal')
    parser.add_option('--regions', dest='regions')
    parser.add_option('--backgrounds', dest='backgrounds')
    parser.add_option('--auto-mc-stats', dest='auto_mc_stats', type='int', default=10)
    opts, _ = parser.parse_args()
    regions = [x.strip() for x in (opts.regions or '').split(',') if x.strip()]
    backgrounds = [x.strip() for x in (opts.backgrounds or '').split(',') if x.strip()]
    make_datacard(opts.template_root, opts.output, opts.signal, regions, backgrounds, opts.auto_mc_stats)

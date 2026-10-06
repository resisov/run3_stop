"""Read-only current-model SR cutflows, through build_highdm_histogram.py.

Count the retained MC population before the topology decision. Reuse the
completed corrected-central histogram audit for the final topology decision;
never re-infer TROTA/GNN or invent intermediate data-driven normalizations.
"""
from pathlib import Path
import ast
import contextlib
from functools import lru_cache
import fcntl
import gzip
import html
import json
import os
import subprocess
import sys
import time
import traceback
import awkward as ak
import numpy as np
import uproot
from .paths import ROOT, internal_path
from .io import read_json, write_json, sha256, validate_root
from .normalization import dataset_label
from .signal_models import signal_topology, signal_mass_key
from .shape_kinematics import varied_view
from .region_kinematics import build_region_blocks
from .event_selections import highdm_core
from .renderers.background_process_groups import background_process_for_sample
from .statistical_templates import ALIASES

ORDER = ('Top', 'WJet', 'ZJet', 'QCD', 'DYJet', 'PhotonJet', 'VV', 'signal')
LABELS = dict(Top='Top (ttbar + single top)', WJet='W to lv', ZJet='Z to vv',
              QCD='QCD', DYJet='DY', PhotonJet='Photon + jet', VV='VV + VVV',
              signal='T2tt (1200, 500)')
STEPS = ('Retained common skim', 'Event-quality baseline', 'Signal trigger',
         'Tau veto', 'Lepton veto', 'MET > 250 GeV', 'HT > 300 GeV',
         'Jet multiplicity', 'At least one medium b jet', 'Azimuthal selection')


@lru_cache(maxsize=1)
def prefix_columns():
    names = {'dataset_id', 'mStop', 'mLSP', 'is_data', 'is_signal'}
    for source in ('shape_kinematics.py', 'region_kinematics.py', 'ids.py'):
        tree = ast.parse((ROOT / 'utils' / source).read_text())
        names.update(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    for flavor in ('electron', 'muon', 'photon'):
        for field in ('pt', 'eta', 'eta_sc', 'phi', 'mass', 'charge'):
            names.add(flavor + '_' + field + '_all')
    for prefix in ('electron_veto', 'electron_medium', 'muon_loose', 'muon_medium', 'photon_medium'):
        for field in ('pt', 'eta', 'eta_sc', 'phi'):
            names.add(prefix + '_' + field)
    names.update(p + '_corrected_' + k for p in ('jet', 'fatjet') for k in ('pt', 'mass'))
    return names


def group(sample):
    if sample.startswith(('mStop', 'T2tt_', 'T2bW_', 'T2tb_')):
        return 'signal'
    return ALIASES[background_process_for_sample(sample)]


def prefix_masks(a):
    """Use actual canonical kinematics, with topology deliberately last.

    Corrected-central anchoring preserves stored p4 except <10 MeV JER-floor
    objects. Those cannot enter the 15 GeV type-1 MET or 30 GeV jet predicates.
    Hence no recalibration or candidate inference is needed for these cuts.
    varied_view retains the production floating-point MET reconstruction.
    """
    momenta = {p + '_corrected_' + k: a[p + '_corrected_' + k]
               for p in ('jet', 'fatjet') for k in ('pt', 'mass')}
    v = varied_view(a, momenta)
    blocks, _ = build_region_blocks(v)
    b = blocks['SR']
    get = lambda k: np.asarray(v[k], dtype=bool)
    common = [np.ones(len(v), dtype=bool), get('pass_base_common'),
              get('pass_signal_trigger'), get('pass_zero_tau'),
              get('pass_no_veto_leptons'), get('pass_met_250'), get('pass_ht_300')]
    result = {}
    for mode in ('highdm', 'lowdm'):
        high = mode == 'highdm'
        cuts = common + [np.asarray(v['njet']) >= 4 if high else b.njet >= 2,
                         np.asarray(v['nb_medium']) >= 1 if high else b.nb >= 1,
                         get('pass_open_high') if high else b.core]
        # b.core includes the low-dM open_pre and all preceding common cuts.
        masks = np.logical_and.accumulate(np.asarray(cuts), axis=0)
        expected = highdm_core(v, blocks, 'SR') if high else b.core & (b.nb >= 1)
        if not np.array_equal(masks[-1], expected):
            raise ValueError('Cutflow does not reproduce canonical pre-topology selection')
        if high:
            masks = np.vstack([masks, masks[-1] & np.isfinite(b.mtb) & (b.mtb >= 175.)])
        result[mode] = masks
    return result


def sample_masks(a, metadata, signal):
    ids = np.asarray(a['dataset_id'], dtype=int)
    for did in np.unique(ids):
        dataset, process, data, issignal = dataset_label(metadata, int(did))
        if data:
            raise ValueError('Observed data is forbidden in the SR cutflow')
        chosen = ids == did
        if issignal:
            if signal_topology(dataset) != signal['model']:
                continue
            chosen &= ((np.asarray(a['mStop']) == signal['mstop']) &
                       (np.asarray(a['mLSP']) == signal['mlsp']))
            family = 'signal'
        else:
            family = group(process)
        if np.any(chosen):
            yield family, chosen


def fingerprint(p):
    s = p.stat()
    return dict(inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def count_file(task, signal):
    path = internal_path(ROOT / task['input'])
    if fingerprint(path) != task['fingerprint']:
        raise ValueError('Input changed: ' + task['key'])
    metadata = validate_root(path, task['year'])
    totals = {mode: {p: np.zeros(len(STEPS) + int(mode == 'highdm'), dtype=np.int64)
                      for p in ORDER} for mode in ('highdm', 'lowdm')}
    entries = 0
    with uproot.open(path, object_cache=None, array_cache=None,
                     decompression_executor=uproot.TrivialExecutor(),
                     interpretation_executor=uproot.TrivialExecutor()) as f:
        tree = f['Events']
        branches = [name for name in tree.keys() if name in prefix_columns()]
        for a in tree.iterate(branches, step_size=25000, library='ak'):
            entries += len(a)
            groups = list(sample_masks(a, metadata, signal))
            if not groups:
                continue
            keep = np.logical_or.reduce([mask for _, mask in groups])
            a = a[keep]
            masks = prefix_masks(a)
            for family, chosen in sample_masks(a, metadata, signal):
                for mode, rows in masks.items():
                    totals[mode][family] += np.sum(rows[:, chosen], axis=1)
        if entries != tree.num_entries or entries != task['entries']:
            raise ValueError('Incomplete event accounting')
    if fingerprint(path) != task['fingerprint']:
        raise ValueError('Input changed during read')
    return dict(key=task['key'], year=task['year'], status='complete', entries=entries,
                counts={m: {p: v.tolist() for p, v in rows.items()} for m, rows in totals.items()},
                input_fingerprint=task['fingerprint'], branches=branches)


def audit_migrations(path):
    """Decode only the early audit.migrations object in a sorted histogram JSON.

    No full multi-GB histogram materialization or event-level export.
    """
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt') as f:
        buffer = f.read(65536)
        if not buffer.startswith('{"audit":{"events_read":'):
            raise ValueError('Unsupported histogram audit serialization')
        marker = '"migrations":'
        offset = buffer.find(marker)
        if offset < 0 or offset > 128:
            raise ValueError('Expected leading audit.migrations')
        buffer = buffer[offset + len(marker):]
        while True:
            try:
                return decoder.raw_decode(buffer)[0]
            except json.JSONDecodeError:
                more = f.read(1048576)
                if not more or len(buffer) > 128 * 1024**2:
                    raise ValueError('Incomplete/oversized migration audit')
                buffer += more


def card_yields(card, template_hashes=None):
    lines = [s.split() for s in card.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    bins = [r[1:] for r in lines if r[0] == 'bin'][-1]
    procs = next(r[1:] for r in lines if r[0] == 'process')
    rates = list(map(float, next(r[1:] for r in lines if r[0] == 'rate')))
    if len(bins) != len(procs) or len(rates) != len(bins):
        raise ValueError('Malformed card columns')
    parameters = {(r[2], r[3]): float(r[4]) for r in lines if len(r) > 4 and r[1] == 'rateParam'}
    result = {str(y): {m: {p: 0. for p in ORDER} for m in ('highdm', 'lowdm')} for y in (2024, 2025)}
    with contextlib.ExitStack() as stack:
        roots, directories = {}, {}
        for channel, process, rate in zip(bins, procs, rates):
            if not channel.startswith('SR_'):
                continue
            _, mode, _, year = channel.split('_')
            family = process.split('_')[0]
            if rate == -1:
                matches = [r for r in lines if r[0] == 'shapes' and r[1] in ('*', process) and r[2] in ('*', channel)]
                rule = max(matches, key=lambda r: (r[1] == process, r[2] == channel))
                path = internal_path(card.parent / rule[3])
                if path not in roots:
                    if template_hashes is None or sha256(path) != template_hashes.get(path.name):
                        raise ValueError('Changed/unverified card template')
                    roots[path] = stack.enter_context(uproot.open(path, object_cache=None, array_cache=None,
                        decompression_executor=uproot.TrivialExecutor(), interpretation_executor=uproot.TrivialExecutor()))
                key = rule[4].replace('$CHANNEL', channel).replace('$PROCESS', process)
                folder, leaf = key.rsplit('/', 1)
                if (path, folder) not in directories:
                    # A template directory has tens of thousands of keys. Do
                    # not deserialize that index again for every component.
                    directories[path, folder] = roots[path][folder]
                rate = float(np.sum(directories[path, folder][leaf].values(flow=False)))
            if family not in ORDER or rate < 0 or not np.isfinite(rate):
                raise ValueError('Unknown process/invalid card rate')
            result[year][mode][family] += rate * parameters.get((channel, process), 1.)
    return result


def prepare(config_path, output):
    config = read_json(config_path)
    parent = internal_path(ROOT / config['input_plan'])
    plan = read_json(parent)
    if plan['shape_mode'] != 'cms_trota_jme':
        raise ValueError('Current central JME input inventory required')
    grid = read_json(ROOT / config['grid'])
    signal = config['signal']
    mass = 'mStop%d_mLSP%d' % (signal['mstop'], signal['mlsp'])
    point = next(r for r in grid['points'] if r['model'] == signal['model'] and r['mass'] == mass)
    card = internal_path(ROOT / point['card'])
    if sha256(card) != point['card_sha256'] or not grid['sr_data_blinded']:
        raise ValueError('Changed or unblinded card')
    tasks = [t for t in plan['tasks'] if Path(t['input']).name.startswith(('mc_shard_', 'signal_shard_'))]
    if len({t['key'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate inputs')
    final = {}
    hist_sources = {}
    key = signal_mass_key(signal['model'], signal['mstop'], signal['mlsp'])
    for year in ('2024', '2025'):
        path = internal_path(ROOT / config['histograms'][year])
        summary = read_json(path.with_name('systematics_' + year + '.summary.json'))
        digest = sha256(path)
        if (summary['sha256'] != digest or summary['status'] != 'complete'
                or summary['scope'] != 'full_combined_systematic_production'
                or not summary['coverage']['full_input_inventory']):
            raise ValueError('Incomplete/changed current histogram')
        audit = audit_migrations(path)
        final[year] = {m: {p: dict(entries=0, sumw=0.) for p in ORDER} for m in ('highdm', 'lowdm')}
        for sample, regions in audit.items():
            if sample == 'data' or sample.startswith(('mStop', 'T2tt_', 'T2bW_', 'T2tb_')) and sample != key:
                continue
            family = group(sample)
            for mode in final[year]:
                r = regions.get('SR', {}).get('highdm_search' if mode == 'highdm' else 'lowdm', {})
                final[year][mode][family]['entries'] += r.get('after_entries', 0)
                final[year][mode][family]['sumw'] += r.get('after_sumw', 0.)
        hist_sources[year] = dict(path=str(path.relative_to(ROOT)), sha256=digest)
    templates = read_json(ROOT / grid['templates_manifest'])
    record = dict(schema='trotasr_sr_cutflow_v1', config=config, config_sha256=sha256(config_path),
                  input_plan_sha256=sha256(parent), grid_sha256=sha256(ROOT / config['grid']),
                  card=point, card_prefit_yields=card_yields(card, templates['files']), final=final,
                  histograms=hist_sources, tasks=tasks, source_sha256=sha256(Path(__file__)),
                  selection_sources={n: sha256(ROOT / 'utils' / n) for n in
                      ('event_selections.py', 'region_kinematics.py', 'shape_kinematics.py',
                       'ids.py', 'normalization.py', 'signal_models.py', 'statistical_templates.py',
                       'renderers/background_process_groups.py')},
                  interpretation='postskim_raw_MC_counts_not_postfit_yields', observed_data=False,
                  topology_last=True, created=time.time())
    write_json(output / 'contract.json', record)
    return record


def worker(config_path, output, number):
    contract = read_json(output / 'contract.json')
    if contract['config_sha256'] != sha256(config_path) or contract['source_sha256'] != sha256(Path(__file__)):
        raise ValueError('Changed active cutflow implementation/configuration')
    if any(sha256(ROOT / 'utils' / n) != h for n, h in contract['selection_sources'].items()):
        raise ValueError('Changed active cutflow selection/grouping sources')
    for task in contract['tasks']:
        dest = output / 'files' / (task['key'] + '.json')
        if dest.exists():
            continue
        with (output / 'locks' / (task['key'] + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if dest.exists():
                continue
            start = time.time()
            write_json(output / ('worker_%d.json' % number), dict(pid=os.getpid(), key=task['key'], started=start))
            try:
                result = count_file(task, contract['config']['signal'])
            except Exception:
                result = dict(status='failed', key=task['key'], year=task['year'], error=traceback.format_exc())
            result.update(seconds=time.time() - start, finished=time.time())
            write_json(dest, result)
            print(task['key'], result['status'], round(result['seconds'], 2), flush=True)
            if result['status'] == 'failed':
                # A schema/selection failure is not an invitation to skip files.
                raise RuntimeError(result['error'])


def render(result, output):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>SR cutflows</title>',
             '<style>body{font:15px system-ui;margin:32px;color:#17202a}table{border-collapse:collapse;margin:20px 0 36px;width:100%}th,td{border-bottom:1px solid #ddd;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef2f5}small{color:#555}</style>',
             '<h1>High-dM and low-dM SR cutflows</h1>',
             '<p>2024 + 2025; T2tt(1200,500); current combined-systematics model. Observed SR data are excluded.</p>',
             '<p>These are cumulative <b>unweighted simulated-event counts</b> starting from the retained common skim, not from generated events or NanoAOD. The final topology selection reuses the completed corrected-central histogram audit. Efficiencies are relative to the retained skim. Intermediate data-driven background predictions are not invented.</p>']
    for mode in ('highdm', 'lowdm'):
        names = list(STEPS)
        names[7] = 'Njet >= 4' if mode == 'highdm' else 'Njet >= 2'
        names[9] = 'dphi(j1..j4,MET) > 0.5' if mode == 'highdm' else 'dphi(j1,MET)>0.5; dphi(j2,j3,MET)>0.15'
        if mode == 'highdm':
            names.append('mTb >= 175 GeV')
        names.append('At least one B/M/R/W tag' if mode == 'highdm' else 'Veto all B/M/R/W tags')
        parts.append('<h2>' + ('High-dM' if mode == 'highdm' else 'Low-dM') + '</h2><table><thead><tr><th>Cumulative selection</th>')
        parts.extend('<th>' + html.escape(LABELS[p]) + '</th>' for p in ORDER)
        parts.append('</tr></thead><tbody>')
        for i, name in enumerate(names):
            parts.append('<tr><td>' + html.escape(name) + '</td>')
            for p in ORDER:
                values = result['combined'][mode][p]
                eff = 100 * values[i] / values[0] if values[0] else 0.
                parts.append('<td>{:,}<br><small>{:.2f}%</small></td>'.format(values[i], eff))
            parts.append('</tr>')
        parts.append('</tbody></table><h3>Final card prefit expected yields (220.66 fb<sup>-1</sup>)</h3><table><tr>')
        parts.extend('<th>' + html.escape(LABELS[p]) + '</th>' for p in ORDER)
        parts.append('</tr><tr>')
        parts.extend('<td>{:,.4g}</td>'.format(sum(result['card_prefit_yields'][y][mode][p] for y in ('2024', '2025'))) for p in ORDER)
        parts.append('</tr></table>')
    parts.append('<p>The card row includes its measured central factors and initial rate parameters; it is not a postfit estimate. The cutflow ordering places mTb before tagging without changing the final selection. B/M/R/W includes the adopted TopMixed arbitration. No additional low-dM ISR, mTb, significance or GNN-score threshold is imposed.</p></html>')
    (output / 'cutflow_tables.html').write_text('\n'.join(parts))


def collect(contract, output):
    totals = {str(y): {m: {p: np.zeros(len(STEPS) + int(m == 'highdm'), dtype=np.int64)
                          for p in ORDER} for m in ('highdm', 'lowdm')} for y in (2024, 2025)}
    for task in contract['tasks']:
        r = read_json(output / 'files' / (task['key'] + '.json'))
        if r['status'] != 'complete' or r['entries'] != task['entries'] or r['input_fingerprint'] != task['fingerprint']:
            raise ValueError('Missing/failed/incomplete input: ' + task['key'])
        for mode in totals[str(task['year'])]:
            for p in ORDER:
                totals[str(task['year'])][mode][p] += r['counts'][mode][p]
    for year, modes in totals.items():
        for mode, columns in modes.items():
            for p, values in columns.items():
                final = contract['final'][year][mode][p]['entries']
                if int(final) != final or final < 0:
                    raise ValueError('Invalid final event count')
                values = np.append(values, int(final))
                if np.any(np.diff(values) > 0):
                    raise ValueError('Non-nested cumulative cutflow: ' + str((year, mode, p)))
                columns[p] = values.astype(int).tolist()
    result = dict(status='complete', years=totals, contract_sha256=sha256(output / 'contract.json'),
                  combined={m: {p: (np.array(totals['2024'][m][p]) + totals['2025'][m][p]).tolist()
                                for p in ORDER} for m in ('highdm', 'lowdm')},
                  card_prefit_yields=contract['card_prefit_yields'], finished=time.time())
    write_json(output / 'cutflow_tables.json', result)
    render(result, output)
    return result


def run(config_path, output):
    from TROTASR.workflows.diagnostic_common import live_resources
    config_path, output = internal_path(config_path), internal_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('files', 'locks', 'logs'):
            (output / name).mkdir(exist_ok=True)
        contract = read_json(output / 'contract.json') if (output / 'contract.json').exists() else prepare(config_path, output)
        if contract['source_sha256'] != sha256(Path(__file__)) or contract['config_sha256'] != sha256(config_path):
            raise ValueError('Preserve prior output with differing contract')
        workers, exited, failed = {}, [], False
        while True:
            for number, (process, log) in list(workers.items()):
                code = process.poll()
                if code is not None:
                    exited.append(dict(worker=number, exit_code=code)); log.close(); del workers[number]
                    failed |= code != 0
            records = list((output / 'files').glob('*.json'))
            state = dict(status='running', pid=os.getpid(), completed_files=len(records), total_files=len(contract['tasks']),
                         active_workers=len(workers), worker_exits=exited, updated=time.time())
            write_json(output / 'state.json', state)
            if not workers and (failed or len(records) == len(contract['tasks'])):
                if failed:
                    state['status'] = 'needs_attention'
                else:
                    collect(contract, output); state['status'] = 'complete'
                write_json(output / 'state.json', state)
                return
            held = set()
            for number in workers:
                status_path = output / ('worker_%d.json' % number)
                if status_path.exists():
                    key = read_json(status_path)['key']
                    if not (output / 'files' / (key + '.json')).exists():
                        held.add(key)
            if (not failed and len(workers) < contract['config']['max_workers']
                    and len(records) + len(held) < len(contract['tasks'])):
                with (ROOT / 'stats/fit_admission.lock').open('a') as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX)
                    slots, memory = live_resources()
                    if slots < 99 and memory > 36 * 1024**3:
                        number = len(exited) + len(workers)
                        log = (output / 'logs' / ('worker_%d.log' % number)).open('ab')
                        command = [sys.executable, '-u', str(ROOT / 'workflows/build_highdm_histogram.py'),
                                   '--output', str(output), '--cutflow-config', str(config_path), '--cutflow-worker', str(number)]
                        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                   cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                        workers[number] = (process, log)
            time.sleep(3)

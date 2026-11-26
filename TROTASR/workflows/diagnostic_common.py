"""Internal finite command execution for fit diagnostics."""
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.selection_contract import require_current_llcr
from TROTASR.utils.io import read_json, write_json, without_provenance_fields
from TROTASR.workflows.combine_limits import runtime, verify_card
from TROTASR.workflows.resources import process_identity as live_process_identity

SPAWN_LOCK = threading.Lock()
COMPUTE = re.compile(r'^(python|root|combine|text2workspace|xrdcp|rsync|hadd$|cp$)')


def fit_admission_slots():
    """Account-wide cap, including controllers and transfers."""
    return 120


def native_arguments(arguments):
    result = list(arguments)
    for index, value in enumerate(result):
        for flag in ('--manifest', '--output', '--work', '--workspace-state'):
            if value == flag:
                result[index + 1] = str(internal_path(result[index + 1]))
            elif value.startswith(flag + '='):
                result[index] = flag + '=' + str(internal_path(value.split('=', 1)[1]))
    return result


def native_runtime(output):
    if '--native-runtime' not in sys.argv:
        env = runtime(internal_path(output) / 'software_runtime')
        command = ['python3', str(internal_path(sys.argv[0])),
                   *native_arguments(sys.argv[1:]), '--native-runtime']
        # Resolve arguments before changing directory for ROOT's PCH probe.
        os.chdir(ROOT)
        os.execvpe('python3', command, env)
    if sys.platform != 'linux':
        raise RuntimeError('Native ROOT products are hep2-only')
    internal_path(Path.cwd())


def live_resources():
    rows = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'stat=,comm='], text=True)
    slots = sum(not stat.startswith('Z') and bool(COMPUTE.match(comm))
                for stat, comm in (line.split(None, 1) for line in rows.splitlines()))
    available = next(int(line.split()[1]) * 1024
                     for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return slots, available


def admitted_popen(command, **kwargs):
    """Count all compute processes owned by the account, including child jobs.

    The lock serializes admission among these diagnostic controllers. All
    children are single threaded; no secondary process pool is used. Full-grid
    prerequisites prevent running fits alongside unfinished histogram inputs.
    """
    if sys.platform != 'linux':
        raise RuntimeError('Fits use the existing hep2 runtime')
    while True:
        with SPAWN_LOCK, (ROOT / 'stats/fit_admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            slots, memory = live_resources()
            if slots < fit_admission_slots() and memory >= 36 * 1024**3:
                return subprocess.Popen(command, **kwargs)
        time.sleep(3)


def record_years(record):
    from TROTASR.utils.paths import EARLY_ERAS, ERAS
    years = tuple(map(str, record.get('years', ('2024', '2025'))))
    if years not in (('2024', '2025'), EARLY_ERAS, ERAS):
        raise ValueError('Unexpected statistical target-era group')
    if years == ERAS:
        from TROTASR.workflows.build_nominal_grid import require_run3_systematics_policy
        if (record.get('scope', 'full_combined_systematic_production') != 'full_combined_systematic_production'
                or not require_run3_systematics_policy(record)):
            raise ValueError('Six-era record must identify the adopted Run 3 systematic treatment')
    return years


def cr_channels(text, years=('2024', '2025')):
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    channels = next((row[1:] for row in rows if row[0] == 'bin'), [])
    expected = {f'{r}_{m}_c1_{y}' for r in ('LLCR', 'QCDCR', 'GCR')
                for m in ('highdm', 'lowdm') for y in years}
    merged = {f'{r}_{m}_c1' for r in ('LLCR','QCDCR','GCR') for m in ('highdm','lowdm')}
    if set(channels) == merged and len(channels) == 6:
        expected = merged
    if len(channels) != len(expected) or set(channels) != expected:
        raise ValueError('CR-only card must contain six current CR channels per era, no SR')
    if ['*', 'autoMCStats', '10', '1', '1'] not in rows:
        raise ValueError('autoMCStats must remain 10 1 1')
    processes = [row[1:] for row in rows if row[0] == 'process']
    if len(processes) != 2 or 'signal' in processes[0] or any(int(x) <= 0 for x in processes[1]):
        raise ValueError('CR-only card must have backgrounds only')
    if any(row[0] == 'shapes' and row[2] not in expected for row in rows):
        raise ValueError('Unexpected CR shape channel')
    return channels


def cr_only_support_ready(record):
    """A preserved SR-only support failure must not waive any CR failure."""
    support = record.get('fit_compatibility', {})
    issues = support.get('background_issues')
    revised_shapes = False
    if record.get('statistical_revision'):
        require_revised_record(record)
        from TROTASR.workflows.build_nominal_grid import background_shape_names
        audit = support.get('background_error_shapes', {})
        years = record_years(record)
        revised_shapes = (audit.get('nuisances') == 17 * len(years)
            and set(audit.get('nuisance_names', [])) == background_shape_names(years)
            and audit.get('measured_factors_unchanged') is True
            and audit.get('existing_shape_endpoints_unchanged') is True)
    return (record.get('status') == 'templates_verified_fit_blocked'
        and support.get('status') == 'blocked' and support.get('cr_only_ready') is True
        and support.get('auto_mc_stats') == [10, 1, 1]
        and (support.get('interpolation_changed') is False or revised_shapes)
        and support.get('endpoints_modified') is False
        and isinstance(issues, list)
        and all(i.get('channel', '').startswith('SR_') and i.get('problem') in (
            'zero_integral_nonempty_component', 'zero_integral_variation') for i in issues))


def require_revised_record(record):
    """Opt-in adopted model, without relabeling historical nominal grids."""
    from TROTASR.workflows.build_nominal_grid import require_statistical_revision, scientific_settings
    from TROTASR.utils.cr_binning import N_BINS, N_RATES
    revised = require_statistical_revision(record.get('statistical_revision'))
    combined = record.get('scope') == 'full_combined_systematic_production'
    if combined and not revised:
        raise ValueError('Combined diagnostics require the adopted CR12 revision')
    if not revised:
        return False
    years = record_years(record)
    merged = record.get('era_merged') is True
    count = record.get('rate_parameters')
    # Earlier accepted merged manifests left this generic count set to their
    # native model. The explicit floating count is the merged-model quantity.
    source_count = (count == 6*N_RATES and record.get('source_bins') == 6*N_BINS
                    and record.get('source_likelihood_channels') == 48)
    if (record.get('bins') != (N_BINS if merged else N_BINS * len(years))
            or not isinstance(count, int) or count <= 0
            or (merged and (len(years) != 6 or (count != N_RATES and not source_count)
                            or record.get('likelihood_channels') != 8
                            or record.get('floating_normalization_parameters') != N_RATES))
            or (not merged and years == ('2024', '2025') and count != 2*N_RATES)
            or record.get('signal_contamination') is not False
            or record.get('auto_mc_stats') != [10, 1, 1]
            or record.get('sr_data_blinded') is not True
            or scientific_settings(record.get('background_estimation_uncertainties')) != dict(
                mode='shape', auto_mc_stats=[10, 1, 1])):
        raise ValueError('Unverified CR12/background-shape model policy')
    if combined:
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        if (scientific_settings(record.get('systematic_combination')) != scientific_settings(COMBINATION_POLICY)
                or record.get('preliminary') is not True
                or record.get('central_conventions_consistent') is not False
                or not record.get('systematic_scope')):
            raise ValueError('Combined nominal-convention limitation must be explicit')
    return True


def verify_revised_card(card, manifest):
    from TROTASR.workflows.build_nominal_grid import background_shape_names
    years = record_years(manifest)
    meta = read_json(card.with_suffix('.manifest.json'))
    fields = ('statistical_revision', 'background_estimation_uncertainties',
              'systematic_combination', 'signal_contamination', 'rate_parameters')
    if any(meta.get(k) != manifest.get(k) for k in fields) or meta.get('fit_ready') is not True:
        raise ValueError('Card metadata does not match the adopted statistical revision')
    rows = [line.split() for line in card.read_text().splitlines() if line.strip() and not line.startswith('#')]
    parameters = [r for r in rows if len(r) > 1 and r[1] == 'rateParam']
    if (len({r[0] for r in parameters}) != manifest['rate_parameters']
            or any(len(r) != 6 or r[-1] != '[0.01,5]' or not .01 <= float(r[-2]) <= 5 for r in parameters)
            or ['*', 'autoMCStats', '10', '1', '1'] not in rows):
        raise ValueError('Actual card normalization/MC-statistics policy changed')
    names = background_shape_names(years)
    for row in rows:
        if row[0] in names and row[1] != 'shape':
            raise ValueError('Measured background uncertainty is not a shape nuisance')
    process = next(r[1:] for r in rows if r[0] == 'process')
    channel_rows = [r[1:] for r in rows if r[0] == 'bin']
    if len(channel_rows) != 2 or len(channel_rows[1]) != len(process) or any(
            p == 'signal' and not c.startswith('SR_') for c, p in zip(channel_rows[1], process)):
        raise ValueError('Actual card violates the SR-only signal policy')
    expected = {f'{r}_{m}_c1_{y}': (118 if m == 'highdm' else 30) if r == 'SR'
                else 10 for r in ('SR', 'LLCR', 'QCDCR', 'GCR')
                for m in ('highdm', 'lowdm') for y in years}
    if len(meta['inputs']) != 1:
        raise ValueError('Revised card must use one shared four-template model')
    for source in meta['inputs']:
        path = internal_path(ROOT/source['path'])/'model_manifest.json'
        record = read_json(path)
        require_revised_record(record)
        if (record['scope'] != manifest['scope'] or any(record.get(k) != manifest.get(k) for k in fields)
                or {c: len(b) for c, b in record['channels'].items()} != expected
                or set(record['files']) != {'template_'+m+'_'+y+'.root'
                    for m in ('highdm', 'lowdm') for y in years}):
            raise ValueError('Revised template layout/provenance changed')
        for key in ('statistical_revision', 'background_estimation_uncertainties', 'systematic_combination'):
            if record['contract'].get(key) != record.get(key):
                raise ValueError('Revised template contract/model disagreement')
        audit = record['fit_compatibility'].get('background_error_shapes', {})
        if (audit.get('nuisances') != 17 * len(years) or set(audit.get('nuisance_names', [])) != names
                or audit.get('measured_factors_unchanged') is not True
                or audit.get('existing_shape_endpoints_unchanged') is not True):
            raise ValueError('Missing verified measured background-shape audit')


def provisional_cronly_products(output, publication, *,
        required=('payload', 'workspace', 'fit_result', 'projection', 'comparison')):
    """Resolve the saved CR-only result without treating an abort as fit success."""
    if publication.get('status') != 'provisional':
        return None
    if (publication.get('kind') != 'cronly'
            or publication.get('sr_observed_data_used') is not False
            or publication.get('signal_strength', 0) != 0
            or publication.get('fit_b_verified') is not True
            or publication.get('fit_status') != 0
            or publication.get('covariance_quality', -1) < 2):
        raise ValueError('Invalid provisional CR-only receipt')
    output = internal_path(output)
    products = {}
    for name in required:
        if not publication.get(name):
            raise ValueError('Current CR-only product is undeclared: '+name)
        path = internal_path(output/publication[name])
        if path.parent != output or not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing current CR-only product: '+name)
        products[name] = path
    return products


def published_diagnostic(output, manifest_path, kind, model='T2tt', mass='mStop1200_mLSP500'):
    """Reuse the delivered current-model diagnostic, without opening fit ROOTs."""
    path = internal_path(output)/'publication.json'
    if not path.is_file():
        return None
    publication = read_json(path)
    manifest = read_json(internal_path(manifest_path))
    provisional = (provisional_cronly_products(output, publication, required=('payload',))
                   if kind == 'cronly' and publication.get('kind') == kind else None)
    if (publication.get('kind') != kind or publication.get('canonical_model') != 'era_merged'
            or manifest.get('canonical_model') != 'era_merged'
            or manifest.get('scope') != 'full_combined_systematic_production'
            or publication.get('sr_observed_data_used') is not False):
        if provisional is not None:
            raise ValueError('Provisional CR-only result does not match the current base grid')
        return None
    if provisional is None and publication.get('complete') is not True:
        return None
    if kind == 'impact' and publication.get('benchmark') != model+'_'+mass:
        return None
    if not internal_path(output/publication['payload']).is_file():
        raise FileNotFoundError('Delivered diagnostic payload is missing')
    return publication


def grid_card(manifest_path, kind, model='T2tt', mass='mStop1200_mLSP500', *,
              require_grid_ready=True):
    manifest_path = internal_path(manifest_path)
    require_current_llcr(manifest_path)
    manifest = read_json(manifest_path)
    require_current_llcr(record=manifest)
    if manifest.get('era_merged') is True:
        from TROTASR.utils.cr_binning import N_BINS, N_RATES
        if ((require_grid_ready and manifest.get('status') != 'cards_ready')
                or manifest.get('sr_data_blinded') is not True
                or manifest.get('scope') != 'full_combined_systematic_production'
                or manifest.get('bins') != N_BINS or manifest.get('likelihood_channels') != 8
                or manifest.get('floating_normalization_parameters') != N_RATES
                or tuple(map(str,manifest['years'])) != ('2022','2022EE','2023','2023BPix','2024','2025')
                or manifest.get('auto_mc_stats') != [10,1,1]):
            raise ValueError('Accepted canonical six-era merged card receipt required')
        point = (manifest['cr_only_card'] if kind == 'cronly' else
                 next(p for p in manifest['points'] if (p['model'],p['mass']) == (model,mass)))
        if kind not in ('cronly','impact') or (kind == 'impact' and point.get('fit_ready') is not True):
            raise ValueError('One accepted fit-ready diagnostic card is required')
        card = internal_path(ROOT/point['card'])
        if not card.is_file() or not card.stat().st_size:
            raise ValueError('Accepted diagnostic card missing or empty')
        if kind == 'cronly':
            metadata=read_json(card.with_suffix('.manifest.json'))
            if metadata.get('fit_ready') is not True or metadata.get('cr_only') is not True:
                raise ValueError('Accepted CR-only card receipt is required')
            cr_channels(card.read_text(),tuple(manifest['years']))
        return card,dict(manifest=str(manifest_path.relative_to(ROOT)),
            card=str(card.relative_to(ROOT)),
            input_check_scope='accepted_current_merged_card')
    years = record_years(manifest)
    revised = require_revised_record(manifest)
    bins = (252 if revised else 270) * len(years)
    if manifest.get('sr_merge'):
        from TROTASR.utils.sr_merge import load_merge
        merge = load_merge(manifest['sr_merge'])
        bins += len(years) * (merge['bins_per_year'] - 162)
    preserved_cr_only = kind == 'cronly' and cr_only_support_ready(manifest)
    if ((manifest.get('status') != 'cards_ready' and not preserved_cr_only)
            or manifest.get('scope') not in ('full_nominal_production', 'full_cms_trota_jme_production', 'full_combined_systematic_production')
            or manifest.get('bins') != bins or manifest.get('th1_channels') != 8 * len(years)
            or not manifest.get('root_integrity_checked')):
        raise ValueError('Diagnostics require the verified full-input current nominal grid')
    counts = {m: sum(p['model'] == m for p in manifest['points']) for m in ('T2tt', 'T2tb', 'T2bW')}
    expected_counts = ({'T2tt': 225, 'T2tb': 365, 'T2bW': 365}
        if manifest['scope'] == 'full_combined_systematic_production' else {'T2tt': 205, 'T2tb': 365, 'T2bW': 354})
    if manifest.get('requested_grid'):
        from TROTASR.workflows.signal_grid_inputs import requested_grid
        source = manifest['requested_grid']
        path = internal_path(ROOT/source['path'])
        grid = requested_grid(path)
        expected_counts = grid['expected_points_by_model']
        if {(p['model'], p['mass']) for p in manifest['points']} != {
                (m, mass) for m, masses in grid['models'].items() for mass in masses}:
            raise ValueError('Signal grid differs from the frozen requested mass points')
    if (counts != expected_counts or counts != manifest['expected_points_by_model']
            or len({(p['model'], p['mass']) for p in manifest['points']}) != sum(expected_counts.values())):
        raise ValueError('Unexpected full signal-grid coverage')
    if kind == 'impact':
        if model not in ('T2tt', 'T2bW', 'T2tb') or not re.fullmatch(r'mStop\d+_mLSP\d+', mass):
            raise ValueError('Invalid impact benchmark')
        matches = [p for p in manifest['points'] if p['model'] == model and p['mass'] == mass]
        if len(matches) != 1:
            raise ValueError('Missing/duplicate impact benchmark')
        point = matches[0]
        card = verify_card(point)
    elif kind == 'cronly':
        point = manifest['cr_only_card']
        card = internal_path(ROOT / point['card'])
        if not card.is_file() or not card.stat().st_size:
            raise ValueError('CR card missing or empty')
        meta = read_json(card.with_suffix('.manifest.json'))
        if preserved_cr_only and meta.get('fit_ready') is not True:
            raise ValueError('Preserved CR-only card has not passed its own fit-support gate')
        if (not meta['cr_only'] or not meta['sr_data_blinded'] or tuple(meta['years']) != years
                or meta['auto_mc_stats'] != [10, 1, 1]):
            raise ValueError('CR card provenance/blinding mismatch')
        cr_channels(card.read_text(), years)
        for source in meta['inputs']:
            folder = internal_path(ROOT / source['path'])
            record_path = folder / 'model_manifest.json'
            record = read_json(record_path)
            compatible = (record['status'] == 'templates_ready'
                          or preserved_cr_only and cr_only_support_ready(record))
            if (not compatible or not record['root_integrity_checked']
                    or record['scope'] != manifest['scope']):
                raise ValueError('Unverified CR templates')
            template_folder = internal_path(ROOT/record['template_directory']) if record.get('template_directory') else folder
            for name in record['files']:
                template = internal_path(template_folder/name)
                if not template.is_file() or not template.stat().st_size:
                    raise ValueError('CR template missing or empty')
            declared = {internal_path(template_folder/name) for name in record['files']}
            referenced = {internal_path(card.parent/f[3]) for f in
                (line.split() for line in card.read_text().splitlines()) if f and f[0]=='shapes'}
            if referenced != declared:
                raise ValueError('CR shapes must reference exactly the verified four template files')
    else:
        raise ValueError('Unknown diagnostic')
    if revised:
        verify_revised_card(card, manifest)
    for line in card.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'shapes':
            internal_path(card.parent / fields[3])
    return card, dict(manifest=str(manifest_path.relative_to(ROOT)), card=str(card.relative_to(ROOT)))


def contract_options(options):
    """Compare fit settings independently of scheduling and saved source paths."""
    provenance = {'original_contract', 'protected', 'previous_status',
                  'central_fit', 'reused_workspace', 'input_check_scope'}
    return without_provenance_fields({key:value for key,value in options.items()
            if key not in provenance | {'workers','pull_ylim'}})


def freeze_contract(output, origins, files, options):
    """Bind a run to its scientific inputs, not a historical source snapshot."""
    target = internal_path(output) / 'contract.json'
    if target.exists():
        saved = read_json(target)
        if contract_options(saved['options']) != contract_options(options):
            raise ValueError('Divergent diagnostic inputs or fit settings; preserve existing run')
        return saved
    contract = dict(origins=origins, sources=list(files), options=options)
    write_json(target, contract)
    return contract


def process_identity(pid):
    """Linux start ticks distinguish a live child from a recycled PID."""
    if not pid:
        return None
    try:
        fields = Path('/proc/%d/stat' % int(pid)).read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == 'Z' else int(fields[19])


def process_active(state, field='pid'):
    start = state.get(field + '_start_ticks', state.get('start_ticks') if field == 'pid' else None)
    if start is not None:
        return start == process_identity(state.get(field))
    # A legacy child receipt may predate start-tick recording.
    command = state.get('command') if field == 'pid' else None
    if not command or len(command) <= 1:
        return False
    current = live_process_identity(state.get(field))
    if current is None:
        return False
    arguments = current['arguments']
    return len(arguments) >= len(command) and arguments[1-len(command):] == command[1:]


def product_stats(products):
    return {str(p.relative_to(ROOT)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in products if p.is_file()}


def check_products(state):
    """Reuse receipts without rereading large immutable ROOT files."""
    for name in state.get('products', {}):
        path = ROOT / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Completed diagnostic product missing: ' + name)


class CommandFailure(RuntimeError):
    """A consumed command failed; the stage may select its next finite recipe."""


class CommandRunner:
    """Keep the exit status of each fit attempt, including interrupted attempts."""
    def __init__(self, output):
        self.output = internal_path(output)
        self.commands = self.output / 'commands'
        self.commands.mkdir(parents=True, exist_ok=True)
        self._previous = None
        self._previous_lock = threading.Lock()

    def receipt_path(self, log):
        relative = log.relative_to(ROOT)
        return self.commands / 'by-log' / relative.parent / (relative.name + '.json')

    def previous_receipt(self, command, cwd, log):
        # Older attempts recorded the same arguments and log in a flat directory.
        with self._previous_lock:
            if self._previous is None:
                self._previous = {}
                for path in self.commands.glob('*.json'):
                    receipt = read_json(path)
                    if 'log' in receipt and 'command' in receipt:
                        self._previous.setdefault(receipt['log'], []).append((path, receipt))
            for path, receipt in self._previous.get(str(log.relative_to(ROOT)), ()):
                if receipt['command'] == command and receipt.get('cwd', cwd) == cwd:
                    return path
        return None

    def run(self, command, cwd, log, products=(), env=None):
        cwd, log = internal_path(cwd), internal_path(log)
        products = [internal_path(p) for p in products]
        command = list(map(str, command))
        path = self.receipt_path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cwd_record = str(cwd.relative_to(ROOT))
            if not path.exists():
                path = self.previous_receipt(command, cwd_record, log) or path
            if path.exists():
                old = read_json(path)
                if old.get('command', command) != command:
                    raise ValueError('This fit log already belongs to another attempt: ' + str(log))
                if old['status'] == 'running':
                    if process_active(old) or process_active(old, 'controller_pid'):
                        raise BlockingIOError('Diagnostic command is still active: ' + str(path))
                    old.update(status='failed', exit_code=None, pid=None, controller_pid=None,
                               finished=time.time(), interruption='owner and child exited; command consumed')
                    old.setdefault('log', str(log.relative_to(ROOT)))
                    write_json(path, old)
                if old['status'] == 'complete':
                    expected = {str(p.relative_to(ROOT)) for p in products}
                    if old.get('exit_code') != 0 or not expected <= set(old.get('products', {})):
                        raise ValueError('Missing successful receipt for requested command products')
                    check_products({'products': expected})
                return old
            if log.exists() or any(p.exists() for p in products):
                old = dict(status='failed', exit_code=None, command=command, pid=None,
                           log=str(log.relative_to(ROOT)), finished=time.time(),
                           interruption='existing command artifacts have no recorded exit status')
                write_json(path, old)
                return old
            log.parent.mkdir(parents=True, exist_ok=True)
            state = dict(status='running', phase='waiting_for_resources', command=command, cwd=str(cwd.relative_to(ROOT)),
                         log=str(log.relative_to(ROOT)), started=time.time(), pid=None,
                         controller_pid=os.getpid(), controller_pid_start_ticks=process_identity(os.getpid()))
            write_json(path, state)
            environment = dict(os.environ if env is None else env, OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            try:
                with log.open('x') as stream:
                    proc = admitted_popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                          stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    state['pid'] = proc.pid
                    state['pid_start_ticks'] = process_identity(proc.pid)
                    state['phase'] = 'executing'
                    write_json(path, state)
                    rc = proc.wait()
                complete = rc == 0 and all(p.is_file() and p.stat().st_size > 0 for p in products)
                state.update(status='complete' if complete else 'failed', phase='finished', exit_code=rc, pid=None,
                             finished=time.time(), controller_pid=None, product_stats=product_stats(products),
                             products=dict.fromkeys(str(p.relative_to(ROOT)) for p in products if p.is_file()))
                write_json(path, state)
                return state
            except BaseException as error:
                # Keep 'running' if a child may still exist; never duplicate it.
                state['error'] = repr(error)
                write_json(path, state)
                raise


def require_success(state):
    if state['status'] != 'complete':
        raise CommandFailure('Diagnostic command failed; finite attempt preserved: ' + state['log'])


def workspace(runner, card, output, cr_only=False):
    target = internal_path(output) / 'workspace.root'
    command = ['text2workspace.py', str(card), '-m', '120', '-o', str(target)]
    if cr_only:
        command += ['--X-allow-no-signal']
    require_success(runner.run(command, card.parent, output / 'workspace.log', [target]))
    import ROOT as root
    source = root.TFile.Open(str(target))
    try:
        if not source or source.IsZombie() or not source.Get('w') or not source.Get('w').obj('ModelConfig'):
            raise ValueError('Missing validated RooWorkspace/ModelConfig')
    finally:
        if source:
            source.Close()
    return target

"""Internal, finite, content-checked command execution for fit diagnostics."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.combine_limits import IntegrityCache, runtime, verify_card

SPAWN_LOCK = threading.Lock()
COMPUTE = re.compile(r'^(python|root|combine|text2workspace|xrdcp|rsync|hadd$|cp$)')
_RUN3_ADMISSION_PID = None
_LOCAL_ADMISSION = None


@contextmanager
def local_fit_admission(workers, minimum_available=2 * 1024**3):
    """Explicit Mac-only admission for the user-authorized T2tt host split.

    The calling queue verifies the pinned runtime and remote ownership lease.
    This changes only process admission; all numerical commands stay native.
    """
    global _LOCAL_ADMISSION
    if sys.platform != 'darwin' or not 1 <= workers <= 4:
        raise ValueError('Local admission requires macOS and one to four workers')
    if threading.current_thread() is not threading.main_thread() or _LOCAL_ADMISSION is not None:
        raise RuntimeError('Local admission must have one main-thread owner')
    _LOCAL_ADMISSION = (os.getpid(), workers, minimum_available)
    try:
        yield
    finally:
        _LOCAL_ADMISSION = None


def fit_admission_slots():
    """Controller-local opt-in, shared by its threads but not inherited forks."""
    return 120 if _RUN3_ADMISSION_PID == os.getpid() else 100


@contextmanager
def run3_fit_admission(manifest_path, *, plot_range_only):
    """Scope the approved 120-slot aggregate guard to the new 895-point pool.

    The controller must already have validated the input/point contract. Enter
    this on its main thread before starting the pool, and exit after joining it.
    Worker threads only read the process-local setting; all other controllers
    keep the historical 100-slot guard. The caller still enforces its separate
    116-slot downstream budget and the existing failure-first scheduling.
    """
    global _RUN3_ADMISSION_PID
    expected = ROOT / 'stats/run3_systematics_20261025/grid/manifest.json'
    if plot_range_only is not True or internal_path(manifest_path) != expected.resolve():
        raise ValueError('120-slot admission is restricted to the new displayed six-era grid')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Fit admission must be configured by the controller main thread')
    with SPAWN_LOCK:
        if _RUN3_ADMISSION_PID is not None:
            raise RuntimeError('Fit admission context is already active')
        _RUN3_ADMISSION_PID = os.getpid()
    try:
        yield
    finally:
        with SPAWN_LOCK:
            _RUN3_ADMISSION_PID = None


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
        # ROOT restores its initial cwd while probing the PCH. SSH's default
        # cwd may be /home, which the adopted scratch guard correctly rejects.
        # Resolve CLI paths FIRST so relative inputs keep their original meaning.
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
    """Count every live user's compute process, including our own children.

    The lock serializes admission among these diagnostic controllers. All
    children are single threaded; no secondary process pool is used. Full-grid
    prerequisites prevent running fits alongside unfinished histogram inputs.
    """
    if sys.platform == 'darwin':
        import psutil
        if _LOCAL_ADMISSION is None or _LOCAL_ADMISSION[0] != os.getpid():
            raise RuntimeError('Local diagnostic launch requires the verified local queue')
        while True:
            with SPAWN_LOCK:
                children = psutil.Process().children(recursive=True)
                native = 0
                for process in children:
                    try:
                        args = process.cmdline()
                        native += (process.is_running() and any(
                            Path(arg).name in ('combine', 'text2workspace.py') for arg in args[:2]))
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                if (native < _LOCAL_ADMISSION[1]
                        and psutil.virtual_memory().available >= _LOCAL_ADMISSION[2]):
                    return subprocess.Popen(command, **kwargs)
            time.sleep(3)
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
        from TROTASR.workflows.build_nominal_grid import require_hybrid_policy, require_run3_systematics_policy
        if not (require_hybrid_policy(record) or require_run3_systematics_policy(record)):
            raise ValueError('Six-era record must identify its systematic treatment')
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
    from TROTASR.workflows.build_nominal_grid import require_statistical_revision
    revised = require_statistical_revision(record.get('statistical_revision'))
    combined = record.get('scope') == 'full_combined_systematic_production'
    from TROTASR.workflows.build_nominal_grid import HYBRID_SCOPE, require_hybrid_policy
    hybrid = record.get('scope') == HYBRID_SCOPE
    if hybrid and not require_hybrid_policy(record):
        raise ValueError('Missing six-era hybrid policy')
    combined = combined or hybrid
    if combined and not revised:
        raise ValueError('Combined diagnostics require the adopted CR12 revision')
    if not revised:
        return False
    years = record_years(record)
    count = record.get('rate_parameters')
    if (record.get('bins') != 214 * len(years)
            or not isinstance(count, int) or count <= 0
            or years == ('2024', '2025') and count != 96
            or record.get('signal_contamination') is not False
            or record.get('auto_mc_stats') != [10, 1, 1]
            or record.get('sr_data_blinded') is not True
            or record.get('background_estimation_uncertainties') != dict(
                id='jme_background_shapes_20260929', mode='shape', auto_mc_stats=[10, 1, 1], user_approved=True)):
        raise ValueError('Unverified CR12/background-shape model policy')
    if combined:
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        if (record.get('systematic_combination') != COMBINATION_POLICY
                or record.get('preliminary') is not True
                or record.get('central_conventions_consistent') is not False
                or not record.get('systematic_scope')):
            raise ValueError('Combined nominal-convention limitation must be explicit')
    return True


def verify_revised_card(card, manifest, digest):
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
                else (12 if m == 'highdm' else 10) for r in ('SR', 'LLCR', 'QCDCR', 'GCR')
                for m in ('highdm', 'lowdm') for y in years}
    if len(meta['inputs']) != 1:
        raise ValueError('Revised card must use one shared four-template model')
    for source in meta['inputs']:
        path = internal_path(ROOT/source['path'])/'model_manifest.json'
        if digest(path) != source['sha256']:
            raise ValueError('Revised template manifest changed')
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


def published_diagnostic(output, manifest_path, kind, model='T2tt', mass='mStop1200_mLSP500'):
    """Reuse the delivered current-model diagnostic, without opening fit ROOTs."""
    path = internal_path(output)/'publication.json'
    if not path.is_file():
        return None
    publication = read_json(path)
    manifest = read_json(internal_path(manifest_path))
    point = (manifest['cr_only_card'] if kind == 'cronly' else
             next(p for p in manifest['points'] if (p['model'],p['mass']) == (model,mass)))
    if (publication.get('kind') != kind or publication.get('canonical_model') != 'era_merged'
            or manifest.get('canonical_model') != 'era_merged'
            or publication.get('current_card_sha256') != point['card_sha256']
            or publication.get('sr_observed_data_used') is not False
            or publication.get('complete') is not True):
        return None
    if kind == 'impact' and publication.get('benchmark') != model+'_'+mass:
        return None
    if not internal_path(output/publication['payload']).is_file():
        raise FileNotFoundError('Delivered diagnostic payload is missing')
    return publication


def grid_card(manifest_path, kind, model='T2tt', mass='mStop1200_mLSP500'):
    manifest_path = internal_path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get('era_merged') is True:
        if (manifest.get('status') != 'cards_ready' or manifest.get('sr_data_blinded') is not True
                or manifest.get('bins') != 214 or manifest.get('likelihood_channels') != 8
                or manifest.get('floating_normalization_parameters') != 48
                or tuple(map(str,manifest['years'])) != ('2022','2022EE','2023','2023BPix','2024','2025')
                or manifest.get('auto_mc_stats') != [10,1,1]):
            raise ValueError('Accepted canonical six-era merged card receipt required')
        point = (manifest['cr_only_card'] if kind == 'cronly' else
                 next(p for p in manifest['points'] if (p['model'],p['mass']) == (model,mass)))
        if kind not in ('cronly','impact') or (kind == 'impact' and point.get('fit_ready') is not True):
            raise ValueError('One accepted fit-ready diagnostic card is required')
        card = internal_path(ROOT/point['card'])
        digest = sha256(card)
        if digest != point['card_sha256']:
            raise ValueError('Accepted small diagnostic card changed')
        if kind == 'cronly':
            metadata=read_json(card.with_suffix('.manifest.json'))
            if metadata.get('fit_ready') is not True or metadata.get('cr_only') is not True:
                raise ValueError('Accepted CR-only card receipt is required')
            cr_channels(card.read_text(),tuple(manifest['years']))
        return card,dict(manifest=str(manifest_path.relative_to(ROOT)),
            manifest_sha256=sha256(manifest_path),card=str(card.relative_to(ROOT)),
            card_sha256=digest,input_check_scope='accepted_current_merged_card; no_upstream_or_ROOT_rehash')
    years = record_years(manifest)
    revised = require_revised_record(manifest)
    bins = (258 if revised else 270) * len(years)
    if manifest.get('sr_merge'):
        from TROTASR.utils.sr_merge import load_merge
        merge = load_merge(manifest['sr_merge'])
        bins += len(years) * (merge['bins_per_year'] - 162)
    preserved_cr_only = kind == 'cronly' and cr_only_support_ready(manifest)
    if ((manifest.get('status') != 'cards_ready' and not preserved_cr_only)
            or manifest.get('scope') not in ('full_nominal_production', 'full_cms_trota_jme_production', 'full_combined_systematic_production', 'full_run3_hybrid_production')
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
        if sha256(path) != source['sha256']:
            raise ValueError('Requested signal grid changed')
        grid = requested_grid(path)
        expected_counts = grid['expected_points_by_model']
        if {(p['model'], p['mass']) for p in manifest['points']} != {
                (m, mass) for m, masses in grid['models'].items() for mass in masses}:
            raise ValueError('Signal grid differs from the frozen requested mass points')
    if (counts != expected_counts or counts != manifest['expected_points_by_model']
            or len({(p['model'], p['mass']) for p in manifest['points']}) != sum(expected_counts.values())):
        raise ValueError('Unexpected full signal-grid coverage')
    digest = IntegrityCache()
    if kind == 'impact':
        if model not in ('T2tt', 'T2bW', 'T2tb') or not re.fullmatch(r'mStop\d+_mLSP\d+', mass):
            raise ValueError('Invalid impact benchmark')
        matches = [p for p in manifest['points'] if p['model'] == model and p['mass'] == mass]
        if len(matches) != 1:
            raise ValueError('Missing/duplicate impact benchmark')
        point = matches[0]
        card = verify_card(point, digest)
    elif kind == 'cronly':
        point = manifest['cr_only_card']
        card = internal_path(ROOT / point['card'])
        if digest(card) != point['card_sha256']:
            raise ValueError('CR-only card changed')
        meta = read_json(card.with_suffix('.manifest.json'))
        if preserved_cr_only and meta.get('fit_ready') is not True:
            raise ValueError('Preserved CR-only card has not passed its own fit-support gate')
        if (not meta['cr_only'] or not meta['sr_data_blinded'] or tuple(meta['years']) != years
                or meta['auto_mc_stats'] != [10, 1, 1] or meta['card_sha256'] != point['card_sha256']):
            raise ValueError('CR card provenance/blinding mismatch')
        cr_channels(card.read_text(), years)
        for source in meta['inputs']:
            folder = internal_path(ROOT / source['path'])
            record_path = folder / 'model_manifest.json'
            if digest(record_path) != source['sha256']:
                raise ValueError('CR template provenance changed')
            record = read_json(record_path)
            compatible = (record['status'] == 'templates_ready'
                          or preserved_cr_only and cr_only_support_ready(record))
            if (not compatible or not record['root_integrity_checked']
                    or record['scope'] != manifest['scope']):
                raise ValueError('Unverified CR templates')
            template_folder = internal_path(ROOT/record['template_directory']) if record.get('template_directory') else folder
            for name, expected in record['files'].items():
                if digest(template_folder / name) != expected:
                    raise ValueError('CR template changed')
            declared = {internal_path(template_folder/name) for name in record['files']}
            referenced = {internal_path(card.parent/f[3]) for f in
                (line.split() for line in card.read_text().splitlines()) if f and f[0]=='shapes'}
            if referenced != declared:
                raise ValueError('CR shapes must reference exactly the verified four template files')
    else:
        raise ValueError('Unknown diagnostic')
    if revised:
        verify_revised_card(card, manifest, digest)
    for line in card.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'shapes':
            internal_path(card.parent / fields[3])
    return card, dict(manifest=str(manifest_path.relative_to(ROOT)), manifest_sha256=digest(manifest_path),
                      card=str(card.relative_to(ROOT)), card_sha256=digest(card))


def freeze_contract(output, origins, files, options):
    contract = dict(origins=origins, code={p: sha256(ROOT / p) for p in files}, options=options)
    target = internal_path(output) / 'contract.json'
    if target.exists() and read_json(target) != contract:
        raise ValueError('Divergent diagnostic contract; preserve existing run')
    write_json(target, contract)
    return contract


def command_key(command, cwd):
    return hashlib.sha256(json.dumps([list(map(str, command)), str(cwd)], separators=(',', ':')).encode()).hexdigest()[:24]


class CommandRunner:
    """One attempt per exact command; exit failures and interrupted attempts persist.

    Numerical retries must have distinct, finite configurations supplied by the
    caller. A heartbeat cannot reset retry counts or overwrite interrupted ROOT.
    """
    def __init__(self, output):
        self.output = internal_path(output)
        (self.output / 'commands').mkdir(parents=True, exist_ok=True)

    def run(self, command, cwd, log, products=(), env=None):
        cwd, log = internal_path(cwd), internal_path(log)
        products = [internal_path(p) for p in products]
        command = list(map(str, command))
        path = self.output / 'commands' / (command_key(command, cwd) + '.json')
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.exists():
                old = read_json(path)
                if old['status'] == 'running':
                    raise RuntimeError('Interrupted/running command retained; explicit recovery required: ' + str(path))
                if old['status'] == 'complete':
                    if set(old['products']) != {str(p.relative_to(ROOT)) for p in products}:
                        raise ValueError('Changed expected command products')
                    for name, digest in old['products'].items():
                        if sha256(ROOT / name) != digest:
                            raise ValueError('Completed diagnostic product changed: ' + name)
                return old
            if log.exists() or any(p.exists() for p in products):
                raise FileExistsError('Unregistered diagnostic artifacts retained: ' + str(log))
            log.parent.mkdir(parents=True, exist_ok=True)
            state = dict(status='running', command=command, cwd=str(cwd.relative_to(ROOT)),
                         log=str(log.relative_to(ROOT)), started=time.time(), pid=None)
            write_json(path, state)
            environment = dict(os.environ if env is None else env, OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            try:
                with log.open('x') as stream:
                    proc = admitted_popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                          stdout=stream, stderr=subprocess.STDOUT)
                    state['pid'] = proc.pid
                    write_json(path, state)
                    rc = proc.wait()
                complete = rc == 0 and all(p.is_file() and p.stat().st_size > 0 for p in products)
                state.update(status='complete' if complete else 'failed', exit_code=rc, pid=None,
                             finished=time.time(), products={str(p.relative_to(ROOT)): sha256(p)
                             for p in products if p.is_file()})
                write_json(path, state)
                return state
            except BaseException as error:
                # Keep 'running' if a child may still exist; never duplicate it.
                state['error'] = repr(error)
                write_json(path, state)
                raise


def require_success(state):
    if state['status'] != 'complete':
        raise RuntimeError('Diagnostic command failed; finite attempt preserved: ' + state['log'])


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

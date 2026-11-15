"""Read source NanoAOD LHE weights and accumulate them before reduction.

The retained nominal event selection supplies file_id/entry and corrected
weights. NanoAOD supplies only generator variations and event identities.
No object, region, GNN boundary, normalization, or intermediate ROOT changes.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
from statistics import NormalDist

import awkward as ak
import numpy as np
import uproot
from .io import read_json
from .paths import ROOT, ERAS, analysis_era
from .signal_models import SIGNAL_TOPOLOGY_IDS, signal_mass_key, signal_topology


SOURCE_AUDIT_SCHEMA = 'trotasr_lhe_source_coverage_v1'
MEMBER_MERGE_SCHEMA = 'trotasr_lhe_member_sums_v1'
LHE_BRANCHES = {'LHEPdfWeight': 'pdf', 'LHEScaleWeight': 'scale'}
MC_SHARD = re.compile(r'mc_shard_\d+\.root')
SIGNAL_SHARD = re.compile(r'signal_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.root', re.I)
SIGNAL_FILES_PER_ERA = 187
SIGNAL_POPULATION = 'regenerated_signal_uuid'
MASS_SUFFIX = re.compile(r'(T2tt|T2tb|T2bW)_(\d+)_(\d+)')
MEMBER_RUNS_BRANCHES = {'pdf': 'LHEPdfSumw', 'scale': 'LHEScaleSumw'}
SCALE_COORDINATES = tuple((r, f) for r in (.5, 1., 2.) for f in (.5, 1., 2.)
                          if max(r / f, f / r) <= 2.)


@dataclass(frozen=True)
class Prescription:
    kind: str
    label: str
    members: tuple
    error_type: str = ""
    confidence_level: float = 68.2689492137
    central_index: int = -1
    envelope_indices: tuple = ()
    total_members: int = 0

    def record(self):
        return asdict(self)


@lru_cache(maxsize=1)
def _signal_inventory():
    base = ROOT / 'production'
    manifest, collection = (read_json(base / name) for name in ('tasks.json', 'collection.json'))
    total = SIGNAL_FILES_PER_ERA * len(ERAS)
    records = manifest['tasks']
    expected = {era: set() for era in ERAS}
    result = {}
    if (tuple(manifest['eras']) != ERAS or manifest['files'] != SIGNAL_FILES_PER_ERA
            or len(records) != total or collection.get('total') != total
            or collection.get('completed') != total or collection.get('active') != 0
            or collection.get('errors') != {}):
        raise ValueError('LHE signal weights require the complete new UUID collection')
    for task in records:
        era, stem, source = (task[k] for k in ('era', 'name', 'source'))
        name, lfn = stem + '.root', source['name']
        if (era not in expected or not SIGNAL_SHARD.fullmatch(name) or stem != 'signal_' + Path(lfn).stem
                or lfn in expected[era] or (era, name) in result):
            raise ValueError('Divergent UUID signal source inventory')
        expected[era].add(lfn)
        result[(era, name)] = source
    if any(len(lfns) != SIGNAL_FILES_PER_ERA or lfns != expected[ERAS[0]] for lfns in expected.values()):
        raise ValueError('The original UUID signal file set differs across target eras')
    return result


def lhe_input_population(path, year):
    """Reject observed and retired input names before ROOT I/O."""
    era = analysis_era(year)
    path = Path(path)
    if path != ROOT / 'inputs' / era / path.name:
        raise ValueError('LHE weights require a canonical intermediate MC ROOT')
    if MC_SHARD.fullmatch(path.name):
        return None
    if not SIGNAL_SHARD.fullmatch(path.name):
        raise ValueError('Only background MC or the new UUID signal collection is allowed for LHE weights')
    source = _signal_inventory().get((era, path.name))
    if not source:
        raise ValueError('Signal ROOT was not registered in the new UUID inventory')
    receipt = read_json(ROOT / 'production' / 'collected' / era / path.with_suffix('.json').name)
    if (receipt.get('status') != 'collected' or receipt.get('era') != era
            or receipt.get('events') != source['nevents']
            or receipt.get('bytes') != path.stat().st_size):
        raise ValueError('Registered UUID signal ROOT disagrees with its collection receipt')
    return source


def signal_lhe_provenance(source, metadata, year):
    era = analysis_era(year)
    if (metadata.get('status') != 'complete' or metadata.get('input_nanoaod_version') != 15
            or metadata.get('analysis_era') != era or metadata.get('events_read') != source['nevents']
            or len(metadata['files']) != 1 or len(metadata['datasets']) != 1):
        raise ValueError('Expected complete signal-only UUID production metadata')
    dataset = next(iter(metadata['datasets'].values()))
    item = metadata['files'][0]
    inspected = source['nano_metadata']
    nominal = {'GenModel_' + k: float(v) for k, v in inspected['mass_sumw'].items()}
    _signal_points(nominal)
    topology = signal_topology(source['dataset'])
    if (not topology or any(not k.startswith('GenModel_' + topology + '_') for k in nominal)
            or set(nominal) != set(inspected['genmodel_branches']) or inspected['events'] != source['nevents']
            or dataset.get('is_signal') is not True or dataset.get('is_data') is not False
            or dataset.get('dataset') != source['dataset'] or dataset.get('signal_sumw_by_genmodel') != nominal
            or item.get('events_read') != source['nevents']
            or item.get('runs_sumw', {}).get('signal_sumw_by_genmodel') != nominal):
        raise ValueError('UUID signal metadata differs from the registered source mass points')
    return dict(population=SIGNAL_POPULATION, source=source['name'], file_id=int(item['file_id']),
                dataset=source['dataset'], topology=topology, entries=source['nevents'],
                bytes=source['size'], nominal_by_genmodel=nominal)


def theory_process(dataset, process):
    """Keep hard processes separate inside the displayed Top and VV stacks."""
    primary = dataset.strip('/').split('/')[0].split('-Run')[0]
    if process not in {'TT', 'ST', 'VV'}:
        return process
    if primary.startswith(('TTto', 'TTTo', 'TTJets')):
        return 'ttbar'
    if process == 'ST':
        if 't-channel' in primary:
            return 'singleTop_tchannel'
        if 's-channel' in primary:
            return 'singleTop_schannel'
        if primary.startswith(('TWminus', 'TbarWplus')):
            return 'singleTop_tW'
    else:
        for name in ('TTTT', 'TTWW', 'TTWZ', 'TTZZ', 'TTW', 'TTZ',
                     'WWW', 'WWZ', 'WZZ', 'ZZZ', 'WW', 'WZ', 'ZZ'):
            if primary.startswith(name):
                return name
    raise ValueError('Unmapped background hard process: ' + primary)


@lru_cache(maxsize=1)
def _pdf_catalog():
    candidates = [Path(p) for p in os.environ.get('LHAPDF_DATA_PATH', '').split(':') if p]
    candidates.append(Path('/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current'))
    for directory in candidates:
        index = directory / 'pdfsets.index'
        if index.is_file():
            rows = [line.split() for line in index.read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
            return directory, sorted((int(row[0]), row[1]) for row in rows)
    raise FileNotFoundError('LHAPDF set metadata (pdfsets.index) is unavailable')


@lru_cache(maxsize=64)
def pdf_prescription(title):
    match = re.search(r'LHA IDs\s+(\d+)\s*-\s*(\d+)', title)
    if not match:
        raise ValueError('PDF member IDs are not specified by the NanoAOD branch')
    first, last = map(int, match.groups())
    directory, catalog = _pdf_catalog()
    index = bisect_right([i for i, _ in catalog], first) - 1
    if index < 0:
        raise ValueError('Unknown LHA ID: ' + str(first))
    base, name = catalog[index]
    info = {}
    for line in (directory / name / (name + '.info')).read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator:
            info[key.strip()] = value.strip().strip('"\'')
    total = int(info['NumMembers'])
    if last >= base + total:
        raise ValueError('PDF branch spans more than one PDF set')
    members = tuple(range(first - base, last - base + 1))
    return Prescription('pdf', name, members, info['ErrorType'].lower(),
                        float(info.get('ErrorConfLevel', 68.2689492137)),
                        members.index(0) if 0 in members else -1,
                        total_members=total)


@lru_cache(maxsize=32)
def scale_prescription(title):
    entries = re.findall(r'\[(\d+)\]([^;\[]+)', title)
    points = {}
    for index, description in entries:
        r = re.search(r'mur\s*=\s*["\']?([\d.]+)', description, re.I)
        f = re.search(r'muf\s*=\s*["\']?([\d.]+)', description, re.I)
        if r and f:
            points[int(index)] = (float(r.group(1)), float(f.group(1)))
    if set(points) != set(range(len(points))) or len(points) not in (8, 9):
        raise ValueError('Scale member coordinates are missing or nonstandard')
    allowed = {(r, f) for r in (.5, 1., 2.) for f in (.5, 1., 2.)
               if max(r / f, f / r) <= 2.}
    if set(points.values()) not in (allowed | {(.5, 2.), (2., .5)},
                                   (allowed - {(1., 1.)}) | {(.5, 2.), (2., .5)}):
        raise ValueError('Scale weights do not describe the standard 7-point scan')
    members = tuple(points[i] for i in range(len(points)))
    return Prescription('scale', 'muR_muF_7point', members,
                        central_index=members.index((1., 1.)) if (1., 1.) in members else -1,
                        envelope_indices=tuple(i for i, point in enumerate(members) if point in allowed))


class SourceWeights:
    """Bounded source reads; join by original entry and check event identity."""
    def __init__(self, metadata, block_size=10000, max_open=2, signal_provenance=None):
        self.signal = signal_provenance
        if self.signal is not None:
            self.sources = {self.signal['file_id']: self.signal['source']}
        else:
            self.sources = {}
            for record in metadata['files']:
                key, path = int(record['file_id']), record['file_path']
                if key in self.sources and self.sources[key] != path:
                    raise ValueError('Ambiguous source file_id')
                self.sources[key] = path
        self.block_size, self.max_open = block_size, max_open
        self.open_files = OrderedDict()
        self.audit = {'schema': SOURCE_AUDIT_SCHEMA, 'files': {}, 'joined_events': 0, 'read_entries': 0}

    def close(self):
        for file in self.open_files.values():
            file.close()
        self.open_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _open(self, file_id):
        if file_id in self.open_files:
            self.open_files.move_to_end(file_id)
            return self.open_files[file_id]['Events']
        path = self.sources[file_id]
        urls = [path]
        if '/store/' in path:
            lfn = '/store/' + path.split('/store/', 1)[1]
            urls += ['root://cms-xrd-global.cern.ch/' + lfn,
                     'root://xrootd-cms.infn.it/' + lfn]
        errors = []
        for url in dict.fromkeys(urls):
            try:
                file = uproot.open(url, timeout=45, object_cache=None, array_cache=None)
                tree = file['Events']
                break
            except (OSError, TimeoutError) as error:
                errors.append(str(error))
        else:
            raise OSError('Source NanoAOD unavailable: ' + path + '\n' + '\n'.join(errors))
        self.open_files[file_id] = file
        while len(self.open_files) > self.max_open:
            self.open_files.popitem(last=False)[1].close()
        missing = [branch for branch in LHE_BRANCHES if branch not in tree]
        info = self.audit['files'].setdefault(str(file_id), {
            'source': path, 'endpoint': url, 'missing': missing, 'joined_events': 0, 'branches': {}})
        if info['missing'] != missing:
            raise ValueError('LHE branch presence changed when reopening original NanoAOD')
        if self.signal is not None and (tree.num_entries != self.signal['entries']
                or file.file.source.num_bytes != self.signal['bytes']):
            raise ValueError('Original signal source event or byte count differs from the registered input')
        return tree

    def _definitions(self, tree, file_id):
        specs = {}
        constructors = {'pdf': pdf_prescription, 'scale': scale_prescription}
        for branch, kind in LHE_BRANCHES.items():
            if branch not in tree:
                continue
            title = tree[branch].title
            try:
                specs[branch] = constructors[kind](title)
            except (ValueError, OSError) as exc:
                if self.signal is None:
                    raise
                record = dict(title=title, reason=str(exc))
                unresolved = self.audit['files'][str(file_id)].setdefault('unidentified_branches', {})
                if unresolved.setdefault(branch, record) != record:
                    raise ValueError('Signal LHE branch title changed when reopening original NanoAOD')
        return specs

    def signal_denominators(self):
        """Use the same approved original file and raw basis as the selected join."""
        if self.signal is None:
            raise ValueError('No registered signal source to normalize')
        file_id = self.signal['file_id']
        tree = self._open(file_id)
        specs = {spec.kind: spec for spec in self._definitions(tree, file_id).values()}
        nominal = self.signal['nominal_by_genmodel']
        original = self.open_files[self.signal['file_id']]
        if 'Runs' not in original:
            raise ValueError('Registered signal source has no Runs tree')
        result = signal_runs_member_denominators(original['Runs'], specs, nominal)
        # Full Events can supply missing per-mass sums, but cannot identify
        # a member ordering or PDF set left undocumented by the source.
        if any(row['kind'] in specs and row['reason'] != 'incomplete_event_member_definition'
               for row in result['uncovered']):
            result = signal_event_member_denominators(tree, nominal, entries=self.signal['entries'],
                                                      step_size=self.block_size, prescriptions=specs)
        unresolved = self.audit['files'][str(file_id)].get('unidentified_branches', {})
        return {**result, 'unidentified_branches': unresolved,
                'source_fingerprint': {k: self.signal[k] for k in ('source', 'entries', 'bytes')}}

    def groups(self, arrays):
        """Yield (row indices, prescription, relative weights) per source file.

        Missing generator branches are recorded as missing coverage, never
        fabricated as unity weights. The caller decides whether a final
        aggregate has the required sample coverage.
        """
        file_ids = np.asarray(arrays['file_id'], dtype=np.int64)
        for file_id in np.unique(file_ids):
            selected = np.flatnonzero(file_ids == file_id)
            rows = arrays[selected]
            entries = np.asarray(rows['entry'], dtype=np.int64)
            tree = self._open(int(file_id))
            if np.any(entries < 0) or np.any(entries >= tree.num_entries):
                raise ValueError('Source entry outside its NanoAOD file')
            specs = self._definitions(tree, file_id)
            if self.signal is not None and not specs:
                info = self.audit['files'][str(file_id)]
                info['unsupported_selected_events'] = info.get('unsupported_selected_events', 0) + len(entries)
                continue
            values = {key: np.empty((len(entries), len(spec.members)), dtype=np.float64)
                      for key, spec in specs.items()}
            model_fields = ()
            if self.signal is not None:
                expected_topo = SIGNAL_TOPOLOGY_IDS[self.signal['topology']]
                if np.any(np.asarray(rows['signal_topology_id']) != expected_topo):
                    raise ValueError('Signal topology in retained events differs from registered source')
                model_fields = tuple(self.signal['nominal_by_genmodel'])
                expected_model = np.asarray([f"GenModel_{self.signal['topology']}_{ms}_{ml}"
                    for ms, ml in zip(np.asarray(rows['mStop'], int), np.asarray(rows['mLSP'], int))])
                if not set(expected_model) <= set(model_fields):
                    raise ValueError('Unknown GenModel among selected signal rows')
            for block in np.unique(entries // self.block_size):
                take = np.flatnonzero(entries // self.block_size == block)
                start = int(block * self.block_size)
                stop = min(start + self.block_size, tree.num_entries)
                raw = tree.arrays(['run', 'luminosityBlock', 'event', *specs, *model_fields],
                                  entry_start=start, entry_stop=stop)
                matched = raw[entries[take] - start]
                for field in ('run', 'luminosityBlock', 'event'):
                    if not np.array_equal(np.asarray(matched[field]), np.asarray(rows[field])[take]):
                        raise ValueError('NanoAOD/nominal event identity mismatch: ' + field)
                for field in model_fields:
                    if not np.array_equal(np.asarray(matched[field], bool), expected_model[take] == field):
                        raise ValueError('NanoAOD/nominal GenModel identity mismatch: ' + field)
                for key, spec in specs.items():
                    array = ak.to_numpy(matched[key])
                    if array.shape != (len(take), len(spec.members)) or not np.isfinite(array).all():
                        raise ValueError('Malformed LHE member array: ' + key)
                    values[key][take] = array
                self.audit['read_entries'] += stop - start
            self.audit['joined_events'] += len(entries)
            info = self.audit['files'][str(file_id)]
            info['joined_events'] += len(entries)
            for key, spec in specs.items():
                ratios = values[key]
                source_spec = spec
                if spec.kind == 'scale':
                    # Align eight- and nine-member NanoAOD conventions by
                    # their documented coordinates before adding samples.
                    points = SCALE_COORDINATES
                    ratios = np.column_stack([ratios[:, spec.members.index(point)]
                        if point in spec.members else np.ones(len(entries)) for point in points])
                    spec = Prescription('scale', spec.label, points,
                        central_index=points.index((1., 1.)), envelope_indices=tuple(range(7)))
                definition = dict(source=source_spec.record(), accumulated=spec.record())
                branch = info['branches'].setdefault(key, dict(definition, events=0))
                if any(branch[k] != val for k, val in definition.items()):
                    raise ValueError('LHE member definition changed when reopening original NanoAOD')
                branch['events'] += len(entries)
                yield selected, spec, ratios


def member_histogram(coordinates, weights, ratios, edges):
    """One nominal bin assignment for every LHE member, including signed MC."""
    weights, ratios = np.asarray(weights, float), np.asarray(ratios, float)
    coords = np.asarray(coordinates, float)
    if coords.ndim == 1:
        coords = coords[:, None]
    axes = [np.asarray(axis, float) for axis in edges]
    shape = tuple(len(axis) - 1 for axis in axes)
    if (coords.shape != (len(weights), len(axes)) or ratios.ndim != 2
            or ratios.shape[0] != len(weights)):
        raise ValueError('LHE histogram row alignment differs')
    if not np.isfinite(weights).all() or not np.isfinite(ratios).all():
        raise ValueError('Nonfinite LHE histogram weight')
    indices = np.column_stack([np.searchsorted(axis, coords[:, d], side='right') - 1
                               for d, axis in enumerate(axes)])
    for d, axis in enumerate(axes):
        indices[coords[:, d] == axis[-1], d] = len(axis) - 2
    keep = np.isfinite(coords).all(axis=1)
    keep &= np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    flat = np.ravel_multi_index(indices[keep].T, shape)
    nbin = math.prod(shape)
    def summed(w):
        return np.bincount(flat, weights=w, minlength=nbin).reshape(shape)
    varied = weights[keep, None] * ratios[keep]
    return {'edges': [axis.tolist() for axis in axes],
            'nominal': summed(weights[keep]).tolist(),
            'sumw2': summed(weights[keep] ** 2).tolist(),
            'entries': summed(np.ones(np.count_nonzero(keep))).tolist(),
            'members': np.asarray([summed(w) for w in varied.T]).tolist(),
            'members_sumw2': np.asarray([summed(w ** 2) for w in varied.T]).tolist()}


def merge_members(target, source):
    if target['edges'] != source['edges']:
        raise ValueError('LHE histogram axes differ')
    for field in ('nominal', 'sumw2', 'entries', 'members', 'members_sumw2'):
        a, b = np.asarray(target[field]), np.asarray(source[field])
        if a.shape != b.shape:
            raise ValueError('LHE member dimensions differ: ' + field)
        target[field] = (a + b).tolist()


def accumulate(target, prescription, process, path, coordinates, weights, ratios, edges):
    """Keep PDF sets and hard processes distinct until the era-wide merge."""
    key = prescription.kind + ':' + prescription.label
    record = target.setdefault(key, {'prescription': prescription.record(), 'processes': {}})
    if record['prescription'] != prescription.record():
        raise ValueError('Inconsistent member definitions within one PDF/scale set')
    node = record['processes'].setdefault(process, {})
    for component in path[:-1]:
        node = node.setdefault(component, {})
    value = member_histogram(coordinates, weights, ratios, edges)
    if path[-1] in node:
        merge_members(node[path[-1]], value)
    else:
        node[path[-1]] = value


def reduce_members(histogram, prescription):
    """Reduce only a fully accumulated member histogram, never per event/file."""
    nominal = np.asarray(histogram['nominal'], float)
    values = np.asarray(histogram['members'], float)
    if prescription.kind == 'scale':
        scan = np.concatenate([nominal[None], values[list(prescription.envelope_indices)]])
        return np.max(scan, axis=0), np.min(scan, axis=0)
    members = list(prescription.members)
    central = values[members.index(0)] if 0 in members else nominal
    error_type = prescription.error_type
    extra = 2 if error_type.endswith('+as') else 0
    total = prescription.total_members or len(members)
    core_ids = set(range(1, total - extra))
    if not core_ids <= set(members):
        raise ValueError('Incomplete PDF eigenvector/replica coverage')
    core = values[[members.index(i) for i in sorted(core_ids)]]
    if error_type.split('+')[0] == 'symmhessian':
        error = np.sqrt(np.sum((core - central) ** 2, axis=0))
        plus = minus = error
    elif error_type.split('+')[0] == 'hessian':
        if len(core) % 2:
            raise ValueError('Hessian members must form plus/minus pairs')
        pairs = (core - central).reshape((-1, 2, *nominal.shape))
        plus = np.sqrt(np.sum(np.maximum(pairs.max(axis=1), 0.) ** 2, axis=0))
        minus = np.sqrt(np.sum(np.maximum(-pairs.min(axis=1), 0.) ** 2, axis=0))
    elif error_type.split('+')[0] == 'replicas':
        if len(core) < 2:
            raise ValueError('Replica uncertainty requires at least two members')
        plus = minus = np.std(core, axis=0, ddof=1)
    else:
        raise ValueError('Unsupported PDF error prescription: ' + error_type)
    scale = NormalDist().inv_cdf((1. + prescription.confidence_level / 100.) / 2.)
    plus, minus = plus / scale, minus / scale
    if extra and {total - 2, total - 1} <= set(members):
        alpha = abs(values[members.index(total - 1)] - values[members.index(total - 2)]) / 2.
        plus, minus = np.hypot(plus, alpha), np.hypot(minus, alpha)
    return nominal + plus, nominal - minus


def _leaves(node, path=()):
    if 'members' in node:
        yield path, node
    else:
        for name, value in node.items():
            yield from _leaves(value, (*path, name))


def _record_prescription(record):
    values = dict(record)
    values['members'] = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in values['members'])
    values['envelope_indices'] = tuple(values['envelope_indices'])
    result = Prescription(**values)
    if result.kind not in LHE_BRANCHES.values() or not result.label or not result.members:
        raise ValueError('Invalid LHE member basis')
    if len(set(result.members)) != len(result.members):
        raise ValueError('Duplicate LHE member IDs')
    return result


def _signal_points(nominal_by_genmodel):
    points = {}
    for branch, value in sorted(nominal_by_genmodel.items()):
        suffix = branch.removeprefix('GenModel_')
        point = MASS_SUFFIX.fullmatch(suffix)
        if not branch.startswith('GenModel_') or point is None:
            raise ValueError('Invalid signal GenModel identity: ' + branch)
        topology, stop, lsp = point.groups()
        key = signal_mass_key(topology, int(stop), int(lsp))
        if key in points or not math.isfinite(float(value)):
            raise ValueError('Duplicate signal point or nonfinite nominal denominator: ' + key)
        points[key] = (branch, suffix, float(value))
    if not points:
        raise ValueError('Explicit per-mass nominal signal coverage is required')
    return points


def _signal_specs(prescriptions):
    specs = {}
    for kind, value in sorted(prescriptions.items()):
        spec = _record_prescription(value.record() if isinstance(value, Prescription) else value)
        if kind not in MEMBER_RUNS_BRANCHES or spec.kind != kind:
            raise ValueError('LHE denominator member kind mismatch: ' + kind)
        if kind == 'scale' and (set(spec.members) - (set(SCALE_COORDINATES) | {(.5, 2.), (2., .5)})
                               or set(SCALE_COORDINATES) - set(spec.members) - {(1., 1.)}):
            raise ValueError('Invalid scale denominator coordinates')
        specs[kind] = spec
    return specs


def _sumw_matches(actual, expected, atol, rtol):
    return math.isfinite(actual) and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol)


def _signal_run_member_values(runs, name, nominal_branch, nominal, spec):
    source = runs[name]
    title = re.sub(r'\s+', ' ', source.title).strip()
    event_branch = 'LHEPdfWeight' if spec.kind == 'pdf' else 'LHEScaleWeight'
    definition = rf'sum of genEventWeight\s*\*\s*{event_branch}\[i\]'
    if re.fullmatch(definition, title, re.I):
        relative = False
    elif (re.fullmatch(definition + ',? divided by ' + re.escape(nominal_branch), title, re.I)
          or re.fullmatch(definition + ',? divided by genEventSumw,? for model label '
                          + re.escape(nominal_branch.removeprefix('genEventSumw_')), title, re.I)):
        relative = True
    else:
        raise ValueError('Runs member-weight title does not certify the per-mass signed denominator')
    values = ak.to_list(source.array(library='ak'))
    if len(values) != len(nominal):
        raise ValueError('Runs member/nominal row counts differ')
    totals = [[] for _ in spec.members]
    for den, row in zip(nominal, values):
        if len(row) == 0 and den == 0.:
            continue
        if len(row) != len(spec.members) or not np.isfinite(row).all():
            raise ValueError('Incomplete or nonfinite Runs member array')
        if relative and den == 0.:
            raise ValueError('Zero signed per-run nominal sum prevents recovering Runs member sums')
        for store, weight in zip(totals, row):
            store.append(float(weight) * float(den) if relative else float(weight))
    return [math.fsum(values) for values in totals]


def signal_runs_member_denominators(runs, prescriptions, nominal_by_genmodel, *, atol=1e-7, rtol=1e-7):
    """Read only explicitly mass-keyed, documented Runs member denominators.

    `prescriptions` maps kind to *original Events* member coordinates or IDs.
    `nominal_by_genmodel` is the accepted original-source per-point Runs map.
    Inclusive NanoAOD arrays or generic relative titles never prove coverage.
    """
    points = _signal_points(nominal_by_genmodel)
    specs = _signal_specs(prescriptions)
    keys = {str(k).split(';', 1)[0] for k in runs.keys()}
    observed = {key.removeprefix('genEventSumw_') for key in keys
                if key.startswith('genEventSumw_') and MASS_SUFFIX.fullmatch(key.removeprefix('genEventSumw_'))}
    expected = {point[1] for point in points.values()}
    if observed != expected:
        raise ValueError('Original Runs mass-point inventory differs from accepted input')
    results, unavailable = {}, []
    for key, (_, suffix, known) in points.items():
        nominal_branch = 'genEventSumw_' + suffix
        raw = np.asarray(runs[nominal_branch].array(library='np'), float)
        if raw.ndim != 1 or not np.isfinite(raw).all() or not _sumw_matches(math.fsum(raw), known, atol, rtol):
            raise ValueError('Original Runs signed nominal denominator differs from accepted input: ' + key)
        result = results[key] = dict(nominal_sumw=math.fsum(raw), variations={})
        for kind, prefix in MEMBER_RUNS_BRANCHES.items():
            branch = prefix + '_' + suffix
            if branch not in keys or kind not in specs:
                unavailable.append(dict(point=key, kind=kind, branch=branch, reason='missing_mass_member_branch'
                    if branch not in keys else 'missing_event_member_definition'))
                continue
            try:
                spec = specs[kind]
                values = _signal_run_member_values(runs, branch, nominal_branch, raw, spec)
                _check_member_central(key, known, values, spec, atol, rtol)
            except ValueError as exc:
                unavailable.append(dict(point=key, kind=kind, branch=branch, reason=str(exc)))
                continue
            result['variations'][kind] = dict(sumw=values, prescription=spec.record(), branch=branch)
            if (missing := _missing_pdf_members(spec)):
                unavailable.append(dict(point=key, kind=kind, branch=branch, member_ids=missing,
                                         reason='incomplete_event_member_definition'))
    return dict(status='incomplete' if unavailable else 'complete', source='mass_keyed_runs', points=results,
                uncovered=unavailable,
                unused_inclusive_branches=sorted(set(MEMBER_RUNS_BRANCHES.values()) & keys))


def _check_member_central(point, nominal, values, spec, atol, rtol):
    # PDF member zero may be an alternate generation PDF. Only the (1,1)
    # scale member is declared to be the generated nominal event weight.
    if (spec.kind == 'scale' and (1., 1.) in spec.members
            and not _sumw_matches(values[spec.members.index((1., 1.))], nominal, atol, rtol)):
        raise ValueError('LHE (muR, muF)=(1,1) signed sum does not close to nominal: ' + point)


def signal_event_member_denominators(tree, nominal_by_genmodel, *, entries, step_size=10000,
                                     atol=1e-7, rtol=1e-7, prescriptions=None):
    """Alternate evidence: stream full *source Events*, not selected/retained rows.

    Read only generator branches. Require complete coverage, exclusive GenModel
    tags, and signed central closure against the accepted per-mass Runs sums.
    """
    points = _signal_points(nominal_by_genmodel)
    genmodel = [point[0] for point in points.values()]
    if (step_size <= 0 or tree.num_entries != entries
            or set(genmodel) != {str(key).split(';', 1)[0] for key in tree.keys()
                                if str(key).startswith('GenModel_')}):
        raise ValueError('Original Events count/mass support differs from the accepted source')
    if prescriptions is None:
        constructors = {'pdf': pdf_prescription, 'scale': scale_prescription}
        prescriptions = {kind: constructors[kind](tree[branch].title)
                         for branch, kind in LHE_BRANCHES.items() if branch in tree}
    specs = _signal_specs(prescriptions)
    nominal = {point: [] for point in points}
    varied = {point: {kind: [[] for _ in spec.members] for kind, spec in specs.items()} for point in points}
    nrows, counts = 0, {point: 0 for point in points}
    branches = [b for b, kind in LHE_BRANCHES.items() if kind in specs]
    for part in tree.iterate(['genWeight', *genmodel, *branches], step_size=step_size, library='ak'):
        weight = np.asarray(part['genWeight'], float)
        if not np.isfinite(weight).all():
            raise ValueError('Nonfinite original Events generator weight')
        mask = np.column_stack([np.asarray(part[b], bool) for b in genmodel])
        if np.any(mask.sum(axis=1) != 1):
            raise ValueError('Original Events require exactly one matching GenModel per row')
        values = {}
        for branch in branches:
            kind = LHE_BRANCHES[branch]
            data = ak.to_numpy(part[branch])
            if data.shape != (len(part), len(specs[kind].members)) or not np.isfinite(data).all():
                raise ValueError('Incomplete signal LHE event member array: ' + branch)
            values[kind] = data
        for i, point in enumerate(points):
            sel = mask[:, i]
            if not np.any(sel):
                continue
            counts[point] += int(np.count_nonzero(sel))
            w = weight[sel]
            nominal[point].append(math.fsum(w))
            for kind, data in values.items():
                totals = np.einsum('ij,i->j', data[sel], w, dtype=np.float64)
                for store, value in zip(varied[point][kind], totals):
                    store.append(value)
        nrows += len(part)
    if nrows != entries:
        raise ValueError('Incomplete original Events traversal for LHE denominators')
    results, unavailable = {}, []
    for key, (_, _, known) in points.items():
        central = math.fsum(nominal[key])
        if not _sumw_matches(central, known, atol, rtol):
            raise ValueError('Original Events per-mass signed genWeight does not close to Runs: ' + key)
        result = results[key] = dict(nominal_sumw=central, events=counts[key], variations={})
        for branch, kind in LHE_BRANCHES.items():
            if kind not in specs:
                unavailable.append(dict(point=key, kind=kind, branch=branch, reason='missing_event_member_definition'
                    if branch in tree else 'missing_event_member_branch'))
                continue
            spec = specs[kind]
            values = [math.fsum(part) for part in varied[key][kind]]
            _check_member_central(key, central, values, spec, atol, rtol)
            result['variations'][kind] = dict(sumw=values, prescription=spec.record(), branch=branch)
            if (missing := _missing_pdf_members(spec)):
                unavailable.append(dict(point=key, kind=kind, branch=branch, member_ids=missing,
                                         reason='incomplete_event_member_definition'))
    return dict(status='incomplete' if unavailable else 'complete', source='all_source_events', points=results,
                uncovered=unavailable)


def _check_member_leaf(histogram, count):
    axes = [np.asarray(axis, float) for axis in histogram['edges']]
    if not axes or any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
                       or np.any(a[1:] <= a[:-1]) for a in axes):
        raise ValueError('Invalid LHE histogram axes')
    shape = tuple(len(a) - 1 for a in axes)
    for field in ('nominal', 'sumw2', 'entries', 'members', 'members_sumw2'):
        values = np.asarray(histogram[field], float)
        if values.shape != ((count, *shape) if field.startswith('members') else shape):
            raise ValueError('Invalid LHE member dimensions: ' + field)
        if not np.isfinite(values).all() or (field not in ('nominal', 'members') and np.any(values < 0)):
            raise ValueError('Invalid LHE member sums: ' + field)
        if field == 'entries' and np.any(values != np.floor(values)):
            raise ValueError('LHE histogram entries must be integer counts')


def _merge_member_record(target, key, record):
    spec = _record_prescription(record['prescription'])
    if key != spec.kind + ':' + spec.label:
        raise ValueError('LHE member key does not match its declared basis')
    outer = target.setdefault(key, {'prescription': spec.record(), 'processes': {}})
    if _record_prescription(outer['prescription']) != spec:
        raise ValueError('Different definitions of LHE member basis: ' + key)
    for process, histograms in sorted(record['processes'].items()):
        node = outer['processes'].setdefault(process, {})
        for path, value in sorted(_leaves(histograms)):
            if not path:
                raise ValueError('Missing LHE histogram path')
            _check_member_leaf(value, len(spec.members))
            current = node
            for part in path[:-1]:
                current = current.setdefault(part, {})
            if path[-1] in current:
                merge_members(current[path[-1]], value)
            else:
                current[path[-1]] = copy.deepcopy(value)


def merge_member_records(records):
    """Sum equivalent member IDs only, without reducing to Up/Down endpoints.

    The caller fixes the order of the input records. Each process and PDF
    set remains separate, with hard errors for incompatible axis/ID layouts.
    """
    result = {}
    for record in records:
        for key, value in sorted(record.items()):
            _merge_member_record(result, key, value)
    return result


def _missing_pdf_members(spec):
    if spec.kind != 'pdf':
        return []
    # Preserve the actual member records. Missing central or alpha_s members
    # are coverage findings, not assumptions about their future treatment.
    if spec.total_members <= 0:
        raise ValueError('PDF total member count is unspecified')
    all_members = set(range(spec.total_members))
    if not set(spec.members) <= all_members:
        raise ValueError('PDF member outside the declared basis')
    return sorted(all_members - set(spec.members))


def _payload_source_coverage(payload):
    coverage = payload['lhe_source_audit']
    if coverage['schema'] != SOURCE_AUDIT_SCHEMA:
        raise ValueError('Missing current source coverage receipt')
    if type(coverage['joined_events']) is not int or coverage['joined_events'] < 0:
        raise ValueError('Invalid source join count')
    files = coverage['files']
    if not isinstance(files, dict):
        raise ValueError('Invalid source coverage file table')
    joined, reported, usable_bases = 0, [], set()
    for _, rec in sorted(files.items()):
        n = rec['joined_events']
        if type(n) is not int or n <= 0 or not isinstance(rec['source'], str) or not rec['source']:
            raise ValueError('Invalid joined source file coverage')
        joined += n
        missing = set(rec['missing'])
        branches = rec['branches']
        if missing != set(LHE_BRANCHES) - set(branches) or not set(branches) <= set(LHE_BRANCHES):
            raise ValueError('Incomplete source branch presence audit')
        bases = {}
        for branch, info in sorted(branches.items()):
            raw = _record_prescription(info['source'])
            spec = _record_prescription(info['accumulated'])
            if (info['events'] != n or spec.kind != LHE_BRANCHES[branch]
                    or (raw.kind, raw.label) != (spec.kind, spec.label)
                    or (raw.kind == 'pdf' and raw != spec)):
                raise ValueError('Incomplete event/member coverage for ' + branch)
            missing_members = _missing_pdf_members(spec)
            bases[branch] = dict(prescription=spec.record(), missing_members=missing_members)
            usable_bases.add(spec)
        reported.append(dict(source=rec['source'], events=n, missing_branches=sorted(missing), bases=bases))
    if joined != coverage['joined_events']:
        raise ValueError('Joined event count differs from the source-file receipt totals')
    return joined, reported, usable_bases


def _check_task_payload(task, era, payload):
    if (payload['status'] != 'complete_member_sums' or payload['sr_data_blinded'] is not True
            or payload['scope'] != 'source_lhe_member_sums_not_reduced'):
        raise ValueError('Expected an unreduced, SR-blinded LHE member payload')
    contract = payload['contract']
    if (Path(contract['input']) != ROOT / task['input'] or analysis_era(contract['year']) != era
            or payload['audit']['events_read'] != task['entries']
            or contract['lhe_weights']['source'] != 'original_nanoaod'
            or contract['lhe_weights']['signal_uncertainties'] is not False):
        raise ValueError('LHE receipt does not match the registered background input')
    fingerprint = contract['fingerprint']
    if fingerprint['size'] != task['fingerprint']['size']:
        raise ValueError('LHE receipt size differs from the registered background input')
    if any(rec.get('is_data') or rec.get('is_signal') for rec in payload['input_metadata'].values()):
        raise ValueError('Observed or signal input cannot be part of a background LHE member merge')
    joined, files, usable_bases = _payload_source_coverage(payload)
    if joined > task['entries']:
        raise ValueError('Source coverage exceeds the registered number of retained events')
    member_records = payload['lhe_histograms']
    if any(_record_prescription(r['prescription']) not in usable_bases for r in member_records.values()):
        raise ValueError('Accumulated LHE histogram has no corresponding source branch receipt')
    return member_records, joined, files


def merge_lhe_member_payloads(expected_tasks, load_payload):
    """Audit registered background shards and sum their same-member records once.

    `load_payload(task)` returns the canonical histogram payload, `None` for a
    missing receipt, or raises its I/O error. Partial sums are diagnostic only
    when `status` is `incomplete_member_sums`. No endpoints or new physics
    choices are produced. `members_by_era` preserves the raw member bases.
    """
    tasks = sorted(expected_tasks, key=lambda t: (analysis_era(t['year']), t['input']))
    if not tasks or len({t['input'] for t in tasks}) != len(tasks):
        raise ValueError('Expected unique registered background MC shards')
    merged, counts, process_bases, basis_eras = {}, {}, {}, {}
    missing, rejected, branch_gaps, member_gaps = [], [], {}, {}
    for task in tasks:
        era, name = analysis_era(task['year']), task['input']
        relative = Path(name)
        if relative.parent != Path('inputs') / era or not MC_SHARD.fullmatch(relative.name):
            raise ValueError('LHE merge requires registered background MC only: ' + name)
        data = counts.setdefault(era, dict(expected=0, accepted=0, events_read=0, events_joined=0))
        data['expected'] += 1
        try:
            payload = load_payload(task)
            if payload is None:
                missing.append(name)
                continue
            records, joined, files = _check_task_payload(task, era, payload)
            # Validate the complete shard's members before committing any of
            # it. Per-shard storage remains bounded; no large ROOT readback.
            records = merge_member_records((records,))
            target = merged.setdefault(era, {})
            for key, record in records.items():
                spec = _record_prescription(record['prescription'])
                if key in target and _record_prescription(target[key]['prescription']) != spec:
                    raise ValueError('Different definitions of LHE member basis: ' + key)
                for process, bins in record['processes'].items():
                    previous = target.get(key, {}).get('processes', {}).get(process, {})
                    prior = dict(_leaves(previous))
                    if any(path in prior and old['edges'] != prior[path]['edges'] for path, old in _leaves(bins)):
                        raise ValueError('Incompatible axes for LHE basis: ' + key)
            for key, value in records.items():
                _merge_member_record(target, key, value)
                spec = _record_prescription(value['prescription'])
                basis_eras.setdefault(key, {}).setdefault(spec, set()).add(era)
                for process in value['processes']:
                    process_bases.setdefault((process, spec.kind), set()).add(spec.label)
        except FileNotFoundError:
            missing.append(name)
            continue
        except (KeyError, TypeError, ValueError, OSError) as exc:
            rejected.append(dict(input=name, error=str(exc)))
            continue
        data['accepted'] += 1
        data['events_read'] += task['entries']
        data['events_joined'] += joined
        for rec in files:
            for branch in rec['missing_branches']:
                hit = branch_gaps.setdefault((era, rec['source'], branch), 0)
                branch_gaps[(era, rec['source'], branch)] = hit + rec['events']
            for branch, basis in rec['bases'].items():
                if basis['missing_members']:
                    found = tuple(basis['missing_members'])
                    k = (era, rec['source'], branch, basis['prescription']['label'], found)
                    member_gaps[k] = member_gaps.get(k, 0) + rec['events']
    # Different PDF sets/bases are preserved and explicitly flagged. A
    # combined model can't quietly reinterpret one as another even if their
    # nominal histogram shapes are identical.
    incompatible_process_bases = [dict(process=process, kind=kind, bases=sorted(bases))
        for (process, kind), bases in sorted(process_bases.items()) if len(bases) > 1]
    incompatible_era_bases = [dict(basis=key, definitions=[dict(prescription=spec.record(), eras=sorted(eras))
        for spec, eras in definitions.items()]) for key, definitions in sorted(basis_eras.items()) if len(definitions) > 1]
    complete = not (missing or rejected or branch_gaps or member_gaps or incompatible_era_bases or incompatible_process_bases)
    return dict(schema=MEMBER_MERGE_SCHEMA, status='complete_member_sums' if complete else 'incomplete_member_sums',
        eras=counts, missing_inputs=missing, rejected_inputs=rejected,
        missing_branches=[dict(era=era, source=source, branch=branch, events=n)
                          for (era, source, branch), n in sorted(branch_gaps.items())],
        missing_members=[dict(era=era, source=source, branch=branch, basis=basis, member_ids=list(ids), events=n)
                         for (era, source, branch, basis, ids), n in sorted(member_gaps.items())],
        incompatible_era_bases=incompatible_era_bases, incompatible_process_bases=incompatible_process_bases,
        members_by_era=merged)


def _aligned_signal_denominator(spec, values, nominal):
    values = np.asarray(values, float)
    if values.shape != (len(spec.members),) or not np.isfinite(values).all():
        raise ValueError('Invalid signed signal member denominator array')
    if spec.kind == 'pdf':
        return spec, values
    # Match the existing selected-member ordering, including the nominal
    # identity for the documented eight-member scale representation.
    aligned = Prescription('scale', spec.label, SCALE_COORDINATES,
        central_index=SCALE_COORDINATES.index((1., 1.)), envelope_indices=tuple(range(len(SCALE_COORDINATES))))
    return aligned, np.array([values[spec.members.index(point)] if point in spec.members else nominal
                             for point in SCALE_COORDINATES], float)


def _validate_signal_lhe_payload(task, source, era, payload, atol, rtol):
    if (payload['status'] != 'complete_member_sums' or payload['sr_data_blinded'] is not True
            or payload['scope'] != 'source_lhe_member_sums_not_reduced'):
        raise ValueError('Expected an unreduced, SR-blinded signal member payload')
    contract = payload['contract']
    entry = contract['lhe_weights']
    if (Path(contract['input']) != ROOT / task['input'] or analysis_era(contract['year']) != era
            or contract['shape_mode'] != 'cms_trota_jme' or contract['variations'] != ['nominal']
            or payload['audit']['events_read'] != task['entries']
            or entry['signal_uncertainties'] is not True or entry['source'] != 'original_nanoaod'
            or entry['selection'] != 'new_uuid_corrected_trota_nominal'
            or entry['normalization'] != 'unscaled_selected_yields_and_signed_per_mass_denominators'):
        raise ValueError('Signal LHE receipt does not match the corrected nominal population')
    fingerprint = contract['fingerprint']
    if fingerprint['size'] != task['fingerprint']['size']:
        raise ValueError('Signal member payload size differs from the registered input')
    datasets = list(payload['input_metadata'].values())
    if (len(datasets) != 1 or datasets[0].get('is_data') is not False or datasets[0].get('is_signal') is not True
            or datasets[0]['dataset'] != source['dataset']):
        raise ValueError('Signal LHE input must match the registered signal-only dataset')
    expected_mass = {'GenModel_' + name: float(value) for name, value in source['nano_metadata']['mass_sumw'].items()}
    registered = entry['registered_source']
    expected_source = dict(entries=source['nevents'], bytes=source['size'])
    if (any(registered.get(k) != v for k, v in expected_source.items())
            or registered.get('nominal_by_genmodel') != expected_mass or registered.get('dataset') != source['dataset']):
        raise ValueError('Signal LHE original source coverage or point registry differs')
    points = _signal_points(expected_mass)
    denoms = payload['lhe_signal_denominators']
    if (any(denoms['source_fingerprint'].get(k) != v for k, v in expected_source.items())
            or set(denoms['points']) != set(points)
            or denoms['source'] not in ('mass_keyed_runs', 'all_source_events')
            or denoms['status'] not in ('complete', 'incomplete')):
        raise ValueError('Invalid signed per-source signal denominator receipt')
    accumulated, incomplete = {}, list(denoms['uncovered'])
    for mass, point in points.items():
        known = point[2]
        origin = denoms['points'][mass]
        nominal = float(origin['nominal_sumw'])
        if not _sumw_matches(nominal, known, atol, rtol):
            raise ValueError('Signal source signed sum did not close to the registered Runs mass: ' + mass)
        aligned = {}
        for kind, value in origin['variations'].items():
            spec = _signal_specs({kind: value['prescription']})[kind]
            _check_member_central(mass, nominal, value['sumw'], spec, atol, rtol)
            aligned[kind] = _aligned_signal_denominator(spec, value['sumw'], nominal)
        for kind in set(MEMBER_RUNS_BRANCHES) - set(aligned):
            if not any(r['point'] == mass and r['kind'] == kind for r in incomplete):
                incomplete.append(dict(point=mass, kind=kind, reason='missing_mass_member_denominator'))
        accumulated[mass] = (nominal, aligned)
    if (denoms['status'] == 'complete') != (not incomplete) or (denoms['status'] == 'complete' and denoms.get('unidentified_branches')):
        raise ValueError('Signal member denominator coverage and status disagree')
    records = merge_member_records((payload['lhe_histograms'],))
    for record in records.values():
        spec = _record_prescription(record['prescription'])
        for mass, hist in record['processes'].items():
            chosen = accumulated.get(mass, (None, {}))[1].get(spec.kind)
            if chosen is None or chosen[0] != spec:
                raise ValueError('Signal raw member numerator does not align with signed denominator: ' + mass)
            if any(len(path) != 5 or path[3] != mass for path, _ in _leaves(hist)):
                raise ValueError('Signal numerator contains an unrelated mass-point histogram')
    return records, accumulated, incomplete, denoms.get('unidentified_branches', {})


def merge_signal_lhe_member_payloads(expected_tasks, load_payload, expected_nominal_by_era,
                                     *, atol=1e-7, rtol=1e-7):
    """Aggregate raw UUID signal numerator/denominator members by era and mass.

    Tasks supply `input/year/fingerprint/entries`, optionally the registered
    `source` from production/tasks.json. `expected_nominal_by_era` supplies the
    independently accepted summed signed Runs denominators of all target masses.
    The return value never divides a yield by any denominator or attaches a
    nuisance: partial results remain diagnostic until coverage is complete.
    """
    tasks = sorted(expected_tasks, key=lambda t: (analysis_era(t['year']), t['input']))
    if not tasks or len({t['input'] for t in tasks}) != len(tasks):
        raise ValueError('Expected unique registered UUID signal shards')
    by_era, mass_sums, members, sources = {}, {}, {}, {}
    missing, rejected, uncovered = [], [], []
    for task in tasks:
        era, name = analysis_era(task['year']), task['input']
        path = Path(name)
        if path.parent != Path('inputs') / era or not SIGNAL_SHARD.fullmatch(path.name):
            raise ValueError('Only the registered replacement signal population can be merged')
        source = task.get('source') or _signal_inventory()[(era, path.name)]
        seen = sources.setdefault(era, set())
        identity = Path(source['name']).stem
        if identity in seen or identity != path.stem.removeprefix('signal_'):
            raise ValueError('Duplicate or mismatched UUID signal input in era ' + era)
        seen.add(identity)
        era_counts = by_era.setdefault(era, dict(expected=0, accepted=0))
        era_counts['expected'] += 1
        target = members.setdefault(era, {})
        sums = mass_sums.setdefault(era, {})
        try:
            payload = load_payload(task)
            if payload is None:
                missing.append(name)
                continue
            records, denom, gaps, unknown = _validate_signal_lhe_payload(task, source, era, payload, atol, rtol)
            for mass, (_, variations) in denom.items():
                for kind, (basis, _) in variations.items():
                    previous = sums.get(mass, {}).get('variations', {}).get(kind)
                    if previous is not None and previous['spec'] != basis:
                        raise ValueError('Incompatible signal member basis for ' + mass + '/' + kind)
            # Cross-shard basis/shape validation before mutating either sum.
            for key, record in records.items():
                prior = target.get(key)
                if prior is None:
                    continue
                if _record_prescription(prior['prescription']) != _record_prescription(record['prescription']):
                    raise ValueError('Incompatible signal numerator member basis: ' + key)
                for mass, hist in record['processes'].items():
                    leaves = dict(_leaves(prior['processes'].get(mass, {})))
                    if any(p in leaves and leaves[p]['edges'] != new['edges'] for p, new in _leaves(hist)):
                        raise ValueError('Incompatible signal numerator axes: ' + key + '/' + mass)
            for key, record in records.items():
                _merge_member_record(target, key, record)
            for mass, (nominal, variations) in denom.items():
                node = sums.setdefault(mass, dict(nominal=[], variations={}))
                node['nominal'].append(nominal)
                for kind, (spec, vals) in variations.items():
                    group = node['variations'].setdefault(kind, dict(spec=spec, sums=[]))
                    group['sums'].append(vals)
            if gaps or unknown:
                uncovered.append(dict(input=name, source=source['name'], missing_mass_members=gaps,
                                      unidentified_branches=unknown))
        except FileNotFoundError:
            missing.append(name)
            continue
        except (KeyError, TypeError, ValueError, OSError) as exc:
            rejected.append(dict(input=name, error=str(exc)))
            continue
        era_counts['accepted'] += 1
    closures, denominators = [], {}
    if set(by_era) != set(expected_nominal_by_era):
        raise ValueError('Expected nominal Runs denominators for the exact target era set')
    for era, sums in mass_sums.items():
        nominal = expected_nominal_by_era[era]
        rows = denominators[era] = {}
        for mass in sorted(set(sums) | set(nominal)):
            source = sums.get(mass, dict(nominal=[], variations={}))
            total = math.fsum(source['nominal'])
            expected = nominal.get(mass)
            if isinstance(expected, dict):
                expected = expected['sumw_mass_point']
            if not source['nominal'] or expected is None or not _sumw_matches(total, float(expected), atol, rtol):
                closures.append(dict(era=era, point=mass, sumw=total, expected=expected))
            rows[mass] = dict(nominal_sumw=total, files=len(source['nominal']), variations={
                kind: dict(sumw=[math.fsum(column) for column in np.stack(group['sums']).T],
                           files=len(group['sums']), prescription=group['spec'].record())
                for kind, group in source['variations'].items()})
    complete = not (missing or rejected or uncovered or closures)
    return dict(schema=MEMBER_MERGE_SCHEMA, population=SIGNAL_POPULATION,
        status='complete_member_sums' if complete else 'incomplete_member_sums', eras=by_era,
        missing_inputs=missing, rejected_inputs=rejected, uncovered_sources=uncovered, failed_nominal_closure=closures,
        denominators_by_era=denominators, members_by_era=members)


def normalize_signal_members(merged, normalizations, *, atol=1e-7, rtol=1e-7):
    """Normalize complete raw signed signal members with the accepted xsec × lumi.

    `normalizations` maps target era to its accepted full normalization table
    (`luminosity_pb` and `signal_mass_points`). This pure step never combines
    eras or members, clips bins, or creates fitted nuisance endpoints.
    """
    if (merged.get('schema') != MEMBER_MERGE_SCHEMA or merged.get('population') != SIGNAL_POPULATION
            or merged.get('status') != 'complete_member_sums'
            or any(merged[key] for key in ('missing_inputs', 'rejected_inputs', 'uncovered_sources', 'failed_nominal_closure'))):
        raise ValueError('Signal normalization requires a complete unreduced member merge')
    raw, denominators, coverage = (merged[k] for k in ('members_by_era', 'denominators_by_era', 'eras'))
    if not raw or not (set(raw) == set(denominators) == set(coverage) == set(normalizations)):
        raise ValueError('Signal member and normalization eras must match exactly')

    outputs, factors = {}, {}
    for era, records in sorted(raw.items()):
        if (coverage[era]['expected'] <= 0 or coverage[era]['accepted'] != coverage[era]['expected']):
            raise ValueError('Incomplete signed signal source coverage in era ' + era)
        table = normalizations[era]
        lumi = float(table['luminosity_pb'])
        if not math.isfinite(lumi) or lumi <= 0.:
            raise ValueError('Invalid accepted signal luminosity in era ' + era)
        accepted, era_factors = table['signal_mass_points'], {}
        if not accepted or set(accepted) != set(denominators[era]):
            raise ValueError('Signal mass points differ from accepted normalization in era ' + era)
        for mass, row in sorted(denominators[era].items()):
            applied = accepted[mass]
            xsec, known, nominal_factor = (float(applied[k]) for k in ('xsec_pb', 'sumw_mass_point', 'normalization_factor'))
            norm = float(row['nominal_sumw'])
            overall = xsec * lumi
            loc = era + '/' + mass
            if (row['files'] <= 0 or not all(map(math.isfinite, (xsec, known, nominal_factor, norm, overall)))
                    or xsec <= 0. or overall <= 0. or known == 0. or norm == 0. or nominal_factor == 0.
                    or not _sumw_matches(norm, known, atol, rtol)
                    or not _sumw_matches(nominal_factor * known, overall, 0., 1e-10)):
                raise ValueError('Signed nominal sum or accepted factor fails signal closure: ' + loc)
            if set(row['variations']) != set(MEMBER_RUNS_BRANCHES):
                raise ValueError('Missing signed signal variation denominator: ' + loc)
            member_factors = {}
            for kind, value in sorted(row['variations'].items()):
                spec = _signal_specs({kind: value['prescription']})[kind]
                den = np.asarray(value['sumw'], float)
                if (value['files'] != row['files'] or _missing_pdf_members(spec)
                        or den.shape != (len(spec.members),) or not np.isfinite(den).all() or np.any(den == 0.)):
                    raise ValueError('Missing, zero or invalid signed signal member denominator: ' + loc + '/' + kind)
                _check_member_central(mass, known, den, spec, atol, rtol)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    variation_factors = overall / den
                    check = den * variation_factors
                if (not np.isfinite(variation_factors).all() or np.any(variation_factors == 0.)
                        or not np.allclose(check, overall, rtol=1e-10, atol=0.)):
                    raise ValueError('Signal variation factor fails inclusive closure: ' + loc + '/' + kind)
                member_factors[kind] = dict(prescription=spec.record(), factors=variation_factors.tolist())
            era_factors[mass] = dict(nominal_factor=nominal_factor, members=member_factors)
        factors[era] = era_factors

        # Confirm that all source bases describing the same selected signal
        # bins share the same central sums and entries before scaling.
        outputs[era] = {}
        central = {}
        for key, record in sorted(records.items()):
            spec = _record_prescription(record['prescription'])
            if key != spec.kind + ':' + spec.label:
                raise ValueError('Signal numerator key disagrees with its member basis')
            processes = outputs[era].setdefault(key, {'prescription': spec.record(), 'processes': {}})['processes']
            for mass, histograms in sorted(record['processes'].items()):
                if mass not in era_factors or _record_prescription(era_factors[mass]['members'][spec.kind]['prescription']) != spec:
                    raise ValueError('Signal numerator and denominator member bases differ: ' + era + '/' + mass)
                mass_factors = era_factors[mass]
                nominal_factor = mass_factors['nominal_factor']
                varied = np.asarray(mass_factors['members'][spec.kind]['factors'], float)
                for path, entry in _leaves(histograms):
                    if len(path) != 5 or path[0] not in ('highdm', 'lowdm') or path[1] != 'SR' or path[3] != mass:
                        raise ValueError('Signal members must contain only their own nominal SR coordinates')
                    _check_member_leaf(entry, len(varied))
                    ledger = central.setdefault(path, dict(entry=entry, kinds=set()))
                    other = ledger['entry']
                    if (spec.kind in ledger['kinds'] or entry['edges'] != other['edges']
                            or not np.array_equal(entry['entries'], other['entries'])
                            or any(not np.allclose(entry[k], other[k], atol=atol, rtol=rtol)
                                   for k in ('nominal', 'sumw2'))):
                        raise ValueError('Signal member bases disagree on nominal selected yield or binning: ' + era + '/' + mass)
                    ledger['kinds'].add(spec.kind)
                    values = np.asarray(entry['members'], float)
                    widths = varied.reshape((len(varied),) + (1,) * (values.ndim - 1))
                    with np.errstate(over='ignore', invalid='ignore'):
                        data = dict(edges=copy.deepcopy(entry['edges']), entries=copy.deepcopy(entry['entries']),
                            nominal=(np.asarray(entry['nominal'], float) * nominal_factor).tolist(),
                            sumw2=((np.asarray(entry['sumw2'], float) * nominal_factor) * nominal_factor).tolist(),
                            members=(values * widths).tolist(),
                            members_sumw2=((np.asarray(entry['members_sumw2'], float) * widths) * widths).tolist())
                    _check_member_leaf(data, len(varied))
                    node = processes.setdefault(mass, {})
                    for component in path[:-1]:
                        node = node.setdefault(component, {})
                    node[path[-1]] = data
        if any(entry['kinds'] != set(MEMBER_RUNS_BRANCHES) for entry in central.values()):
            raise ValueError('Missing selected signal members for one or more SR coordinates in era ' + era)
    return dict(schema='trotasr_normalized_signal_members_v1', status='normalized_member_sums',
                population=SIGNAL_POPULATION, members_by_era=outputs, normalization_by_era=factors)


def finalize_histograms(records, nominal):
    """Produce weight endpoints after the full member sum, using nominal MC stat.

    Only the affected hard process changes. Other contributions within the
    displayed Top/VV sample remain in the endpoint at their nominal yield.
    Low-dM marginals are projections of the same varied GNN x recoil array.
    The caller is responsible for checking input-file/weight coverage first.
    """
    result, pdf_sets = {}, {}
    for record in records.values():
        prescription = _record_prescription(record['prescription'])
        for process, histograms in record['processes'].items():
            if prescription.kind == 'pdf':
                previous = pdf_sets.setdefault(process, prescription.label)
                if previous != prescription.label:
                    raise ValueError('Multiple PDF bases within one hard process require an explicit combination: ' + process)
            nuisance = ('pdf_' if prescription.kind == 'pdf' else 'QCDscale_') + process
            for path, member_hist in _leaves(histograms):
                mode, region, category, sample, observable = path
                original = nominal[mode]['nominal'][region][category][sample][observable]
                up, down = reduce_members(member_hist, prescription)
                for direction, varied in (('Up', up), ('Down', down)):
                    node = result.setdefault(mode, {}).setdefault(nuisance + direction, {})
                    for key in (region, category, sample):
                        node = node.setdefault(key, {})
                    leaf = node.setdefault(observable, copy.deepcopy(original))
                    leaf['sumw'] = (np.asarray(leaf['sumw']) + varied
                                    - np.asarray(member_hist['nominal'])).tolist()
                    if mode == 'lowdm':
                        if observable != 'gnn_recoil':
                            raise ValueError('Low-dM LHE reduction requires the joint histogram')
                        for name, axis, edges in (('gnn_score', 1, original['edges'][0]),
                                                  ('recoil', 0, original['edges'][1])):
                            node[name] = {'edges': edges, **{field:
                                np.asarray(leaf[field]).sum(axis=axis).tolist()
                                for field in ('sumw', 'sumw2', 'entries')}}
    return result

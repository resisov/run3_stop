"""Immutable, row-aligned nominal corrected TROTA tags for accepted MC ROOTs."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import errno
import json
import os
import re
import shutil
import tempfile

import awkward as ak
import numpy as np
import uproot

from .paths import EARLY_ERAS, ROOT, analysis_era, era_year, tagging_definition
from .systematic_views import anchored_endpoints
from .trota_candidates import input_fields
from .trota_inference import MIXED_FLOATS, MIXED_INTS, P4_FIELDS, bounded_slices, infer_chunk, load_models, p4_view


SCHEMA = 'trotasr_nominal_corrected_mc_trota_friend_v1'
FRIEND_TREE = 'TROTACorrectedNominal'
METADATA = 'TROTACorrectedNominal_metadata'
MC_NAME = re.compile(r'(?:mc_shard|signal_shard|signal_v15(?:_batch)?)_\d+\.root')
READ_CHUNK = 4096
ID_FIELDS = ('file_id', 'entry')
RESOLVED_INTS = ('candidateIndex', 'sourceJetIdx0', 'sourceJetIdx1', 'sourceJetIdx2')
RESOLVED_FLOATS = ('pt', 'eta', 'phi', 'mass', 'FTScore', 'TTScore', 'QCDScore', 'QCDDiscriminant')


def _layout(kind, ints, floats):
    prefix = 'Top' + kind + '1pct'
    return {('n' + prefix): 'int32',
            **{prefix + '_' + k: 'var * int32' for k in ints},
            **{prefix + '_' + k: 'var * float32' for k in floats}}


MIXED = _layout('Mixed', MIXED_INTS, MIXED_FLOATS)
RESOLVED = _layout('Resolved', RESOLVED_INTS, RESOLVED_FLOATS)


def _source_record(task, source):
    era = analysis_era(task['year'])
    relative = Path(task['input'])
    exact_path = ROOT / relative
    if (relative.parent != Path('inputs') / era or not MC_NAME.fullmatch(relative.name)
            or Path(source) != exact_path or exact_path.resolve() != exact_path):
        raise ValueError('Expected an accepted, non-symlink intermediate MC ROOT path')
    if not exact_path.is_file():
        raise ValueError('MC input is not a regular file')
    tag = tagging_definition(era)
    return dict(schema=SCHEMA, status='complete', mc_only=True, sr_observations_read=False,
                input=str(relative), era=era, events=int(task['entries']),
                p4='canonical_nominal_anchored_endpoints',
                models={kind: dict(wp=tag[kind + '_wp'])
                        for kind in ('mixed', 'resolved')})


def _destination(task, output):
    era = analysis_era(task['year'])
    expected = ROOT / 'inputs' / 'friends' / 'trota_nominal' / era / Path(task['input']).name
    path = Path(output)
    if path != expected or path.resolve() != expected:
        raise ValueError('Friend output must use the canonical MC friend ROOT path')
    return path


@lru_cache(maxsize=1)
def _friend_inventory():
    from TROTASR.workflows.validation_regions import inventory
    return {t['input']: t for tasks in inventory().values() for t in tasks}


def find_friend_task(source, year):
    """Find only an officially registered MC source at its canonical path."""
    source = Path(source)
    era = analysis_era(year)
    relative = Path('inputs') / era / source.name
    if source != ROOT / relative or not MC_NAME.fullmatch(source.name):
        return None
    return _friend_inventory().get(relative.as_posix())


def friend_production_complete():
    """Require a full successful run, not a pilot or partial cache."""
    path = ROOT / 'inputs' / 'friends' / 'trota_nominal' / 'state.json'
    if not path.is_file():
        return False
    state = json.loads(path.read_text())
    return (state.get('schema') == 'trotasr_trota_friends_state_v1'
            and state.get('status') == 'complete' and state.get('pilot') is False
            and state.get('failures') == [] and state.get('active') == 0
            and state.get('completed') == state.get('total') == len(_friend_inventory()))


def _branches(year):
    names = {'file_id', 'entry', 'is_data', 'run', 'luminosityBlock', 'event', 'rho'}
    for collection in input_fields(year):
        names.update(collection)
    for reco, gen, index in (('jet', 'genjet', 'jet_genjet_index_all'),
                             ('fatjet', 'genjetak8', 'fatjet_genjetak8_index_all')):
        names.update(reco + '_' + field for field in ('raw_factor', 'corrected_pt', 'corrected_mass'))
        names.update((index, gen + '_pt_all'))
        if analysis_era(year) in EARLY_ERAS:
            names.update(gen + '_' + field + '_all' for field in ('eta', 'phi'))
    return tuple(sorted(names))


@lru_cache(maxsize=1)
def _models(year):
    return load_models(year)


def _disk_headroom(parent):
    if shutil.disk_usage(parent).free < 80 * 2**30:
        raise OSError(errno.ENOSPC, 'Preserving 80 GiB of hep2 disk headroom; friend publication stopped')


def _validate_open_product(root, expected, path):
    found = json.loads(str(root[METADATA]))
    models = found.get('models', {})
    if (any(found.get(k) != v for k, v in expected.items() if k not in ('models', 'input'))
            or set(models) != set(expected['models'])
            or any(models[k].get('wp') != record['wp'] for k, record in expected['models'].items())):
        raise ValueError('Divergent corrected nominal friend receipt: ' + str(path))
    for key in ('mixed_passing', 'resolved_passing'):
        if type(found.get(key)) is not int or found[key] < 0:
            raise ValueError('Invalid corrected nominal friend candidate count: ' + str(path))
    tree = root[FRIEND_TREE]
    required = {*ID_FIELDS, *MIXED, *RESOLVED}
    if tree.num_entries != expected['events'] or not required <= set(tree.keys()):
        raise ValueError('Invalid corrected nominal friend coverage/schema: ' + str(path))
    for key in ID_FIELDS:
        if tree[key].interpretation.numpy_dtype.kind not in 'iu':
            raise ValueError('Friend event keys must be integers: ' + str(path))
    return found


def _validate_product(path, expected, source=None):
    with uproot.open(path, object_cache=None, array_cache=None) as root:
        found = _validate_open_product(root, expected, path)
        if source is not None:
            with uproot.open(source, object_cache=None, array_cache=None) as original:
                events, friend = original['Events'], root[FRIEND_TREE]
                if (events.num_entries != friend.num_entries
                        or any(events[k].interpretation.numpy_dtype.kind not in 'iu' for k in ID_FIELDS)):
                    raise ValueError('Source Events and nominal friend have incompatible event coverage')
                if events.num_entries:
                    for index in {0, events.num_entries - 1}:
                        interval = dict(entry_start=index, entry_stop=index + 1)
                        keys = events.arrays(list(ID_FIELDS), **interval)
                        cached = friend.arrays(list(ID_FIELDS), **interval)
                        if not all(np.array_equal(a, b) for a, b in zip(_identity(keys), _identity(cached))):
                            raise ValueError('Nominal friend boundary event keys differ from Events')
        return found


def validate_friend(task, source, output):
    """Check the input, adopted working points and whole-tree coverage."""
    return _validate_product(_destination(task, output), _source_record(task, source), source)


@contextmanager
def open_friend(task, source):
    """Open the matching nominal friend with its identity and schema validated."""
    expected = _source_record(task, source)
    output = (ROOT / 'inputs' / 'friends' / 'trota_nominal' / expected['era'] / Path(source).name)
    with uproot.open(_destination(task, output), object_cache=None, array_cache=None) as root:
        _validate_open_product(root, expected, output)
        yield root[FRIEND_TREE]


def _resolved_columns(state, n):
    record = state['resolved']
    keep = record['passing']
    ev = record['event'][keep]
    # infer_chunk emits candidates in input row order. Assert before unflattening.
    if np.any(ev < 0) or np.any(ev >= n) or np.any(ev[1:] < ev[:-1]):
        raise ValueError('Corrected Resolved candidates are not aligned with event rows')
    counts = np.bincount(ev, minlength=n).astype(np.int32)
    sources, kin, scores = record['sources'][keep], record['p4'][keep], record['scores'][keep]
    flat = (record['index'][keep], *sources.T, *kin.T, *scores.T, record['discriminant'][keep])
    prefix = 'TopResolved1pct'
    return {('n' + prefix): counts,
            **{(prefix + '_' + key): ak.unflatten(v, counts)
               for key, v in zip(RESOLVED_INTS + RESOLVED_FLOATS, flat)}}


def _identity(a):
    values = tuple(np.asarray(a[k]) for k in ID_FIELDS)
    if any(v.dtype.kind not in 'iu' for v in values):
        raise ValueError('Friend event keys must remain integers')
    return values


def write_friend(task, source, output):
    """Calculate all rows once; atomically publish a different ROOT without overwriting."""
    expected = _source_record(task, source)
    output = _destination(task, output)
    if output.exists():
        return _validate_product(output, expected, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    _disk_headroom(output.parent)
    fd, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.pending', dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        year = era_year(expected['era'])
        events_seen = mixed = resolved = 0
        with uproot.open(source, object_cache=None, array_cache=None) as src, uproot.recreate(temporary) as target:
            tree = src['Events']
            if tree.num_entries != expected['events']:
                raise ValueError('Accepted input event count changed: ' + str(source))
            key_types = {k: tree[k].interpretation.numpy_dtype for k in ID_FIELDS}
            if any(t.kind not in 'iu' for t in key_types.values()):
                raise ValueError('Source event keys must be integers: ' + str(source))
            target.mktree(FRIEND_TREE, {**key_types, **MIXED, **RESOLVED}, title='Nominal corrected MC TROTA tags')
            for a in tree.iterate(_branches(year), step_size=READ_CHUNK):
                _disk_headroom(output.parent)
                if ak.any(a['is_data']):
                    raise ValueError('Observed rows are forbidden in corrected nominal MC friends')
                keys = _identity(a)
                momenta = anchored_endpoints(a, year, nominal_only=True)['nominal']
                view = p4_view(a, {k: momenta[k.replace('nanoaod', 'corrected')] for k in P4_FIELDS})
                rows_seen = 0
                for sl in bounded_slices(view, year=year):
                    out, _, audit, state = infer_chunk(view[sl], _models(year))
                    if not all(np.array_equal(x, y[sl]) for x, y in zip(_identity(out), keys)):
                        raise ValueError('Inference changed the ordering of event keys')
                    columns = {k: out[k] for k in (*ID_FIELDS, *MIXED)}
                    columns.update(_resolved_columns(state, len(out)))
                    if (int(ak.sum(columns['nTopMixed1pct'])) != audit['mixed_passing']
                            or int(np.sum(columns['nTopResolved1pct'])) != audit['resolved_passing']):
                        raise ValueError('Corrected candidate counts disagree with inference')
                    target[FRIEND_TREE].extend(columns)
                    rows_seen += len(out)
                    mixed += audit['mixed_passing']
                    resolved += audit['resolved_passing']
                if rows_seen != len(a):
                    raise ValueError('Corrected nominal TROTA did not cover the whole event slice')
                events_seen += rows_seen
            if events_seen != expected['events']:
                raise ValueError('Corrected nominal TROTA did not cover all Events rows')
            receipt = dict(expected, mixed_passing=mixed, resolved_passing=resolved)
            target[METADATA] = json.dumps(receipt, sort_keys=True, allow_nan=False)
        _validate_product(temporary, receipt, source)
        _source_record(task, source)
        _disk_headroom(output.parent)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            return _validate_product(output, expected, source)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def read_friend_slice(friend_tree, original, start, stop, *, selection=None):
    """Attach corrected Mixed tags and return the canonical Resolved map.

    Call after validate_friend, for a contiguous Events slice *before* masks.
    Event keys are compared row-by-row; weights, corrected jet momenta and
    unrelated branches are never changed. An optional mask is applied only
    after the complete range is aligned. There is no legacy-tag fallback.
    """
    if start < 0 or stop < start or stop - start != len(original) or stop > friend_tree.num_entries:
        raise ValueError('Invalid Events range for corrected nominal friend')
    if 'is_data' in original.fields and ak.any(original['is_data']):
        raise ValueError('The nominal corrected TROTA friend is MC-only')
    a = friend_tree.arrays([*ID_FIELDS, *MIXED, *RESOLVED], entry_start=start, entry_stop=stop)
    keys = _identity(original)
    if not all(np.array_equal(x, y) for x, y in zip(_identity(a), keys)):
        raise ValueError('Corrected nominal friend event keys disagree with Events slice')
    if len(np.unique(np.rec.fromarrays(keys))) != len(original):
        raise ValueError('Duplicate event keys in corrected nominal friend slice')
    if selection is not None:
        selection = np.asarray(selection)
        if selection.dtype.kind != 'b' or selection.shape != (len(original),):
            raise ValueError('Invalid event selection for corrected nominal friend')
        if not selection.all():
            original, a = original[selection], a[selection]
            keys = tuple(k[selection] for k in keys)
    for names in (MIXED, RESOLVED):
        count, *fields = names
        counts = np.asarray(a[count])
        if np.any(counts < 0) or any(not np.array_equal(np.asarray(ak.num(a[k])), counts) for k in fields):
            raise ValueError('Corrected nominal friend has unaligned candidate columns')
    updated = original
    for k in MIXED:
        updated = ak.with_field(updated, a[k], k)
    prefix = 'TopResolved1pct_'
    counts = np.asarray(a['nTopResolved1pct'])
    if not np.any(counts):
        return updated, {}
    use = ('candidateIndex', 'sourceJetIdx0', 'sourceJetIdx1', 'sourceJetIdx2', 'QCDDiscriminant', 'eta', 'mass')
    columns = [ak.to_list(a[prefix + k]) for k in use]
    resolved = {}
    for i in np.flatnonzero(counts):
        key = (int(keys[0][i]), int(keys[1][i]))
        if key in resolved:
            raise ValueError('Duplicate event key in corrected Resolved friend')
        resolved[key] = [dict(index=index, jets=[j0, j1, j2], score=score, eta=eta, mass=mass)
                         for index, j0, j1, j2, score, eta, mass in zip(*(v[i] for v in columns))]
    return updated, resolved

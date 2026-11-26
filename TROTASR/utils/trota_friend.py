"""Immutable, row-aligned nominal corrected TROTA tags for accepted MC ROOTs."""
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property, lru_cache
from operator import index
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
                        if not _same_identity(_identity(keys), _identity(cached)):
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


def _same_identity(left, right):
    for a, b in zip(left, right):
        if a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or a.shape != b.shape:
            return False
        if a.dtype.kind != b.dtype.kind:
            signed = a if a.dtype.kind == 'i' else b
            if np.any(signed < 0):
                return False
            # Comparing uint64 to int64 directly can promote both to float64.
            a, b = a.astype(np.uint64, copy=False), b.astype(np.uint64, copy=False)
        if not np.array_equal(a, b):
            return False
    return True


def _candidate_support(a, year):
    """TROTA needs three AK4s, or two AK4s plus an AK8. Do not cut on raw pT."""
    resolved, jets, fatjets = input_fields(year)
    def count(group, id_index):
        return np.asarray(ak.sum((a[group[id_index]] != 0) & (abs(a[group[1]]) < 2.5), axis=1))
    return (count(resolved, 6) >= 3) | ((count(jets, 6) >= 2) & (count(fatjets, 9) >= 1))


def _friend_rows(keys, positions, columns, layouts=(MIXED, RESOLVED)):
    """Scatter accepted tag lists among explicit empty rows, keeping every source identity."""
    result = dict(zip(ID_FIELDS, keys))
    for layout in layouts:
        count, *fields = layout
        totals = np.zeros(len(keys[0]), dtype=np.int32)
        if len(positions):
            totals[positions] = np.asarray(columns[count])
        result[count] = totals
        for field in fields:
            flat = (ak.flatten(columns[field]) if len(positions)
                    else np.empty(0, dtype=layout[field].split()[-1]))
            result[field] = ak.unflatten(flat, totals)
    return result


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
            from .reader import project_candidate_entries
            r, j, f = input_fields(year)
            gates = tuple(dict.fromkeys((*ID_FIELDS, 'is_data', r[1], r[6], j[1], j[6], f[1], f[9])))
            rest = tuple(k for k in _branches(year) if k not in gates)
            for first in tree.iterate(gates, step_size=READ_CHUNK):
                _disk_headroom(output.parent)
                if ak.any(first['is_data']):
                    raise ValueError('Observed rows are forbidden in corrected nominal MC friends')
                keys = _identity(first)
                positions = np.flatnonzero(_candidate_support(first, year))
                if not len(positions):
                    target[FRIEND_TREE].extend(_friend_rows(keys, positions, {}))
                    events_seen += len(first)
                    continue
                heavy = project_candidate_entries(tree, rest, events_seen + positions)
                a = ak.zip({**{k: first[k][positions] for k in gates},
                            **{k: heavy[k] for k in rest}}, depth_limit=1)
                momenta = anchored_endpoints(a, year, nominal_only=True)['nominal']
                view = p4_view(a, {k: momenta[k.replace('nanoaod', 'corrected')] for k in P4_FIELDS})
                rows_seen = 0
                row_start = 0
                for sl in bounded_slices(view, year=year):
                    out, _, audit, state = infer_chunk(view[sl], _models(year), materialize_resolved=False)
                    take = positions[sl]
                    if not _same_identity(_identity(out), tuple(k[take] for k in keys)):
                        raise ValueError('Inference changed the ordering of event keys')
                    columns = {k: out[k] for k in (*ID_FIELDS, *MIXED)}
                    columns.update(_resolved_columns(state, len(out)))
                    if (int(ak.sum(columns['nTopMixed1pct'])) != audit['mixed_passing']
                            or int(np.sum(columns['nTopResolved1pct'])) != audit['resolved_passing']):
                        raise ValueError('Corrected candidate counts disagree with inference')
                    row_stop = len(first) if sl.stop == len(a) else int(take[-1]) + 1
                    target[FRIEND_TREE].extend(_friend_rows(tuple(k[row_start:row_stop] for k in keys),
                                                            take - row_start, columns))
                    row_start = row_stop
                    rows_seen += len(out)
                    mixed += audit['mixed_passing']
                    resolved += audit['resolved_passing']
                if rows_seen != len(a) or row_start != len(first):
                    raise ValueError('Corrected nominal TROTA did not cover the whole event slice')
                events_seen += len(first)
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


class _ResolvedCandidateRow(Sequence):
    """Materialize the legacy candidate record only when a consumer visits it."""

    def __init__(self, columns, start, stop):
        self._columns, self._start, self._stop = columns, start, stop

    def __len__(self):
        return self._stop - self._start

    def _candidate(self, i):
        idx, j0, j1, j2, score, eta, mass = (column[i] for column in self._columns)
        return dict(index=int(idx), jets=[int(j0), int(j1), int(j2)],
                    score=float(score), eta=float(eta), mass=float(mass))

    def __iter__(self):
        return (self._candidate(i) for i in range(self._start, self._stop))

    def __getitem__(self, i):
        indices = range(self._start, self._stop)
        if isinstance(i, slice):
            return [self._candidate(j) for j in indices[i]]
        return self._candidate(indices[index(i)])

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))


class ResolvedFriendRows(Mapping):
    """Flat candidate columns with offsets and an event-row-aligned view.

    Projection and support masks share candidate buffers. Legacy lookup by
    (file_id, entry) indexes only populated rows, lazily; aligned callers can
    use ``row(i)`` directly without building that index.
    """

    def __init__(self, keys, offsets, columns, rows=None, enabled=None):
        self._keys, self._offsets, self._columns = keys, offsets, columns
        self._rows = np.arange(len(keys[0]), dtype=np.intp) if rows is None else rows
        self._enabled = enabled

    @property
    def row_count(self):
        return len(self._rows)

    @cached_property
    def _counts(self):
        n = self._offsets[self._rows + 1] - self._offsets[self._rows]
        return n if self._enabled is None else np.where(self._enabled, n, 0)

    @property
    def candidate_count(self):
        return int(self._counts.sum())

    def matches_rows(self, file_ids, entries):
        return _same_identity((np.asarray(file_ids), np.asarray(entries)),
                              tuple(k[self._rows] for k in self._keys))

    def select_rows(self, selection):
        """Take a slice, boolean mask, or distinct positional row indices."""
        if not isinstance(selection, slice):
            selection = np.asarray(selection)
            kind = selection.dtype.kind
            if (selection.ndim != 1 or kind not in 'biu'
                    or (kind == 'b' and len(selection) != self.row_count)):
                raise ValueError('Invalid Resolved friend row selection')
            if kind != 'b' and (np.any(selection < 0) or np.any(selection >= self.row_count)
                    or (np.any(selection[1:] <= selection[:-1])
                        and len(np.unique(selection)) != len(selection))):
                raise ValueError('Invalid Resolved friend row indices')
        enabled = None if self._enabled is None else self._enabled[selection]
        return type(self)(self._keys, self._offsets, self._columns,
                          self._rows[selection], enabled)

    def mask_rows(self, mask):
        """Suppress candidates outside support without removing event rows."""
        mask = np.asarray(mask)
        if mask.dtype.kind != 'b' or mask.shape != (self.row_count,):
            raise ValueError('Invalid Resolved friend row mask')
        if mask.all():
            return self
        enabled = mask.copy() if self._enabled is None else self._enabled & mask
        return type(self)(self._keys, self._offsets, self._columns, self._rows, enabled)

    def row(self, i):
        i = index(i)
        source = self._rows[i]
        if self._enabled is not None and not self._enabled[i]:
            return ()
        start, stop = map(int, self._offsets[source:source + 2])
        return _ResolvedCandidateRow(self._columns, start, stop) if start != stop else ()

    @cached_property
    def _populated(self):
        return np.flatnonzero(self._counts)

    @cached_property
    def _lookup(self):
        files, entries = (c[self._rows[self._populated]] for c in self._keys)
        # Keep integer fields separate (including uint64) and exact.
        order = np.lexsort((entries, files))
        keys = np.rec.fromarrays((files[order], entries[order]))
        bounds = tuple(np.iinfo(dtype) for dtype, _ in keys.dtype.fields.values())
        return keys, self._populated[order], bounds

    def __len__(self):
        return len(self._populated)

    def __iter__(self):
        files, entries = self._keys
        return ((int(files[i]), int(entries[i])) for i in self._rows[self._populated])

    def __getitem__(self, key):
        keys, rows, bounds = self._lookup
        if isinstance(key, tuple) and len(key) == 2 and len(keys):
            try:
                # Bounds must be checked before casting (older NumPy can wrap).
                values = tuple(map(index, key))
                if all(b.min <= v <= b.max for v, b in zip(values, bounds)):
                    target = np.array(values, dtype=keys.dtype)
                    i = keys.searchsorted(target)
                    if i < len(keys) and keys[i] == target:
                        return self.row(rows[i])
            except TypeError:
                pass
        raise KeyError(key)


def read_friend_slice(friend_tree, original, start, stop, *, selection=None,
                      row_indices=None, project=None):
    """Attach corrected Mixed tags and return row-aligned Resolved columns.

    Call after validate_friend. By default, ``original`` is the whole
    contiguous range. For a gathered Events range, ``row_indices`` are unique
    absolute tree positions inside [start, stop), in original's order. The
    optional projector receives sorted positions. Keys are always compared
    row-by-row before applying the optional boolean selection. Weights,
    kinematics and unrelated branches are not changed.
    """
    if start < 0 or stop < start or stop > friend_tree.num_entries:
        raise ValueError('Invalid Events range for corrected nominal friend')
    if 'is_data' in original.fields and ak.any(original['is_data']):
        raise ValueError('The nominal corrected TROTA friend is MC-only')
    fields = (*ID_FIELDS, *MIXED, *RESOLVED)
    if row_indices is None:
        if stop - start != len(original) or project is not None:
            raise ValueError('Invalid Events range for corrected nominal friend')
        a = friend_tree.arrays(list(fields), entry_start=start, entry_stop=stop)
    else:
        rows = np.asarray(row_indices)
        if (rows.shape != (len(original),) or rows.dtype.kind not in 'iu'
                or np.any(rows < start) or np.any(rows >= stop)):
            raise ValueError('Invalid gathered positions for corrected nominal friend')
        if not len(rows):
            a = friend_tree.arrays(list(fields), entry_start=start, entry_stop=start)
        else:
            rows = rows.astype(np.int64, copy=False)
            order = None
            if np.any(rows[1:] <= rows[:-1]):
                order = np.argsort(rows, kind='stable')
                rows = rows[order]
                if np.any(rows[1:] == rows[:-1]):
                    raise ValueError('Duplicate gathered position for corrected nominal friend')
            if project is None:
                from .private_scales import project_entries
                a = project_entries(friend_tree, fields, rows)
            else:
                a = project(friend_tree, rows, fields)
            if order is not None:
                a = a[np.argsort(order)]
    keys = _identity(original)
    if not _same_identity(_identity(a), keys):
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
    use = ('candidateIndex', 'sourceJetIdx0', 'sourceJetIdx1', 'sourceJetIdx2', 'QCDDiscriminant', 'eta', 'mass')
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    columns = tuple(np.asarray(ak.flatten(a[prefix + k], axis=1)) for k in use)
    return updated, ResolvedFriendRows(keys, offsets, columns)

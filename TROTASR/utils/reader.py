"""Bounded Events reads; Resolved and truth joins stay inside the input ROOT."""
import numpy as np
import awkward as ak
from .paths import tagging_definition


def project_candidate_entries(tree, fields, entries):
    """Read sorted candidate rows; use one bounded bulk read for dense MC slices."""
    entries = np.asarray(entries)
    if (entries.ndim == 1 and entries.dtype.kind in 'iu' and len(entries) > 1
            and 0 <= entries[0] <= entries[-1] < tree.num_entries
            and np.all(entries[1:] > entries[:-1])):
        start, stop = int(entries[0]), int(entries[-1]) + 1
        if len(entries) < stop - start <= 2 * len(entries):
            block = tree.arrays(fields, entry_start=start, entry_stop=stop, library='ak')
            if len(block) != stop - start:
                raise RuntimeError('Incomplete dense ROOT projection')
            selected = block[entries - start]
            pack = getattr(ak, 'to_packed', getattr(ak, 'packed', None))
            return pack(selected) if pack else selected
    from .private_scales import project_entries
    return project_entries(tree, fields, entries)


def gated_event_chunks(reader, columns, step_size, gate_columns, gate, audit):
    """Project heavy columns after a necessary selection, retaining ROOT row IDs.

    `gate` is a caller-supplied union of all still-possible regions/endpoints.
    This reader does not know any region or apply any physics selection itself.
    """
    gate_columns = tuple(dict.fromkeys(gate_columns))
    other_columns = tuple(name for name in columns if name not in gate_columns)
    record = audit.setdefault('input_preselection', {})
    pack = getattr(ak, 'to_packed', getattr(ak, 'packed', None))
    for start in range(0, reader.num_entries, step_size):
        stop = min(start + step_size, reader.num_entries)
        first = reader.events.arrays(gate_columns, entry_start=start, entry_stop=stop)
        if len(first) != stop - start:
            raise ValueError('Incomplete Events preselection read')
        use = np.asarray(gate(first))
        if use.shape != (len(first),) or use.dtype.kind != 'b':
            raise ValueError('Events preselection must return one Boolean per input row')
        rows = start + np.flatnonzero(use)
        audit['events_read'] += len(first)
        record['scanned'] = record.get('scanned', 0) + len(first)
        record['projected'] = record.get('projected', 0) + len(rows)
        record['rejected_before_projection'] = record.get('rejected_before_projection', 0) + len(first) - len(rows)
        if not len(rows):
            continue
        rest = reader.project(other_columns, rows) if other_columns else None
        if rest is not None and len(rest) != len(rows):
            raise ValueError('Incomplete selected Events projection')
        if len(rows) == len(first):
            thin = {name: first[name] for name in gate_columns if name in columns}
        else:
            thin = {name: (pack(first[name][use]) if pack else first[name][use])
                    for name in gate_columns if name in columns}
        fields = {name: thin[name] if name in thin else rest[name] for name in columns}
        if rest is not None:
            fields.update({name: rest[name] for name in ak.fields(rest) if name not in fields})
        selected = ak.zip(fields, depth_limit=1)
        del first, fields, rest, thin
        yield selected, start, stop, rows


class StoredResolvedProjection:
    """Index only the two TROTA identity columns; load selected candidates per chunk.

    Events and the flat (possibly unsorted) candidate tree have different row
    numbers. Join by (file_id, entry), keeping original candidate ordering even
    for mixed files or a sparse, out-of-order sequence of Events identities.
    """

    def __init__(self, tree, year):
        self.tree = tree
        self.wp = tagging_definition(year)['resolved_wp']
        keys = tree.arrays(['file_id', 'entry'], library='np')
        files, entries = (np.asarray(keys[k]) for k in ('file_id', 'entry'))
        if any(c.dtype.kind not in 'iu' or c.shape != (tree.num_entries,) for c in (files, entries)):
            raise ValueError('Invalid stored Resolved candidate identities')
        self.order = np.lexsort((entries, files))
        self.keys = np.rec.fromarrays((files[self.order], entries[self.order]), names='file_id,entry')

    def for_events(self, a):
        from .private_scales import project_entries
        from .trota_friend import ResolvedFriendRows
        event_keys = tuple(np.asarray(a[k]) for k in ('file_id', 'entry'))
        needle = np.empty(len(a), dtype=self.keys.dtype)
        for name, vals in zip(needle.dtype.names, event_keys):
            bounds = np.iinfo(needle.dtype[name])
            if (vals.ndim != 1 or vals.dtype.kind not in 'iu' or (vals.size and
                    (int(vals.min()) < bounds.min or int(vals.max()) > bounds.max))):
                raise ValueError('Invalid Events/Resolved candidate join key')
            needle[name] = vals
        first, last = (self.keys.searchsorted(needle, side=side) for side in ('left', 'right'))
        counts = last - first
        offsets = np.r_[0, np.cumsum(counts, dtype=np.intp)]
        groups = np.repeat(np.arange(len(a), dtype=np.intp), counts)
        rows = self.order[np.arange(len(groups)) + np.repeat(first - offsets[:-1], counts)]
        # Project in physical ROOT row order, then restore Events order for the
        # aligned topology consumer. Only selected candidate fields are read.
        names = ['TopResolved1pct_' + n for n in ('candidateIndex', 'sourceJetIdx0',
                 'sourceJetIdx1', 'sourceJetIdx2', 'QCDDiscriminant', 'eta', 'mass')]
        if len(rows):
            physical, take = np.unique(rows, return_inverse=True)
            block = project_entries(self.tree, names, physical)
            columns = tuple(np.asarray(block[k])[take] for k in names)
            idx, j0, j1, j2, score, _, _ = columns
            if (np.any((j0 < 0) | (j1 < 0) | (j2 < 0) | (j0 == j1) | (j1 == j2) | (j0 == j2))
                    or np.any(~np.isfinite(score) | (score < self.wp))):
                raise ValueError('Invalid stored Resolved candidate')
            order = np.lexsort((idx, groups))
            if np.any((groups[order[1:]] == groups[order[:-1]]) & (idx[order[1:]] == idx[order[:-1]])):
                raise ValueError('Duplicate Resolved candidate identity')
        else:
            columns = (np.empty(0, dtype=np.intp),) * 4 + (np.empty(0),) * 3
        return ResolvedFriendRows(event_keys, offsets, columns)


def resolved_candidates(root, year=2024, *, project=None, rows=None):
    wp = tagging_definition(year)['resolved_wp']
    tree = root['TROTA']
    if rows is None:
        columns = tree.arrays(library='np')
    else:
        if not len(rows):
            return {}
        if project is None:
            raise ValueError('Selected Resolved rows require a branch projector')
        names = ['file_id', 'entry', 'TopResolved1pct_candidateIndex',
                 'TopResolved1pct_QCDDiscriminant', 'TopResolved1pct_eta', 'TopResolved1pct_mass',
                 *(f'TopResolved1pct_sourceJetIdx{i}' for i in range(3))]
        columns = project(tree, rows, names)
    records = {}
    seen = set()
    for i in range(len(columns['entry'])):
        key = (int(columns['file_id'][i]), int(columns['entry'][i]))
        index = int(columns['TopResolved1pct_candidateIndex'][i])
        if (*key, index) in seen:
            raise ValueError('Duplicate Resolved candidate identity')
        seen.add((*key, index))
        jets = [int(columns['TopResolved1pct_sourceJetIdx' + str(k)][i]) for k in range(3)]
        score = float(columns['TopResolved1pct_QCDDiscriminant'][i])
        if min(jets) < 0 or len(set(jets)) != 3 or not np.isfinite(score) or score < wp:
            raise ValueError('Invalid stored Resolved candidate')
        records.setdefault(key, []).append(dict(index=index, jets=jets, score=score,
                    eta=float(columns['TopResolved1pct_eta'][i]), mass=float(columns['TopResolved1pct_mass'][i])))
    return records

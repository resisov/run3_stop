"""Bounded Events reads; Resolved and truth joins stay inside the input ROOT."""
import numpy as np
from .event_selections import MIXED_WP


def resolved_candidates(root):
    tree = root['TROTA']
    columns = tree.arrays(library='np')
    records = {}
    seen = set()
    for i in range(tree.num_entries):
        key = (int(columns['file_id'][i]), int(columns['entry'][i]))
        index = int(columns['TopResolved1pct_candidateIndex'][i])
        if (*key, index) in seen:
            raise ValueError('Duplicate Resolved candidate identity')
        seen.add((*key, index))
        jets = [int(columns['TopResolved1pct_sourceJetIdx' + str(k)][i]) for k in range(3)]
        score = float(columns['TopResolved1pct_QCDDiscriminant'][i])
        if min(jets) < 0 or len(set(jets)) != 3 or not np.isfinite(score) or score < 0.9433798789978027:
            raise ValueError('Invalid stored Resolved candidate')
        records.setdefault(key, []).append(dict(index=index, jets=jets, score=score,
                    eta=float(columns['TopResolved1pct_eta'][i]), mass=float(columns['TopResolved1pct_mass'][i])))
    return records

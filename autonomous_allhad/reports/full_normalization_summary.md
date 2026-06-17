# Full Production Normalization

Status: `blocked`

This artifact is intentionally separate from feature-side subset normalization.

## Blockers

- full production feature table is not complete
- metadata does not contain full dataset signed sumw/Runs denominators
- correction weight products for full production are not available
- systematic shifted event weights are not available

## Exact unblock steps

- Run full production over the configured NanoAOD files or provide a production feature table with per-dataset signed sumw denominators.
- Add/derive full dataset Runs sumw metadata for every MC and signal dataset.
- Recompute full_normalization_factors.json and full_normalized_yields.json before final search-bin selection.

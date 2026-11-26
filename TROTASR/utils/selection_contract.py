"""Scientific selection constraints shared by canonical histogram, card and fit stages."""
from pathlib import Path

from . import paths as analysis_paths


RETIRED_LLCR_MESSAGE = ('The 175 GeV high-dM LLCR selection is retired. '
                        'Use the canonical LLCR without an mT,b cut.')


def require_current_llcr(*paths, record=None):
    """Do not use the cut-study products as inputs/outputs of current production."""
    retired = (analysis_paths.ROOT / 'stats/run3/new_signal/llcr_mtb').resolve()
    for path in paths:
        if path is not None:
            path = Path(path).resolve()
            if path == retired or retired in path.parents:
                raise ValueError(RETIRED_LLCR_MESSAGE)
    if record is not None:
        for fields in (record, record.get('contract') or {}, record.get('physics_contract') or {}):
            if fields.get('highdm_llcr_mtb_min_gev') is not None:
                raise ValueError(RETIRED_LLCR_MESSAGE)

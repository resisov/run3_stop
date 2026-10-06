"""Low-dM entry point, sharing the same bounded ROOT reader and selections."""
from pathlib import Path
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.workflows.build_highdm_histogram import main

if __name__ == '__main__':
    main('lowdm')

"""Export compact validation evidence, never claiming end-to-end completion."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.io import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = []
    for path in a.input:
        data = read_json(path)
        chunks = data['audit']['chunks']
        regions = {}
        for r in ('SR', 'LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M'):
            if any(c[r].get('blinded') for c in chunks):
                regions[r] = {'blinded': True}; continue
            regions[r] = {k: sum(c[r][k] for c in chunks)
                          for k in chunks[0][r] if isinstance(chunks[0][r][k], (int, float))}
        records.append(dict(input=data['contract']['input'], result_sha256=sha256(path),
                            status=data['status'], events=data['audit']['events_read'], seconds=data['seconds'],
                            regions=regions, migrations=data['audit'].get('migrations'),
                            versions=data['versions'], source_contract=data['contract']))
    write_json(a.output, dict(status='partial_pipeline_validation', production_adopted=False,
        comparison='Legacy B/W-before-Resolved and trial B>M>R>independent-W on identical non-topology cores; not a comparison of old merged SR binning',
        records=records, not_validated=['T2bW sensitivity', 'T2tt/T2tb acceptance over the full signal grid',
            '2025 representative input', 'NanoAOD-to-integrated-ROOT clean producer',
            'background estimation on complete data/MC coverage', 'object-shape propagation',
            'template/datacard/Combine end-to-end', 'final SR statistical bin map']))


if __name__ == '__main__': main()

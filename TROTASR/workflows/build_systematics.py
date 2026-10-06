"""Weight endpoints and optional full in-memory CMS-p4 TROTA/JME propagation."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.histogramming import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--year', type=int, required=True, choices=(2024, 2025))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variation', action='append', required=True,
                   help='Exact weight endpoint name, or all_weights; repeat to select endpoints')
    p.add_argument('--chunk-size', type=int, default=2000)
    p.add_argument('--cms-trota-jme', action='store_true', help='New CMS-p4 central plus all four re-inferred JES/JER endpoints')
    p.add_argument('--stored-trota-nonjme', action='store_true', help='All 16 existing non-JME endpoints, stored TROTA unchanged')
    p.add_argument('--object-endpoint', action='append',
                   help='Stored-TROTA endpoint (or nominal); repeat to evaluate a file-local group in one pass')
    p.add_argument('--candidate-budget', type=int, default=150000)
    a = p.parse_args()
    if a.cms_trota_jme and a.stored_trota_nonjme:
        p.error('Choose one shape convention; distinct centrals must not be mixed')
    if any(x.lower().startswith(('jes', 'jer', 'metuncl', 'electron_scale', 'muon_scale', 'photon_scale')) for x in a.variation):
        p.error('Object-shape adapter is not validated; no output was produced')
    result = build(a.input, a.year, a.output, variations=tuple(a.variation), chunk_size=a.chunk_size,
                   shape_mode='cms_trota_jme' if a.cms_trota_jme else 'stored_trota_nonjme' if a.stored_trota_nonjme else None,
                   candidate_budget=a.candidate_budget,
                   object_endpoint=a.object_endpoint[0] if a.object_endpoint and len(a.object_endpoint) == 1 else None,
                   object_endpoints=tuple(a.object_endpoint) if a.object_endpoint and len(a.object_endpoint) > 1 else None)
    print(result['status'], result['contract']['variations'])


if __name__ == '__main__':
    main()

"""Plot explicitly selected region(s) using the unchanged legacy renderer."""
from pathlib import Path
import argparse
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.plot_adapter import blocks, renderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('highdm', 'lowdm'), required=True)
    p.add_argument('--region', choices=('SR', 'LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M'), action='append', required=True)
    p.add_argument('--observable', choices=('recoil', 'gnn_score'), default='recoil')
    p.add_argument('--bin-map', type=Path,
                   help='Required for high-dM SR: the full fixed bin map, never a test-occupancy subset')
    a = p.parse_args()
    if a.mode == 'highdm' and 'SR' in a.region and a.bin_map is None:
        p.error('High-dM SR requires --bin-map; test inputs must not shrink the SR layout')
    a.output = internal_path(a.output)
    bin_map = read_json(a.bin_map) if a.bin_map is not None else None
    payload = read_json(a.input)
    expected = next(x['sha256'] for x in read_json(ROOT / 'jsons/sources.json')['files']
                    if x['target'] == 'utils/renderers/plot_control_search_bins_style.py')
    if sha256(Path(renderer.__file__)) != expected:
        raise ValueError('Legacy renderer was changed')
    if payload['status'] not in ('complete_one_file_test', 'complete'):
        raise ValueError('Incomplete histogram input')
    # Validate every requested layout before creating any plots/output directory.
    prepared = [(region, blocks(payload, a.mode, region, a.observable, bin_map=bin_map))
                for region in a.region]
    result = []
    for region, b in prepared:
        if not b:
            result.append(dict(region=region, status='no_selected_events')); continue
        outbase = a.output / (a.mode + '_' + region + '_' + a.observable)
        outbase.parent.mkdir(parents=True, exist_ok=True)
        info = renderer.draw_flat_blocks(b, outbase, xlabel='GNN bin' if a.observable == 'gnn_score' else 'Bin',
                         uncertainty_label_override='MC stat. unc.',
                         luminosity_fb={2024:109.82, 2025:110.84}[payload['contract']['year']])
        info.update(region=region, source_scope=payload['scope'], status='generated_pending_visual_QA')
        if a.mode == 'highdm' and region == 'SR':
            expected_bins = bin_map.get('bins_per_year', bin_map.get('total_bins'))
            if info['bins'] != expected_bins or info.get('dropped_input_bins_1based'):
                raise ValueError('Renderer did not preserve the full SR bin map')
            info.update(bin_map_sha256=sha256(a.bin_map), layout_scope='full', empty_bins_preserved=True)
        result.append(info)
    write_json(a.output / 'plot_manifest.json', dict(status='generated_pending_visual_QA', plots=result,
                renderer_sha256=expected, input_sha256=sha256(a.input)))


if __name__ == '__main__':
    main()

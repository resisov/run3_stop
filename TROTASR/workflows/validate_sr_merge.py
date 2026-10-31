"""Compare the re-exported merged native cards with the exact SR projection."""
import argparse
import numpy as np
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.sr_merge import load_merge, project_prediction


def verify(output):
    output = internal_path(output)
    payload_path = output/'plots/predictions/prediction_payload.json'
    payload = read_json(payload_path)
    config = load_merge(payload['sr_merge'])
    source_path = ROOT/config['decision_payload']
    if sha256(source_path) != config['decision_payload_sha256']:
        raise ValueError('Changed decision prediction')
    source = read_json(source_path)
    plot_manifest = read_json(output/'plots/predictions/plot_manifest.json')
    if plot_manifest['payload_sha256'] != sha256(payload_path):
        raise ValueError('Changed plot export')
    for name, digest in plot_manifest['files'].items():
        if sha256(payload_path.parent/name) != digest:
            raise ValueError('Changed plot file')
    if set(payload['channels']) != set(source['channels']):
        raise ValueError('Changed prediction channel set')
    checked = 0
    totals = {}
    for name, previous in source['channels'].items():
        expected = project_prediction(previous, config) if name.endswith('/SR/highdm') else previous
        actual = payload['channels'][name]
        if actual['labels'] != expected['labels']:
            raise ValueError('Changed native label order')
        if '/SR/' in name and (not actual['sr_data_blinded'] or 'data' in actual):
            raise ValueError('Observed SR data exposed')
        for field in ('background', 'variance'):
            np.testing.assert_allclose(actual[field], expected[field], rtol=2e-10, atol=1e-10)
            checked += 1
        for field in ('groups', 'group_variances', 'signals', 'nuisance_deltas'):
            if set(actual[field]) != set(expected[field]):
                raise ValueError('Changed process/signal/nuisance set')
            for key in expected[field]:
                if field == 'nuisance_deltas':
                    for direction in ('up', 'down'):
                        np.testing.assert_allclose(actual[field][key][direction], expected[field][key][direction],
                                                   rtol=2e-10, atol=1e-10)
                        checked += 1
                else:
                    np.testing.assert_allclose(actual[field][key], expected[field][key], rtol=2e-10, atol=1e-10)
                    checked += 1
        totals[name] = dict(before=float(np.sum(previous['background'])),
                            after=float(np.sum(actual['background'])))
    combined = sum(np.asarray(payload['channels'][year+'/SR/highdm']['background']) for year in ('2024','2025'))
    categories = []
    offset = 0
    for index, category in enumerate(config['categories'], 1):
        count = len(category['edges'])-1
        total = float(combined[offset:offset+count].sum())
        np.testing.assert_allclose(total, category['background_total'], rtol=2e-10, atol=1e-10)
        if total < 2:
            raise ValueError('Merged category still below threshold')
        categories.append(dict(category=index, source_categories=category['source_categories_1based'],
                               bins=count, background_total=total))
        offset += count
    audit = dict(status='passed', prediction_sha256=sha256(payload_path),
        source_prediction_sha256=sha256(source_path), sr_merge=payload['sr_merge'],
        compared_arrays=checked, category_count=len(categories), highdm_bins=len(combined),
        lowdm_bins=30, threshold_scope=config['threshold_scope'], minimum_category_background=min(
            c['background_total'] for c in categories), categories=categories, background_totals=totals,
        all_reexported_background_signal_variance_and_nuisance_arrays_match=True,
        visual_QA_is_separate=True, full_workflow_complete=False)
    write_json(output/'grid/prediction_merge_audit.json', audit)
    print('passed:', checked, 'arrays,', len(categories), 'categories, minimum B =', audit['minimum_category_background'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    verify(parser.parse_args().output)

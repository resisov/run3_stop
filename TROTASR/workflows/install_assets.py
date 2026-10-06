"""Explicit one-time migration of immutable calibration/model inputs.

The source bundle must already be staged inside TROTASR after explicit approval.
Existing divergent targets are rejected, not overwritten.
"""
from pathlib import Path
import argparse
import shutil
import json
import gzip
import math
import hashlib
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, POG_NAMES, internal_path
from TROTASR.utils.io import read_json, write_json, sha256


EARLY_ERAS = ('2022', '2022EE', '2023', '2023BPix')
EARLY_FILES = {
    'BTV': ('btagging', 'ctagging'),
    'EGM': ('electron', 'photon', 'electronHlt', 'electronID_highPt',
            'photonID_highPt', 'electronSS_EtDependent', 'photonSS_EtDependent'),
    'JME': ('jet_jerc', 'fatJet_jerc', 'jetid', 'jetvetomaps'),
    'LUM': ('puWeights',),
    'MUO': ('muon_Z', 'muon_HighPt', 'muon_JPsi', 'muon_scalesmearing',
            'muon_scalesmearing_VXBS'),
    'TAU': ('tau',),
}


def install_early_eras(bundle, catalog, eras, check_only=False):
    """Deploy explicit official payloads without changing live 2024/25 assets.

    The downloaded catalog and files must already be inside TROTASR. Catalog
    source paths are provenance only: this function never opens CVMFS/EOS or
    imports another analysis repository. Installing a payload is not evidence
    that its corresponding correction has been applied to any histogram.
    """
    bundle, catalog = internal_path(bundle), internal_path(catalog)
    eras = tuple(eras)
    if not eras or len(set(eras)) != len(eras) or not set(eras) <= set(EARLY_ERAS):
        raise ValueError('Choose unique earlier Run-3 eras explicitly')
    metadata = read_json(catalog)
    if metadata.get('schema') != 'trotasr_run3_early_assets_v1':
        raise ValueError('Unknown early Run-3 catalog schema')
    source_root = Path(metadata['catalog_source'])
    expected = set()
    for era in eras:
        for pog, names in EARLY_FILES.items():
            names = (*names, f'met_xyCorrections_{era[:4]}_{era}') if pog == 'JME' else names
            expected.update(f'scales/{pog}/{era}/{name}.json.gz' for name in names)
    records = [r for r in metadata['files'] if r['era'] in eras]
    targets = [r['target'] for r in records]
    if len(set(targets)) != len(targets) or set(targets) != expected:
        raise ValueError('Incomplete, duplicate or unexpected official payload set')
    operations = []
    for record in records:
        relative = Path(record['source']).relative_to(source_root)
        source = internal_path(bundle / relative)
        if source != bundle and bundle not in source.parents:
            raise ValueError('Catalog source escapes the staged bundle')
        target = internal_path(ROOT / record['target'])
        if target != ROOT / record['target']:
            raise ValueError('Payload target must not be redirected by a symlink')
        if sha256(source) != record['sha256']:
            raise ValueError('Staged official payload changed: ' + str(source))
        with gzip.open(source, 'rt') as stream:
            payload = json.load(stream)
        if payload.get('schema_version') != 2 or not payload.get('corrections'):
            raise ValueError('Invalid correctionlib payload: ' + str(source))
        if target.name == 'btagging.json.gz':
            keys = {c['name'] for c in payload['corrections']}
            if not {'particleNet_comb', 'particleNet_light'} <= keys:
                raise ValueError('Required PNet AK4 b-tag corrections are absent')
        if target.exists() and sha256(target) != record['sha256']:
            raise ValueError('Divergent existing payload: ' + str(target))
        operations.append((source, target))
    registry = ROOT / 'jsons/early_run3_assets.json'
    result = dict(metadata, files=records, eras=list(eras),
                  purpose='Official correction inputs for the earlier Run-3 extension',
                  installation_only=True, histogram_application_verified=False,
                  nanoaod_version=12, ak4_btag='PNet', ak8_topw='PNetWithMass',
                  met_xy_required=True, trota_model_year=2022,
                  signal_campaign={'2022': '2022', '2022EE': '2022EE',
                                   '2023': '2022', '2023BPix': '2022'},
                  private_calibrations='Use per-era measured efficiencies/trigger SFs; not replaced by POG files',
                  live_2024_2025_asset_registry_unchanged=True)
    if registry.exists():
        installed = read_json(registry)
        if {k: v for k, v in installed.items() if k != 'analysis_inputs'} != result:
            raise ValueError('Existing early Run-3 registry differs; explicit update required')
    if not check_only:
        # Every source and existing target has passed before the first write.
        for source, target in operations:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != sha256(source):
                raise ValueError('Payload copy failed: ' + str(target))
        if not registry.exists():
            write_json(registry, result)
    print(('Checked' if check_only else 'Installed and verified'), len(records),
          'official payloads for', ', '.join(eras))
    return result


def topw_reference_payload(reference, era, source_sha256):
    """Serialize the already approved PNetWithMass cells, without a new fit."""
    aliases = dict(zip(EARLY_ERAS, ('2022pre', '2022post', '2023pre', '2023post')))
    if era not in aliases:
        raise ValueError('Unknown early Run-3 era')
    source = reference['eras'][aliases[era]]
    corrections = []
    provenance = {}
    for tag, edges in (('top', [300, 400, 480, 600, 1200]), ('w', [200, 300, 400, 800])):
        measurement = source[tag]
        rows = measurement['scale_factors']
        pairs = list(zip(edges[:-1], edges[1:]))
        expected = {(cat, pair) for cat in ('tp1', 'tp2', 'tp3', 'other') for pair in pairs}
        actual = [(r['category'], tuple(r['pt_gev'])) for r in rows]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError('Incomplete or duplicated Top/W measurement cells')
        for row in rows:
            sf = row['adopted_sf']
            low, mid, high = [float(sf[k]) for k in ('down', 'nominal', 'up')]
            if not all(math.isfinite(x) for x in (low, mid, high)) or not 0 <= low <= mid <= high:
                raise ValueError('Malformed Top/W measured endpoint')
            if row['explicit_fit_failure'] and (low, mid, high) != (1., 1., 1.):
                raise ValueError('Failed Top/W cell violates the user-approved unity policy')
        for cat in ('tp1', 'tp2', 'tp3', 'other'):
            selected = {tuple(r['pt_gev']): r for r in rows if r['category'] == cat}
            corrections.append(dict(
                name=f'topw_{tag}_{cat}_sf', version=1,
                description=f'PNetWithMass 1% WP, {era}, {tag}, {cat}; source {source_sha256}',
                inputs=[dict(name='variation', type='string'), dict(name='pt', type='real')],
                output=dict(name='weight', type='real'),
                data=dict(nodetype='category', input='variation', content=[
                    dict(key=v, value=dict(nodetype='multibinning', inputs=['pt'],
                         edges=[list(map(float, edges))],
                         content=[float(selected[p]['adopted_sf'][v]) for p in pairs], flow='clamp'))
                    for v in ('nominal', 'up', 'down')])))
        provenance[tag] = {key: measurement[key] for key in
                           ('fit_url', 'fit_sha256', 'global_config_url') if key in measurement}
    return dict(schema_version=2, corrections=corrections, compound_corrections=[],
                description=json.dumps(dict(era=era, tagger='PNetWithMass', source_sha256=source_sha256,
                    published_measurements=provenance, failed_fit_policy='unity only in explicit failed cells',
                    efficiency_application='Use each era measured MC efficiency; bound SF*eff at one'), sort_keys=True))


def install_topw_references(bundle, eras):
    """Convert existing approved references into the canonical Private payloads."""
    bundle = internal_path(bundle)
    records, prepared = [], []
    for era in eras:
        if era not in EARLY_ERAS:
            raise ValueError('Unknown early Run-3 era')
        source = internal_path(bundle / ('flat' + era) / 'topw_reference.json')
        payload = topw_reference_payload(read_json(source), era, sha256(source))
        encoded = (json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        import correctionlib
        parsed = correctionlib.CorrectionSet.from_string(encoded.decode())
        for item in payload['corrections']:
            for endpoint in item['data']['content']:
                values = endpoint['value']
                edges = values['edges'][0]
                for lo, hi, expected in zip(edges[:-1], edges[1:], values['content']):
                    if not math.isclose(parsed[item['name']].evaluate(endpoint['key'], float((lo + hi) / 2)),
                                        expected, rel_tol=1.e-7, abs_tol=1.e-7):
                        raise ValueError('Top/W correctionlib readback changed a measured value')
        # mtime=0 and no filename make deployment byte-identical on both hosts.
        import io
        compressed = io.BytesIO()
        with gzip.GzipFile(filename='', fileobj=compressed, mode='wb', mtime=0) as stream:
            stream.write(encoded)
        data = compressed.getvalue()
        target = internal_path(ROOT / 'scales/Private' / era / 'topw_tagging_sf.json.gz')
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Divergent existing Top/W payload: ' + str(target))
        prepared.append((target, data))
        records.append(dict(era=era, source_sha256=sha256(source), target=str(target.relative_to(ROOT)),
                            sha256=hashlib.sha256(data).hexdigest()))
    registry = ROOT / 'jsons/early_run3_assets.json'
    registered = read_json(registry)
    inputs = dict(registered.get('analysis_inputs', {}))
    for record in records:
        if record['target'] in inputs and inputs[record['target']] != record:
            raise ValueError('Existing Top/W provenance differs')
        inputs[record['target']] = record
    for target, data in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    if inputs != registered.get('analysis_inputs'):
        write_json(registry, dict(registered, analysis_inputs=inputs))
    print('Installed and verified', len(records), 'approved PNetWithMass Top/W payloads')
    return records


def install(bundle):
    bundle = internal_path(bundle)
    records = []
    def copy(source, target):
        source, target = internal_path(source), internal_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError("Divergent existing payload: " + str(target))
        if not target.exists():
            shutil.copyfile(source, target)
        records.append(dict(source=str(source), target=str(target.relative_to(ROOT)),
                            sha256=sha256(target), source_sha256=sha256(source)))
    for year in (2024, 2025):
        for old, new in POG_NAMES.items():
            source = internal_path(bundle / 'main/analysis/data' / old / str(year))
            for p in sorted(source.glob('*.json.gz')):
                copy(p, ROOT / 'scales' / new / str(year) / p.name)
        for kind in ('btag', 'topwtag'):
            name = kind + 'eff' + str(year) + '.merged'
            copy(bundle / 'main/analysis/hists' / name, ROOT / 'estimations' / name)
        source = bundle / 'inputs' / str(year) / 'norm.json'
        target = ROOT / 'estimations' / ('normalization_' + str(year) + '.json.gz')
        data = read_json(source)
        if target.exists() and read_json(target) != data:
            raise ValueError('Divergent normalization; explicit validated migration required')
        if not target.exists():
            write_json(target, data)
        records.append(dict(source=str(source), target=str(target.relative_to(ROOT)),
                            source_sha256=sha256(source), sha256=sha256(target),
                            transform='lossless JSON gzip; factors unchanged'))
    model = bundle / 'gnn/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010'
    for name in ('diagonal_v3_numpy.npz', 'selection.json'):
        copy(model / name, ROOT / 'gnn4lowdm' / name)
    copy(bundle / 'gnn/autonomous_allhad/signals/stop_xsec_13p6TeV.json',
         ROOT / 'stats/stop_xsec_13p6TeV.json')
    write_json(ROOT / 'jsons/assets.json', dict(schema='trotasr_assets_v1', files=records))
    print('Installed and verified', len(records), 'immutable assets')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-bundle', required=True, type=Path)
    p.add_argument('--prepare-eras', nargs='+', choices=EARLY_ERAS,
                   help='Add official payloads for these eras; never alters live 2024/25 assets')
    p.add_argument('--catalog', type=Path, help='Explicit official catalog staged inside TROTASR')
    p.add_argument('--topw-references', action='store_true',
                   help='Convert approved per-era Top/W references from --source-bundle')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.prepare_eras:
        if args.topw_references:
            if args.catalog or args.check_only:
                p.error('--topw-references does not accept --catalog or --check-only')
            install_topw_references(args.source_bundle, args.prepare_eras)
        elif args.catalog is None:
            p.error('--prepare-eras requires --catalog')
        else:
            install_early_eras(args.source_bundle, args.catalog, args.prepare_eras, args.check_only)
    else:
        if args.catalog or args.check_only or args.topw_references:
            p.error('--catalog and --check-only require --prepare-eras')
        install(args.source_bundle)

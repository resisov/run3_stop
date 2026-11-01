"""Import calibration and model inputs from an in-package staged bundle.

Existing divergent targets are rejected rather than overwritten.
"""
from pathlib import Path
import argparse
import shutil
import json
import gzip
import math
import re
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
        if {k: v for k, v in installed.items()
                if k not in ('analysis_inputs', 'background_cross_sections')} != result:
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
    """Serialize existing PNetWithMass measurement cells without refitting."""
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
    """Convert existing reference measurements into Private correction payloads."""
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


def trigger_measurement_payload(result, era, measurement):
    """Same measured-bin SF +/- total error serialization as measure_trigger."""
    if era not in EARLY_ERAS or measurement not in ('photon', 'met_genuine'):
        raise ValueError('Unknown trigger measurement era/type')
    if (result.get('measurement_type') != measurement
            or str(result.get('year') or result.get('measurement', '').rsplit('_', 1)[-1]) != era):
        raise ValueError('Trigger measurement does not belong to the requested era')
    if result.get('files_failed') or result.get('incomplete_files') or result.get('adoption_blockers'):
        raise ValueError('Trigger measurement has unresolved input failures')
    bins = result['bins']
    if not bins or any(not b.get('valid') for b in bins):
        raise ValueError('Trigger measurement contains invalid bins')
    axes = ([('met', result['bin_edges_gev'])] if measurement == 'met_genuine'
            else [('abseta', result['abseta_edges']), ('pt', result['pt_edges_gev'])])
    size = math.prod(len(edges)-1 for _, edges in axes)
    if size != len(bins) or any(not all(a < b for a, b in zip(e, e[1:])) for _, e in axes):
        raise ValueError('Trigger measurement binning is malformed')
    nominal = [float(b['scale_factor']) for b in bins]
    error = [float(b['scale_factor_uncertainty']) for b in bins]
    if any(not math.isfinite(v) or v <= 0 for v in nominal) or any(
            not math.isfinite(v) or v < 0 for v in error):
        raise ValueError('Invalid measured trigger SF/error')
    values = dict(nominal=nominal, up=[v+e for v, e in zip(nominal, error)],
                  down=[max(1.e-6, v-e) for v, e in zip(nominal, error)])
    name = 'met_trigger_sf_genuine' if measurement == 'met_genuine' else 'photon_trigger_sf'
    return dict(schema_version=2, description=f'{era} measured {measurement} trigger SF',
                compound_corrections=[], corrections=[dict(name=name, version=1,
        inputs=[dict(name='variation', type='string'),
                *[dict(name=n, type='real') for n, _ in axes]],
        output=dict(name='weight', type='real'),
        data=dict(nodetype='category', input='variation', content=[
            dict(key=v, value=dict(nodetype='multibinning', inputs=[n for n, _ in axes],
                 edges=[list(map(float, e)) for _, e in axes], content=content, flow='clamp'))
            for v, content in values.items()]))])


def install_era_measurements(source_root, eras):
    """Explicit read-only import of delivered measurements, not event inputs.

    --source-bundle is the delivered workflow directory containing flat<era>.
    Only the two known efficiency files and two trigger results per era are
    read. No ROOT, worker, source NanoAOD or frozen measurement is modified.
    """
    import io
    import correctionlib
    source_root = Path(source_root).resolve()
    prepared, records = [], []
    for era in eras:
        if era not in EARLY_ERAS:
            raise ValueError('Unknown early era')
        folder = source_root / ('flat' + era)
        for kind in ('btag', 'topwtag'):
            name = f'{kind}eff{era}.merged'
            source = folder / 'efficiencies' / name
            data = source.read_bytes()
            target = internal_path(ROOT / 'estimations' / name)
            prepared.append((target, data))
            records.append(dict(era=era, source=str(source), source_sha256=hashlib.sha256(data).hexdigest(),
                                target=str(target.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest()))
        for measurement, filename in (('photon', 'photon_trigger_sf.json.gz'),
                                      ('met_genuine', 'met_trigger_sf.json.gz')):
            directory = folder / 'trigger_measurements' / measurement
            if era in ('2022', '2022EE'):
                directory /= 'retry_eos_namespace_20260915'
            source = directory / 'result.json'
            raw = source.read_bytes()
            result = json.loads(raw)
            payload = trigger_measurement_payload(result, era, measurement)
            encoded = (json.dumps(payload, allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n').encode()
            correctionlib.CorrectionSet.from_string(encoded.decode())
            compressed = io.BytesIO()
            with gzip.GzipFile(filename='', fileobj=compressed, mode='wb', mtime=0) as stream:
                stream.write(encoded)
            data = compressed.getvalue()
            target = internal_path(ROOT / 'scales/Private' / era / filename)
            prepared.append((target, data))
            records.append(dict(era=era, source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                source_status=result.get('status'), target=str(target.relative_to(ROOT)),
                sha256=hashlib.sha256(data).hexdigest(),
                adoption='User requested per-era connection on 2026-10-20; measured bins unchanged'))
    registry = ROOT / 'jsons/early_run3_assets.json'
    registered = read_json(registry)
    inputs = dict(registered.get('analysis_inputs', {}))
    for target, data in prepared:
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Divergent existing measurement: ' + str(target))
    for record in records:
        old = inputs.get(record['target'])
        if old is not None and old != record:
            raise ValueError('Different measurement provenance: ' + record['target'])
        inputs[record['target']] = record
    for target, data in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    if inputs != registered.get('analysis_inputs'):
        write_json(registry, dict(registered, analysis_inputs=inputs))
    print('Installed', len(prepared), 'per-era efficiency/trigger inputs')
    return records


def primary_dataset(dataset: str) -> str:
    """Drop production/reconstruction labels, not process, generator or cuts."""
    name = str(dataset)
    if name.startswith("/"):
        parts = name.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError("Malformed DAS dataset: " + name)
        return parts[0]
    name = re.sub(r"____\d+_$", "", name)
    return re.split(r"-(?:RunIII|Run3)", name, maxsplit=1)[0]


def install_signal_efficiencies(source_root, eras):
    """Install only a completed, additive v15 signal-count supplement."""
    import numpy as np
    from TROTASR.utils.io import load_efficiency, decode_efficiency
    source_root = Path(source_root).resolve()
    receipt = read_json(source_root / 'state.json')
    manifest = read_json(source_root / 'input_manifest.json')
    if (receipt.get('status') != 'complete' or receipt.get('failed')
            or receipt.get('files') != len(manifest['records'])
            or set(eras) != set(manifest['eras'])):
        raise ValueError('Signal efficiency production is not complete')
    expected = {r['dataset'] for r in manifest['records']}
    registry_path = ROOT / 'jsons/early_run3_assets.json'
    registry = read_json(registry_path)
    inputs = dict(registry.get('analysis_inputs', {}))
    operations = []
    for era in eras:
        if era not in EARLY_ERAS:
            raise ValueError('Only the explicit early-era signal supplement is supported')
        for kind in ('btag', 'topwtag'):
            name = f'{kind}eff{era}.merged'
            relative = f'merged/flat{era}/efficiencies/{name}'
            source = source_root / relative
            proof = receipt['outputs'][relative]
            if sha256(source) != proof['sha256']:
                raise ValueError('Changed signal efficiency output')
            target = internal_path(ROOT / 'estimations' / name)
            key = str(target.relative_to(ROOT))
            if sha256(target) == proof['sha256']:
                continue
            if sha256(target) != proof['baseline_sha256']:
                raise ValueError('Unrecognized efficiency baseline: ' + key)
            # Numeric proof that every old dataset, bin and variance is retained.
            old, new = load_efficiency(target), decode_efficiency(source.read_bytes())
            if set(old) != set(new):
                raise ValueError('Changed tagger inventory')
            for tagger in old:
                if set(new[tagger]) - set(old[tagger]) != expected:
                    raise ValueError('Supplement is not exactly the registered v15 datasets')
                for dataset, hist in old[tagger].items():
                    other = new[tagger][dataset]
                    if hist.axes != other.axes or not np.array_equal(hist.values(flow=True), other.values(flow=True)):
                        raise ValueError('Changed existing efficiency counts')
                    a, b = hist.variances(flow=True), other.variances(flow=True)
                    if (a is None) != (b is None) or a is not None and not np.array_equal(a, b):
                        raise ValueError('Changed existing efficiency variances')
            # Do not revise inputs of a frozen early-era histogram campaign.
            for plan_path in (ROOT / 'campaigns').glob('*/plan.json'):
                if era in read_json(plan_path).get('contracts', {}):
                    raise ValueError('Efficiency inputs already frozen: ' + str(plan_path))
            record = dict(era=era, source=str(source), source_sha256=proof['sha256'],
                target=key, sha256=proof['sha256'], baseline_sha256=proof['baseline_sha256'],
                signal_datasets=len(expected), measurement_manifest_sha256=sha256(source_root/'input_manifest.json'),
                completion_sha256=sha256(source_root/'state.json'),
                adoption='2026-10-22 nominal continuation: full-source v15 signal counts; prior counts unchanged')
            operations.append((source, target, record))
    for source, target, record in operations:
        history = ROOT / 'estimations/history' / (target.name + '.' + record['baseline_sha256'])
        history.parent.mkdir(parents=True, exist_ok=True)
        if not history.exists():
            shutil.copy2(target, history)
        if sha256(history) != record['baseline_sha256']:
            raise ValueError('Baseline backup differs')
        previous = inputs.get(record['target'])
        if previous:
            record['previous_provenance'] = previous
        shutil.copy2(source, target)
        if sha256(target) != record['sha256']:
            raise ValueError('Installed efficiency checksum mismatch')
        inputs[record['target']] = record
    if operations:
        write_json(registry_path, dict(registry, analysis_inputs=inputs))
    print('Installed', len(operations), 'additive signal efficiency files')
    return len(operations)


def cross_section_sample_key(dataset: str) -> str:
    """Normalize only the explicit W/Z parton-bin naming permutation."""
    name = primary_dataset(dataset)
    match = re.fullmatch(
        r"(WtoLNu|Zto2Nu)-2Jets_(PTLNu|PTNuNu)-(\d+(?:to\d+)?)_([12]J)_"
        r"(TuneCP5_13p6TeV_amcatnloFXFX-pythia8)", name)
    if match:
        process, variable, interval, jets, generator = match.groups()
        if (process, variable) in (('WtoLNu', 'PTLNu'), ('Zto2Nu', 'PTNuNu')):
            return f'{process}-2Jets_Bin-{jets}-{variable}-{interval}_{generator}'
    return name


def background_xsec_index(references):
    """One adopted cross section per primary sample, independent of target era.

    References contain existing analysis normalization records. Neither their
    luminosity nor their generator denominators are copied to another campaign.
    """
    index = {}
    for source, payload in references:
        for record in payload["datasets"].values():
            if record.get("is_data") or record.get("is_signal"):
                continue
            dataset = record.get("physical_dataset") or record["dataset"]
            key = cross_section_sample_key(dataset)
            value = record.get("xsec_pb")
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                continue
            value = float(value)
            old = index.get(key)
            if old is not None and not math.isclose(old["xsec_pb"], value, rel_tol=1.e-7, abs_tol=0.):
                raise ValueError("Conflicting adopted cross sections for " + key)
            item = index.setdefault(key, dict(xsec_pb=value, sources=[]))
            if source not in item["sources"]:
                item["sources"].append(source)
    return index


def adopted_background_xsecs():
    """Reuse the existing common 13.6 TeV analysis values; no target-year key."""
    paths = [f"estimations/normalization_{year}.json.gz" for year in (2024, 2025)]
    references = [(path, read_json(ROOT/path)) for path in paths]
    registry = ROOT/'jsons/early_run3_assets.json'
    if registry.exists():
        extra = read_json(registry).get('background_cross_sections', {})
        records = {}
        for primary, record in extra.items():
            if (record.get('energy_tev') != 13.6 or
                    primary_dataset(record['DAS']) != primary or
                    not record.get('source_records') or
                    not math.isfinite(float(record['xsec_pb'])) or record['xsec_pb'] <= 0):
                raise ValueError('Malformed registered cross section: ' + primary)
            records[primary] = dict(record, dataset=primary, is_data=False, is_signal=False)
        references.append((str(registry.relative_to(ROOT)), dict(datasets=records)))
    return background_xsec_index(references)


def apply_background_xsecs(datasets, index):
    """Connect matching samples in an in-memory normalization aggregation."""
    missing = []
    for dsid, record in datasets.items():
        if record.get("is_data") or record.get("is_signal"):
            continue
        primary = primary_dataset(record.get("physical_dataset") or record["dataset"])
        key = cross_section_sample_key(primary)
        reference = index.get(key)
        if reference is None:
            missing.append(dict(dataset_id=str(dsid), primary_dataset=primary))
            continue
        previous = record.get("xsec_pb")
        value = reference["xsec_pb"]
        if previous is not None and not math.isclose(float(previous), value, rel_tol=1.e-7, abs_tol=0.):
            raise ValueError("Existing cross section differs from adopted value: " + key)
        record["xsec_pb"] = value if previous is None else previous
        record["xsec_source"] = dict(primary_dataset=primary, matched_sample=key,
            references=list(reference["sources"]),
            mapping=("exact_primary_dataset" if key == primary else "WZ_parton_bin_name_permutation")
                    + "; independent_of_analysis_era")
    return missing


def build_era_physical(datasets, luminosity_pb):
    """Combine original/extension generator sums for each target-era sample.

    Input dataset identities and their full Runs sums remain unchanged. Each
    extension is additional MC statistics, not another copy of the cross section.
    """
    from copy import deepcopy
    from TROTASR.workflows.normalization_metadata_legacy import build_physical
    grouped = deepcopy(datasets)
    for record in grouped.values():
        if not record.get('is_data') and not record.get('is_signal'):
            record['physical_dataset'] = cross_section_sample_key(
                record.get('physical_dataset') or record['dataset'])
    return build_physical(grouped, luminosity_pb)


def install_early_lowdm(source, eras):
    """Install the explicitly selected 2022 export and frozen validation bins."""
    source = Path(source).resolve()
    if set(eras) != set(EARLY_ERAS):
        raise ValueError('The adopted early-era handoff covers all four target eras')
    load = lambda name: json.loads((source / name).read_text())
    parity, bins = load('numpy_parity.json'), load('sr/sr_binning.json')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected_checkpoint = '66a48926734124674e3c644cc0f7fc08948a4941a108af3544fa1df4f9ac6bf6'
    expected_bins = '38926dbe25880d309fc709c504e70460859a04f08dcec15f6b98c6325dacd685'
    if (parity['status'] != 'complete' or parity['mismatches_at_2e_6'] != 0
            or parity['provenance']['partition'] != 'validation'
            or parity['provenance']['checkpoint_sha256'] != expected_checkpoint
            or bins['provenance']['checkpoint_sha256'] != expected_checkpoint
            or digest(source/'sr/sr_binning.json') != expected_bins
            or digest(source/'diagonal_v3_numpy.npz') != parity['numpy_model_sha256']):
        raise ValueError('Model/binning/parity do not match the user-authorized handoff')
    from copy import deepcopy
    from TROTASR.gnn4lowdm.configuration import global_feature_names
    if tuple(parity['provenance']['global_feature_names']) != global_feature_names('2022'):
        raise ValueError('Portable training feature order differs')
    config = deepcopy(read_json(ROOT/'gnn4lowdm/config.json'))
    folder = ROOT/'gnn4lowdm/early_run3_2022'
    config.update(status='adopted_20261021', target_eras=list(eras),
        feature_names=list(global_feature_names('2022')),
        model=dict(id='2022pre_lr020', portable_numpy=str((folder/'diagonal_v3_numpy.npz').relative_to(ROOT)),
                   portable_numpy_sha256=parity['numpy_model_sha256'], checkpoint_sha256=expected_checkpoint,
                   selected_epoch=9, max_jets=10, jet_features=6, global_features=39,
                   hidden_dimension=48, message_passing_layers=3),
        provenance=dict(binning_sha256=expected_bins, parity_sha256=digest(source/'numpy_parity.json'),
                        trained_era='2022', other_eras='target-era feature/selection/inference validation required',
                        cr_score_edges='Existing adopted CR edges retained; no CR reoptimization',
                        user_authorization='2026-10-21 low-dM model/bin handoff and full analysis request'),
        years={era: {} for era in eras})
    config['sr_binning'] = dict(status='adopted_20261021', total_bins=30, bins_per_category=5,
        category_labels=list(config['sr_binning']['category_labels']),
        edges_by_category=bins['best_by_scheme']['nb2_nisr3']['edges_by_category'])
    operations = [(source/'diagonal_v3_numpy.npz', folder/'diagonal_v3_numpy.npz'),
                  (source/'numpy_parity.json', folder/'numpy_parity.json'),
                  (source/'sr/sr_binning.json', folder/'sr_binning.json')]
    for src, dst in operations:
        if dst.exists() and digest(dst) != digest(src):
            raise ValueError('Existing early-era model asset differs: '+str(dst))
    target = folder/'config.json'
    if target.exists() and read_json(target) != config:
        raise ValueError('Existing early-era configuration differs')
    folder.mkdir(parents=True, exist_ok=True)
    for src, dst in operations:
        if not dst.exists():
            shutil.copyfile(src, dst)
    if not target.exists():
        write_json(target, config)
    print('Installed 39-feature early-era model and frozen 30-bin boundaries; 2024/25 unchanged')


def install_era_normalization(source_root, era, luminosity_fb, check_only=False, refresh=False):
    """Build factors from delivered metadata using the common adopted xsecs.

    Source ROOT and metadata are read-only. Old v12 signal shards are excluded
    from the new v15 signal request. Coverage is the explicitly listed delivered
    files, not a claim that every production task has completed.
    """
    from copy import deepcopy
    from TROTASR.workflows.normalization_metadata_legacy import (
        merge_dataset, build_signal_mass_points, load_signal_xsec)
    if era not in EARLY_ERAS or not math.isfinite(luminosity_fb) or luminosity_fb <= 0:
        raise ValueError('Select one early era and its positive luminosity explicitly')
    folder = Path(source_root).resolve() / ('flat' + era) / 'outputs/nominal'
    sources = sorted(p for p in folder.glob('*.json')
                     if p.name.startswith(('mc_shard_', 'data_shard_', 'data_completion_', 'signal_v15_')))
    datasets, records = {}, []
    for path in sources:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (payload.get('campaign_year') != era or
                payload.get('status') not in ('complete', 'complete_with_bad_files')):
            raise ValueError('Metadata is not a completed target-era input: ' + str(path))
        if not path.with_suffix('.root').is_file():
            raise ValueError('Metadata has no delivered ROOT pair: ' + str(path))
        for dsid, rec in payload['datasets'].items():
            if rec.get('is_signal') and not path.name.startswith('signal_v15_'):
                raise ValueError('Old signal must not enter the v15 normalization')
            if str(dsid) not in datasets:
                datasets[str(dsid)] = deepcopy(rec)
            else:
                target = datasets[str(dsid)]
                for key in ('dataset', 'physical_dataset', 'is_data', 'is_signal', 'process'):
                    if target.get(key) != rec.get(key):
                        raise ValueError('Dataset identity collision: ' + str(dsid))
                merge_dataset(target, rec)
        records.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            status=payload['status'], files_processed=payload['files_processed'],
            files_attempted=payload['files_attempted'], events_written=payload['events_written'],
            dataset_ids=sorted(payload['datasets']),
            fingerprint=dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)))
    if not records:
        raise ValueError('No delivered metadata found: ' + str(folder))
    lumi_pb = luminosity_fb * 1000.
    apply_background_xsecs(datasets, adopted_background_xsecs())
    physical, split, factors = build_era_physical(datasets, lumi_pb)
    signal_path = ROOT/'stats/stop_xsec_13p6TeV.json'
    points = build_signal_mass_points(datasets, load_signal_xsec(signal_path), lumi_pb)
    blocked = {k:v for k,v in factors.items()
               if not v.get('is_signal') and v.get('normalization_factor') is None}
    blocked_signal = {k:v for k,v in points.items() if v.get('normalization_factor') is None}
    result = dict(schema_version='flat_ntuple_campaign_normalization_v1',
        status='incomplete' if blocked or blocked_signal else 'complete',
        year=era, source_sidecars=records, source_sidecar_count=len(records),
        input_coverage='listed_delivered_files_only; not_a_full_production_completion_claim',
        luminosity_pb=lumi_pb, luminosity_fb=luminosity_fb, luminosity_source='explicit_cli',
        normalization_policy=dict(background_formula='xsec_pb * luminosity_pb / target_era_primary_sample_sumw',
            extension_policy='pool original and extension Runs sums; preserve input DAS identities',
            signal_formula='stop_xsec_pb(mStop) * luminosity_pb / topology_mass_point_Runs_sumw',
            cross_section_mapping='common primary sample; explicit WZ bin-name permutation; independent of analysis era',
            source_root_and_metadata_unchanged=True),
        datasets=datasets, physical_datasets=physical, physical_dataset_split_counts=split,
        dataset_factors=factors, signal_mass_points=points,
        blocked_dataset_factors=blocked, blocked_signal_mass_points=blocked_signal,
        signal_xsec_source=dict(path=str(signal_path.relative_to(ROOT)), sha256=sha256(signal_path)),
        background_xsec_references={str(p.relative_to(ROOT)):sha256(p) for p in
            (ROOT/'estimations/normalization_2024.json.gz', ROOT/'estimations/normalization_2025.json.gz',
             ROOT/'jsons/early_run3_assets.json') if p.exists()})
    if not check_only:
        if result['status'] != 'complete':
            raise ValueError('Unmatched sample definitions remain; use --check-only to list them')
        target = ROOT/'estimations'/f'normalization_{era}.json.gz'
        if target.exists() and read_json(target) != result:
            if not refresh:
                raise ValueError('Existing normalization differs; keep active results unchanged')
            old_digest = sha256(target)
            for plan_path in (ROOT/'campaigns').glob('*/plan.json'):
                if era in read_json(plan_path).get('contracts', {}):
                    raise ValueError('Era already has a frozen campaign; normalization is immutable')
            previous = ROOT/'estimations/history'/('normalization_'+era+'_'+old_digest+'.json.gz')
            previous.parent.mkdir(parents=True, exist_ok=True)
            if not previous.exists():
                shutil.copyfile(target, previous)
            if sha256(previous) != old_digest:
                raise ValueError('Previous normalization backup differs')
            write_json(target, result)
        if not target.exists():
            write_json(target, result)
    print(json.dumps(dict(era=era, status=result['status'], files=len(records),
        mapped_backgrounds=sum(not v.get('is_data') and not v.get('is_signal') and
                               v.get('normalization_factor') is not None for v in factors.values()),
        unmatched_backgrounds=sorted({v['physical_dataset'] for v in blocked.values()}),
        signal_mass_points=len(points), blocked_signal_mass_points=sorted(blocked_signal),
        installed=not check_only)))
    return result


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
    p.add_argument('--measurements', action='store_true',
                   help='Install delivered b/Top/W efficiencies and measured trigger SFs by era')
    p.add_argument('--signal-efficiencies', action='store_true',
                   help='Install completed additive v15 signal counts, preserving the old counts and provenance')
    p.add_argument('--normalization', action='store_true',
                   help='Connect common xsecs to one era of delivered metadata, without editing ROOT')
    p.add_argument('--refresh-normalization', action='store_true',
                   help='Refresh pre-production early-era coverage, preserving the previous normalization')
    p.add_argument('--lowdm-model', action='store_true', help='Install the adopted 2022 portable model and SR bins')
    p.add_argument('--luminosity-fb', type=float,
                   help='Explicit luminosity for --normalization (one --prepare-eras value)')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.signal_efficiencies:
        if not args.prepare_eras or any((args.normalization, args.catalog, args.measurements,
                                        args.topw_references, args.lowdm_model, args.check_only)):
            p.error('--signal-efficiencies requires only --prepare-eras and --source-bundle')
        install_signal_efficiencies(args.source_bundle, args.prepare_eras)
        raise SystemExit(0)
    if args.lowdm_model:
        if not args.prepare_eras or args.normalization or args.catalog or args.measurements or args.topw_references:
            p.error('--lowdm-model requires only --prepare-eras and --source-bundle')
        install_early_lowdm(args.source_bundle, args.prepare_eras)
        raise SystemExit(0)
    if args.refresh_normalization and not args.normalization:
        p.error('--refresh-normalization requires --normalization')
    if args.luminosity_fb is not None and not args.normalization:
        p.error('--luminosity-fb requires --normalization')
    if args.prepare_eras:
        if args.normalization:
            if (len(args.prepare_eras) != 1 or args.luminosity_fb is None or
                    args.catalog or args.measurements or args.topw_references):
                p.error('--normalization requires one era and --luminosity-fb, without other install modes')
            install_era_normalization(args.source_bundle, args.prepare_eras[0],
                                      args.luminosity_fb, args.check_only, args.refresh_normalization)
        elif args.measurements:
            if args.catalog or args.check_only or args.topw_references:
                p.error('--measurements cannot be combined with another installation mode')
            install_era_measurements(args.source_bundle, args.prepare_eras)
        elif args.topw_references:
            if args.catalog or args.check_only:
                p.error('--topw-references does not accept --catalog or --check-only')
            install_topw_references(args.source_bundle, args.prepare_eras)
        elif args.catalog is None:
            p.error('--prepare-eras requires --catalog')
        else:
            install_early_eras(args.source_bundle, args.catalog, args.prepare_eras, args.check_only)
    else:
        if args.catalog or args.check_only or args.topw_references or args.measurements or args.normalization:
            p.error('--catalog and --check-only require --prepare-eras')
        install(args.source_bundle)

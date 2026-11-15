"""Boundary adapter for the unchanged AN physical-control-distribution renderer.

No event traversal, new bins, fit, or SR data. DY scaling intentionally uses the
legacy display policy: channel-specific effective RZ for inclusive variables,
and channel/Nb RZ for Nb itself. The likelihood's exact component mapping is a
separate operation. GCR unit-area normalization remains in the original renderer.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.paths import ROOT, internal_path, analysis_era, EARLY_ERAS, luminosity_fb
from TROTASR.gnn4lowdm.configuration import configuration_path
from TROTASR.utils.plot_adapter import renderer
from TROTASR.utils.statistical_templates import json_ready
from TROTASR.utils.statistics_inputs import combined_shape_endpoints, SHAPE_PREFIXES
from TROTASR.utils.statistics_model import nps_nuisance_name
from TROTASR.utils.observable_specs import (LOWDM_VARIABLE_SPECS,
                                           HIGHDM_DISTRIBUTION_VARIABLE_SPECS)

REGIONS = ('LLCR', 'QCDCR', 'GCR', 'DYCR')
SCHEMES = dict(LLCR='cat2_LLCR_lowDeltaM', QCDCR='cat3_QCDCR_lowDeltaM',
               GCR='cat4_GCR_lowDeltaM', DY2E='cat5_DY2E_lowDeltaM',
               DY2M='cat6_DY2M_lowDeltaM')
HIGH_VARIABLES = {r: ('met' if r in ('LLCR', 'QCDCR') else 'ut',
                       'nb', 'njet', 'ntop', 'nw') for r in REGIONS}
LOW_VARIABLES = {
    'LLCR': ('met', 'njet', 'nb_medium_lowdm', 'ht'),
    'QCDCR': ('met', 'njet', 'nb_medium_lowdm', 'ht'),
    'GCR': ('recoil_gcr', 'njet_photon_clean', 'nb_photon_clean', 'ht_photon_clean'),
    'DY2E': ('recoil_dy2e', 'njet_lepton_clean', 'nb_lepton_clean', 'ht_lepton_clean'),
    'DY2M': ('recoil_dy2m', 'njet_lepton_clean', 'nb_lepton_clean', 'ht_lepton_clean'),
}


def validate_histograms(payload, allow_validation=False):
    from TROTASR.workflows.rare_top_policy import require_clean
    require_clean(payload)
    if payload.get('status') != 'complete' or not payload.get('sr_data_blinded'):
        raise ValueError('Completed SR-blinded merged histograms required')
    scope = payload.get('scope')
    combined = scope == 'full_combined_systematic_production'
    shapes = combined_shape_endpoints(payload) if combined else set()
    if scope != 'full_nominal_production' and not combined and not allow_validation:
        raise ValueError('Full production required; validation needs explicit opt-in')
    year = payload['contract']['year']
    analysis_era(year)
    for key, file in (('sr_binning_sha256', 'jsons/highdm_sr_binning.json'),
                      ('configuration_sha256', str(configuration_path(year).relative_to(ROOT)))):
        if payload['contract'].get(key) != sha256(ROOT/file):
            raise ValueError('Changed histogram configuration: ' + key)
    for modes in ('histograms', 'physical_histograms'):
        for source in payload.get(modes, {}).values():
            if set(source) - {'nominal'} and not combined:
                raise ValueError('This workflow is nominal only')
            if combined:
                names = set(source)-{'nominal'}
                bases = {n[:-2] if n.endswith('Up') else n[:-4] if n.endswith('Down') else n for n in names}
                if names != {n+d for n in bases for d in ('Up','Down')} or {
                        n for n in names if n.startswith(SHAPE_PREFIXES)} != shapes:
                    raise ValueError('Incomplete/unpaired combined systematic endpoint inventory')
            for samples in source.get('nominal', {}).get('SR', {}).values():
                if 'data' in samples or 'data_obs' in samples:
                    raise ValueError('SR data are forbidden')
    return year


def systematic_sources(payload):
    if payload.get('scope') != 'full_combined_systematic_production':
        return []
    combined_shape_endpoints(payload)
    inventories = [set(payload[section][mode])-{'nominal'}
        for section in ('histograms','physical_histograms') for mode in ('highdm','lowdm')]
    if any(s != inventories[0] for s in inventories[1:]):
        raise ValueError('Physical and analysis histograms have different endpoint coverage')
    return sorted(n[:-2] for n in inventories[0] if n.endswith('Up'))


def endpoint_container(payload, section, mode, region, category, variable, edges, sources):
    """Preserve complete sparse shape populations; weight absence is nominal.

    Explicit zeros are boundary-array representations of verified absent shape
    cells, not missing-input replacements. Shifted-only samples remain visible.
    """
    variations = payload[section][mode]
    nominal = variations['nominal'].get(region,{}).get(category,{})
    shapes = combined_shape_endpoints(payload) if sources else set()
    endpoints = [n+d for n in sources for d in ('Up','Down')]
    samples = set(nominal)
    for endpoint in endpoints:
        samples.update(variations[endpoint].get(region,{}).get(category,{}))
    empty = dict(sumw=[0.]*(len(edges)-1),sumw2=[0.]*(len(edges)-1),entries=[0.]*(len(edges)-1))
    converted = {}
    for sample in sorted(samples):
        if sample.startswith(('mStop','T2tt_','T2bW_','T2tb_')):
            continue
        name = 'data_obs' if sample == 'data' else sample
        if name != 'data_obs': renderer.process_to_group(name)
        central = checked_leaf(nominal[sample][variable],edges) if sample in nominal else deepcopy(empty)
        record = converted[name] = dict(nominal=central)
        for endpoint in endpoints:
            varied = variations[endpoint].get(region,{}).get(category,{}).get(sample)
            if name == 'data_obs':
                if varied is not None: raise ValueError('Data must not be shifted')
                continue
            if endpoint not in shapes and sample not in nominal and varied is not None:
                raise ValueError('Weight endpoint has a non-nominal population')
            if varied is not None:
                record[endpoint] = checked_leaf(varied[variable],edges)
            elif endpoint in shapes:
                record[endpoint] = deepcopy(empty)
            else:
                record[endpoint] = deepcopy(central)
    if 'data_obs' not in converted or 'data' not in nominal:
        raise ValueError('Missing collision-data coverage: '+mode+'/'+region+'/'+variable)
    return converted


def checked_leaf(leaf, edges):
    if leaf.get('edges') != edges:
        raise ValueError('Changed physical histogram edges; no automatic rebinning')
    result = {}
    for key in ('sumw', 'sumw2', 'entries'):
        values = np.asarray(leaf[key], dtype=float)
        if (values.shape != (len(edges)-1,) or not np.isfinite(values).all()
                or (key != 'sumw' and (values < 0).any())):
            raise ValueError('Malformed physical histogram: ' + key)
        result[key] = values.tolist()
    return result


def physical_boundary(payload, include_systematics=False):
    """Sum the two stored Nb streams without changing physical variable bins."""
    result = dict(highdm_distribution_variable_specs=deepcopy(HIGHDM_DISTRIBUTION_VARIABLE_SPECS),
                  lowdm_variable_specs=deepcopy(LOWDM_VARIABLE_SPECS),
                  highdm_variable_histograms={}, lowdm_variable_histograms={})
    sources = systematic_sources(payload) if include_systematics else []
    for mode, specs, target in (
            ('highdm', HIGHDM_DISTRIBUTION_VARIABLE_SPECS, result['highdm_variable_histograms']),
            ('lowdm', LOWDM_VARIABLE_SPECS, result['lowdm_variable_histograms'])):
        regions = payload['physical_histograms'][mode]['nominal']
        for region in ('LLCR', 'QCDCR', 'GCR', 'DY2E', 'DY2M'):
            if set(regions.get(region, {})) != {'Nb1', 'Nb2'}:
                raise ValueError('Missing or extra physical CR Nb stream: ' + mode+'/'+region)
            variables = (HIGH_VARIABLES['DYCR' if region.startswith('DY') else region]
                         if mode == 'highdm' else LOW_VARIABLES[region])
            output = target.setdefault(region if mode == 'highdm' else SCHEMES[region], {})
            for variable in variables:
                containers = []
                for category, samples in regions[region].items():
                    if include_systematics:
                        containers.append(endpoint_container(payload,'physical_histograms',mode,region,
                            category,variable,specs[variable]['bins'],sources))
                        continue
                    converted = {}
                    for sample, records in samples.items():
                        if sample.startswith(('mStop', 'T2tt_', 'T2bW_', 'T2tb_')):
                            continue  # Established physical-CR policy: no signal overlays.
                        name = 'data_obs' if sample == 'data' else sample
                        if name != 'data_obs':
                            renderer.process_to_group(name)  # reject unknown processes
                        converted[name] = {'nominal': checked_leaf(records[variable], specs[variable]['bins'])}
                    if 'data_obs' not in converted:
                        raise ValueError('Missing collision-data coverage: '+mode+'/'+region+'/'+variable)
                    containers.append(converted)
                output[variable] = renderer.combine_histogram_containers(*containers)
    return result


def dy_display_config(high, low):
    """Preserve the existing effective-RZ display prescription, not a new fit."""
    channels = {channel: {} for channel in ('DY2E', 'DY2M')}
    for mode, product, key in (('highdm', high, 'rz_high'), ('lowdm', low, 'rz_low')):
        if product.get('status') != 'complete':
            raise ValueError('Incomplete current RZ measurement: ' + mode)
        for channel in channels:
            record, yields = {}, {}
            for group in ('Nb1', 'Nb2plus'):
                fit = product[key]['channels'][channel][group]
                if fit.get('status') != 'complete' or not fit.get('fit_converged'):
                    raise ValueError('Unsuccessful current RZ channel fit')
                value = float(fit['RZ'])
                weight = float(product[key+'_raw'][channel][group]['on']['zll']['sumw'])
                if not np.isfinite([value, weight]).all() or value < 0 or weight < 0:
                    raise ValueError('Unsupported effective RZ display measurement')
                record[group], yields[group] = value, weight
            denominator = sum(yields.values())
            if denominator <= 0:
                raise ValueError('No DY support for effective RZ')
            record.update(effective=sum(record[g]*yields[g] for g in yields)/denominator,
                          dy_yields_for_effective=yields)
            channels[channel][mode] = record
    return dict(dy_rz=dict(status='complete', channels=channels,
        factor_policy='Legacy channel-specific on-Z DY-weighted effective RZ; exact Nb factors for Nb plots'))


@contextmanager
def nominal_renderer(year, sources=()):
    """Bind the current display bins and era settings to the shared renderer."""
    mapping = deepcopy(renderer.HIGHDM_MULTIPLICITY_PLOT_BINS)
    mapping['njet'] = dict(
        source_edges=HIGHDM_DISTRIBUTION_VARIABLE_SPECS['njet']['bins'],
        groups=[[3], [4], [5], [6], [7, 8, 9]],
        edges=[3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        labels=['4', '5', '6', '7-8', r'$\geq9$'],
    )
    updates = dict(LUMINOSITY_FB=luminosity_fb(year),
                   LUMINOSITY_RELATIVE_UNCERTAINTY=0. if analysis_era(year) in EARLY_ERAS else .016,
                   PLOT_SYSTEMATIC_SOURCES=list(sources),
                   HIGHDM_MULTIPLICITY_PLOT_BINS=mapping)
    previous = {key:getattr(renderer, key) for key in updates}
    try:
        for key,value in updates.items():
            setattr(renderer, key, value)
        yield
    finally:
        for key,value in previous.items():
            setattr(renderer, key, value)


def omitted_physical_cells(raw, variable):
    """Audit nonzero selected entries outside the defined display groups."""
    config = renderer.HIGHDM_MULTIPLICITY_PLOT_BINS.get(variable)
    if config is None:
        return []
    covered = {i for group in config['groups'] for i in group}
    dropped = sorted(set(range(len(config['source_edges'])-1))-covered)
    audit = []
    for sample, variations in raw.items():
        leaf = variations['nominal']
        for i in dropped:
            values = {key:float(leaf[key][i]) for key in ('sumw','sumw2','entries')}
            if any(value != 0 for value in values.values()):
                audit.append(dict(sample=sample,source_bin_1based=i+1,
                    interval=config['source_edges'][i:i+2],**values))
    return audit


def draw_distribution(blocks, outbase, xlabel, luminosity_fb, uncertainty_label=None):
    """One rendering entry point for physical and GNN CR distributions.

    Numerical inputs, normalization and category boundaries are supplied by the
    caller; this function never recalculates them. GNN-specific canvas, legend,
    axis-range and annotation overrides do not belong in this shared template.
    """
    blocks = deepcopy(blocks)
    for block in blocks:
        block['reference_style'] = True
    if len(blocks) > 1:
        for block in blocks:
            block.update(label_box=True, category_labels_on_main=True,
                         category_label_y=.72)
    rotation = 90.0 if any(b.get('variable')=='gnn_score' for b in blocks) else 0.0
    import matplotlib
    # Reserve space for vertical interval labels on this plot while retaining
    # the standard margins for ordinary distributions.
    with matplotlib.rc_context({'figure.subplot.bottom':.25} if rotation else {}):
        return renderer.draw_flat_blocks(blocks, outbase, xlabel=xlabel,
            reference_style=True, show_yields=True, luminosity_fb=luminosity_fb,
            uncertainty_label_override=uncertainty_label,x_tick_rotation=rotation)


def _physical_variables(mode, region):
    if mode == 'highdm':
        return HIGH_VARIABLES[region]
    return LOW_VARIABLES['DY2E' if region == 'DYCR' else region]


def _physical_plot_name(mode, region, variable):
    display = 'recoil_dy' if variable == 'recoil_dy2e' else variable
    prefix = 'cr_' if mode == 'highdm' else 'lowdm_cr_'
    return prefix + region.lower() + '_' + display, display


def make_records(boundary, modes=('highdm', 'lowdm'), regions=REGIONS):
    output = []
    for mode in modes:
        for region in regions:
            if mode == 'highdm':
                for variable in _physical_variables(mode, region):
                    source = boundary['highdm_variable_histograms']
                    raw = (renderer.combine_histogram_containers(source['DY2E'][variable], source['DY2M'][variable])
                           if region == 'DYCR' else source[region][variable])
                    record = renderer.highdm_variable_record(boundary, region, variable, raw_override=raw)
                    if record is not None:
                        record['display_omission_audit'] = omitted_physical_cells(raw, variable)
                    name, display = _physical_plot_name(mode, region, variable)
                    output.append((name, mode, region, display, record))
            else:
                for variable in _physical_variables(mode, region):
                    name, display = _physical_plot_name(mode, region, variable)
                    if region == 'DYCR':
                        source = boundary['lowdm_variable_histograms']
                        muon = 'recoil_dy2m' if variable == 'recoil_dy2e' else variable
                        raw = renderer.combine_histogram_containers(source[SCHEMES['DY2E']][variable],
                                                                    source[SCHEMES['DY2M']][muon])
                        scheme, label = 'DYCR', r'DYCR ($ee+\mu\mu$), $N_{res}=0$'
                    else:
                        scheme = SCHEMES[region]
                        raw = boundary['lowdm_variable_histograms'][scheme][variable]
                        label = region + r' low $\Delta m$, $N_{res}=0$'
                    record = renderer.lowdm_variable_record(boundary, scheme, variable, label,
                        allow_signal=False, raw_override=raw, spec_variable=variable, display_variable=display)
                    if record is not None:
                        record['blind_data'] = False
                    output.append((name, mode, region, display, record))
    if any(row[-1] is None for row in output):
        raise ValueError('Unsupported empty physical plot: '+str([row[0] for row in output if row[-1] is None]))
    return output


def assert_band_only(before, after):
    """Exact plot-input identity except the two uncertainty fields."""
    omitted = {'background_unc','background_systematic_totals'}
    if json_ready({k:v for k,v in before.items() if k not in omitted}) != json_ready({
            k:v for k,v in after.items() if k not in omitted}):
        raise ValueError('Systematic update changed central/statistics/data/axis/style input')


def _cr_inputs(inputs):
    return {key: value for key, value in inputs.items() if key != 'hists'}


def _same_cr_plot_contract(previous, current):
    """The full merged file hash is provenance, not a CR cache key."""
    for key, value in current.items():
        if key == 'reference_input_sha256':
            continue
        old_value = previous.get(key)
        if key == 'inputs' and isinstance(old_value, dict) and isinstance(value, dict):
            old_value, value = _cr_inputs(old_value), _cr_inputs(value)
        if old_value != value:
            return False
    return True


def _verify_plot_manifest(output, receipt, contract, names):
    if not _same_cr_plot_contract(receipt['contract'], contract):
        raise ValueError('Divergent plot outputs must be preserved')
    requested = set(names)
    blocked = {row['name'] for row in receipt.get('blocked_plots', ())}
    expected = {name + suffix for name in requested - blocked for suffix in ('.pdf', '.png')}
    if not blocked <= requested or set(receipt['files']) != expected:
        raise ValueError('Existing figures do not cover the requested inventory')
    if any(sha256(output/name) != digest for name,digest in receipt['files'].items()):
        raise ValueError('Divergent plot outputs must be preserved')


def render_input(payload, output):
    if payload.get('schema') != 'trotasr_physical_plot_inputs_v1':
        raise ValueError('Unrecognized compact physical plot input')
    contract = payload['contract']; year = contract['year']
    manifest_path = output/'plot_manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        _verify_plot_manifest(output, saved, contract, (row[0] for row in payload['records']))
        return saved
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve incomplete outputs; choose a new directory')
    output.mkdir(parents=True,exist_ok=True)
    reports, blocked = [], []
    with nominal_renderer(year,payload['systematic_sources']):
        for name,mode,region,variable,record in payload['records']:
            if record.get('display_omission_audit'):
                blocked.append(dict(name=name,mode=mode,region=region,variable=variable,
                    status='blocked_display_omits_selected_events',
                    cells=record['display_omission_audit'],renderer_changed=False))
                continue
            info=draw_distribution([record],output/name,record['xlabel'],luminosity_fb(year))
            info.update(status='generated_pending_visual_QA',year=year,mode=mode,region=region,
                variable=variable,scope=contract['scope'],
                an_reference='figures/results_distributions/'+str(year)+'/'+name+'.pdf')
            reports.append(info)
    files={p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.png','.pdf')}
    if len(files)!=len(reports)*2: raise ValueError('Missing physical figures')
    result=dict(status='partial_generated_pending_visual_QA' if blocked else 'generated_pending_visual_QA',
        contract=contract,plots=reports,blocked_plots=blocked,files=files,
        dy_rz_application=payload['dy_rz_application'],sr_data_blinded=True,
        uncertainty_audit=payload['uncertainty_audit'],full_workflow_complete=False)
    write_json(output/'plot_inputs.json',payload)
    write_json(manifest_path,result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--hists',type=Path)
    inputs.add_argument('--input',type=Path,help='Compact verified physical plot inputs')
    for name in ('rz-high','rz-low'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--gnn-output',type=Path,help='Optional compact GNN export from this same loaded histogram')
    p.add_argument('--predictions',type=Path,help='Verified card compact export for --gnn-output')
    p.add_argument('--mode', choices=('highdm', 'lowdm'), action='append')
    p.add_argument('--region', choices=REGIONS, action='append')
    p.add_argument('--allow-validation', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.input:
        if a.export_only or a.rz_high or a.rz_low or a.mode or a.region or a.allow_validation or a.gnn_output or a.predictions:
            raise ValueError('Compact inputs already freeze measurements and figure selection')
        payload=read_json(internal_path(a.input))
        if a.validate_only:
            print(payload['contract']);return
        result=render_input(payload,internal_path(a.output))
        if result['blocked_plots']:raise SystemExit(2)
        return
    if not a.rz_high or not a.rz_low:
        raise ValueError('Both current RZ products are required with merged histograms')
    if bool(a.gnn_output)!=bool(a.predictions) or a.gnn_output and not a.export_only:
        raise ValueError('Shared GNN projection requires explicit compact predictions and export-only')
    if a.gnn_output:
        target=internal_path(a.gnn_output)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError('Preserve existing GNN compact inputs before reading histograms')
    paths = {name:internal_path(getattr(a, name)) for name in ('hists', 'rz_high', 'rz_low')}
    output = internal_path(a.output)
    source_hash = {name:sha256(path) for name,path in paths.items()}
    hists, high, low = (read_json(paths[n]) for n in ('hists', 'rz_high', 'rz_low'))
    year = validate_histograms(hists, a.allow_validation)
    for factor in (high, low):
        if (str(factor['provenance'].get('campaign_year')) != str(year)
                or factor.get('scope') != hists['scope']):
            raise ValueError('Measured RZ does not match histogram year/scope')
    contract = dict(inputs=source_hash, year=year, scope=hists['scope'], modes=a.mode or ['highdm', 'lowdm'],
                    regions=a.region or list(REGIONS), include_four_jet_bin=True)
    manifest_path = output/'plot_manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        names = (_physical_plot_name(mode, region, variable)[0]
                 for mode in contract['modes'] for region in contract['regions']
                 for variable in _physical_variables(mode, region))
        _verify_plot_manifest(output, saved, contract, names)
        print(saved['status'], '(verified existing outputs)')
        if saved.get('blocked_plots'):
            raise SystemExit(2)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve incomplete outputs; choose a new directory')
    sources=systematic_sources(hists)
    boundary = physical_boundary(hists,include_systematics=bool(sources))
    baseline=physical_boundary(hists) if sources else None
    config = dy_display_config(high, low)
    # apply_dy_rz is byte-preserved and reads an internal compact manifest.
    # A temporary internal JSON exists only during validation, not a new event sidecar.
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dy_display_', dir=output.parent) as temporary:
        config_path = Path(temporary)/'dy_display.json'
        write_json(config_path, config)
        application = renderer.apply_dy_rz(boundary, config_path)
        if baseline is not None:renderer.apply_dy_rz(baseline,config_path)
    if baseline is not None:
        with nominal_renderer(year):
            central_records=make_records(baseline,contract['modes'],contract['regions'])
    with nominal_renderer(year,sources):
        records = make_records(boundary, contract['modes'], contract['regions'])
        if baseline is not None:
            if [r[:4] for r in records]!=[r[:4] for r in central_records]:
                raise ValueError('Changed physical figure inventory')
            for nominal, varied in zip(central_records,records):assert_band_only(nominal[-1],varied[-1])
        if a.validate_only:
            blocked = [row[0] for row in records if row[-1].get('display_omission_audit')]
            print(dict(status='validated_with_blocked_plot_inputs' if blocked else 'validated_plot_inputs_not_generated',
                       count=len(records), blocked_plots=blocked, contract=contract))
            if blocked:
                raise SystemExit(2)
            return
    payload=dict(schema='trotasr_physical_plot_inputs_v1',contract=contract,
        systematic_sources=sources,records=json_ready(records),dy_rz_application=application,
        uncertainty_audit=dict(event_nuisances={n:nps_nuisance_name(n,str(year)) for n in sources},
            same_name_shifts_summed_before_envelope=True,mcstat_counted_once=True,
            central_sumw2_data_labels_axes_exactly_preserved=True,
            gcr_unit_area_normalizes_each_varied_total=True,
            luminosity='Existing 0.016; omitted for unit-area GCR',
            measured_sr_only_shapes_not_assigned_to_cr=True,
            auxiliary_dy_rz_fit_uncertainty='Not propagated: existing channel/Nb effective display correction has no approved correlated fit-error template mapping',
            preliminary=hists.get('preliminary',False),
            combination_provenance=hists.get('combination_provenance')))
    if a.export_only:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'plot_inputs.json',payload)
        if a.gnn_output:
            from TROTASR.workflows.plotting_gnn_controls import export_loaded
            target=internal_path(a.gnn_output)
            if target.exists() and any(target.iterdir()):raise FileExistsError('Preserve existing GNN compact inputs')
            compact=export_loaded(hists,low,read_json(internal_path(a.predictions)),
                source_hash['hists'],source_hash['rz_low'],a.allow_validation)
            target.mkdir(parents=True,exist_ok=True);write_json(target/'plot_inputs.json',compact)
        print(dict(status='compact_inputs_exported',records=len(records)));return
    result=render_input(payload,output)
    if result['blocked_plots']:raise SystemExit(2)


if __name__ == '__main__':
    main()

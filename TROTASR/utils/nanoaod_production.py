"""One-file, one-era signal production using the current TROTASR definitions."""
from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from .paths import ROOT, analysis_era, EARLY_ERAS, tagging_definition
from .io import write_json
from . import nanoaod_events as events
from . import nanoaod_features as features
from .nanoaod_calibration import calibrate, jet_id
from .reader import project_candidate_entries


def open_source(lfn):
    errors = []
    for host in ('cms-xrd-global.cern.ch', 'xrootd-cms.infn.it', 'cmsxrootd.fnal.gov'):
        try:
            return uproot.open('root://' + host + '/' + lfn, timeout=90)
        except Exception as exc:
            errors.append(str(exc)[-400:])
    raise OSError('Source read failed: ' + lfn + ': ' + '; '.join(errors))


def cache_source(source, directory):
    """Bounded worker-local copy; no random remote read for every event column."""
    target = Path(directory) / (Path(source['name']).stem + '.nanoaod.root')
    if target.exists():
        return target
    temporary = target.with_suffix('.partial')
    errors = []
    for host in ('cms-xrd-global.cern.ch', 'xrootd-cms.infn.it', 'cmsxrootd.fnal.gov'):
        try:
            command = ['xrdcp', '--force',
                       'root://' + host + '/' + source['name'], str(temporary)]
            subprocess.run(command, check=True, timeout=1200)
            if temporary.stat().st_size != source['size']:
                raise OSError('Incomplete NanoAOD transfer: ' + source['name'])
            os.replace(temporary, target)
            return target
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            errors.append(str(exc))
    raise OSError('NanoAOD transfer failed after all redirectors: ' + '; '.join(errors))


def file_record(source, era):
    return dict(dataset=source['dataset'], file_path=source['name'], year=era,
                process_group='SMS', is_signal=True, is_data=False, is_background=False,
                simulation_type='FastSim signal dataset', signal_point=None, xsec_pb=None)


def schema_record():
    return dict(int64=events.INT64_FIELDS, int32=events.INT32_FIELDS,
                float32=[k for k in events.FLOAT_FIELDS if k not in events.FLOAT64_STORAGE_FIELDS],
                float64=sorted(events.FLOAT64_STORAGE_FIELDS), boolean=events.BOOL_FIELDS,
                vector_float32=events.VECTOR_FLOAT_FIELDS, vector_int32=events.VECTOR_INT_FIELDS)


def _skim_candidates(light, *, trigger_bypass):
    """Necessary conditions for every branch of feature_flat_preselection in SMS."""
    possible, _ = features.all_filters(light, len(light))
    if 'nJet' in light.fields:
        # Corrections/cleaning cannot create extra AK4 objects.
        possible &= np.asarray(light['nJet']) >= 2
    if not trigger_bypass:
        possible &= features.bool_branch(light, features.SIGNAL_HLT + features.PHOTON_HLT
                                         + features.ELECTRON_HLT + features.MUON_HLT, len(light))
    return np.flatnonzero(possible)


def build_events(source, era, output, *, chunk_size=2000):
    """Read one original file completely; preserve signed weights and Runs sums."""
    era = analysis_era(era)
    output = Path(output)
    events.configure_era(era)
    record = file_record(source, era)
    file_id = events.stable_id(source['name'])
    dataset = events.empty_dataset_record(record)
    dataset.update(files_attempted=1)
    report = dict(file_path=source['name'], file_id=file_id, dataset_id=dataset['dataset_id'],
                  events_read=0, events_written=0, processed_entry_ranges=[])
    writer = events.RootRowWriter(output)
    writer._ensure_open()
    truth_types = {k: np.int64 for k in events.TOPW_ID_FIELDS}
    truth_types.update(fatjet_source_index_all='var * int32', fatjet_decay_flavor_all='var * int8')
    writer.root_file.mktree('TopWTruth', truth_types)
    started = time.time()
    try:
        cached = cache_source(source, output.parent)
        with uproot.open(cached) as root:
            tree = root['Events']
            if tree.num_entries != source['nevents']:
                raise ValueError('Original NanoAOD entry count differs from DAS')
            runs = events.read_runs_sumw(root)
            report['runs_sumw'] = runs
            dataset['signal_sumw_by_genmodel'] = runs['signal_sumw_by_genmodel']
            dataset['sumw'] = runs['generic_sumw']
            fields = list(tree.keys())
            early = era in EARLY_ERAS
            bypass = not early or not any(k in fields for k in features.SIGNAL_HLT)
            report['fastsim_trigger_bypass'] = bypass
            report['required_filters'] = list(features.FILTERS)
            models = [k for k in fields if events.signal_genmodel_branch(k)]
            light_fields = [k for k in dict.fromkeys(['genWeight', 'nJet', *models, *features.FILTERS,
                *features.SIGNAL_HLT, *features.PHOTON_HLT, *features.ELECTRON_HLT, *features.MUON_HLT])
                if k in fields]
            light_set = set(light_fields)
            heavy_fields = [k for k in fields if k not in light_set]
            report['object_correction_input_rows'] = 0
            for start in range(0, tree.num_entries, chunk_size):
                stop = min(start + chunk_size, tree.num_entries)
                light = tree.arrays(light_fields, entry_start=start, entry_stop=stop, library='ak')
                if len(light) != stop - start:
                    raise ValueError('Incomplete source scan for the signal normalization')
                # Book the full source population, even when no event in this slice is retained.
                weights = np.asarray(light['genWeight'], dtype=np.float64)
                dataset['sumw2'] += float(np.sum(weights * weights))
                for branch in models:
                    value = float(weights[np.asarray(light[branch], dtype=bool)].sum())
                    events.add_float_map(dataset['signal_event_genweight_sum_by_genmodel'], {branch: value})
                report['events_read'] += stop - start
                report['processed_entry_ranges'].append(dict(entry_start=start, entry_stop=stop))
                candidates = _skim_candidates(light, trigger_bypass=bypass)
                if not len(candidates):
                    continue
                absolute = candidates + start
                heavy = project_candidate_entries(tree, heavy_fields, absolute)
                raw = ak.zip({**{k: light[k][candidates] for k in light_fields},
                              **{k: heavy[k] for k in heavy_fields}}, depth_limit=1)
                report['object_correction_input_rows'] += len(raw)
                if early:
                    for prefix in ('Jet', 'FatJet'):
                        raw = ak.with_field(raw, ak.values_astype(jet_id(raw, prefix), np.int32) * 6,
                                            prefix + '_jetId')
                corrected, calibration = calibrate(raw, era)
                rows, summary = features.extract_chunk(
                    corrected, source['dataset'], 'SMS', None, era, source['name'], 0, len(raw),
                    fastsim_trigger_bypass=bypass, shift_name='nominal', compute_weights=False,
                    materialize_skim_flag='feature_flat_preselection')
                rows = events.decorate(rows, raw, corrected, 0, era, record, file_id)
                indices = np.asarray([row['entry'] for row in rows], dtype=np.int64)
                # Decoration indexes the compact arrays; publish original absolute NanoAOD entries.
                for row, entry in zip(rows, absolute[indices]):
                    row['entry'] = int(entry)
                writer.extend(rows)
                if rows:
                    stored = ak.Array(events.rows_to_arrays(rows))
                    truth = events.topw_truth_payload(stored, raw[indices], events.decay_flavor)
                    writer.root_file['TopWTruth'].extend(truth)
                report['events_written'] += len(rows)
                report.setdefault('object_corrections', calibration)
                print(json.dumps(dict(era=era, source=Path(source['name']).name,
                                      events=stop, total=tree.num_entries, retained=report['events_written'])), flush=True)
            report.update(processing_status='processed', number_of_entries=tree.num_entries)
        dataset.update(files_processed=1, events_read=report['events_read'],
                       events_written=report['events_written'],
                       sumw_source_counts={'Runs.genEventSumw': 1},
                       signal_runs_sumw_source_counts={'Runs.genEventSumw_by_GenModel': 1})
        writer.root_file['TopWTruth_metadata'] = json.dumps(dict(
            schema_version='topw_truth_v1', status='complete', events_entries=report['events_written']))
        writer.close(commit=True)
    except BaseException:
        writer.close(commit=False)
        raise
    metadata = dict(status='events_complete', schema_version=events.SCHEMA_VERSION, analysis_era=era,
                    input_nanoaod_version=15, source_nanoaod_modified=False, root_file=str(output),
                    files_attempted=1, files_processed=1, files=[report],
                    events_read=report['events_read'], events_written=report['events_written'],
                    datasets={str(dataset['dataset_id']): dataset}, branch_schema=schema_record(),
                    seconds=time.time() - started, bad_files=[], multiplicative_sf='deferred_to_histograms')
    events.update_physical_norm(metadata)
    write_json(output.with_suffix('.json'), metadata)
    return metadata


def candidate_markers(era, counts):
    """Completion metadata shared by the producer and histogram reader."""
    era = analysis_era(era)
    tag = tagging_definition(era)
    counts = {key: int(counts[key]) for key in ('events', 'resolved_passing', 'mixed_passing')}
    resolved = dict(status='complete', **counts, application_year=era[:4],
                    analysis_era=era, events_entries=counts['events'])
    mixed = dict(status='complete', year=era[:4], analysis_era=era,
                 entries=counts['events'], passing_candidates=counts['mixed_passing'],
                 operator='>=', threshold=tag['mixed_wp'])
    return resolved, mixed


def parent_deepcsv_complete(output, metadata):
    """Use the committed ROOT branch to resume the early-era parent step."""
    with uproot.open(output) as root:
        tree = root['Events']
        if 'fatjet_btagdeepb_all' not in tree or 'TopMixedInputs_metadata' not in root:
            return False
        marker = json.loads(str(root['TopMixedInputs_metadata']))
        if (tree.num_entries != metadata['events_written']
                or tree['fatjet_btagdeepb_all'].num_entries != tree.num_entries
                or marker.get('entries') != tree.num_entries or marker.get('status') != 'complete'
                or marker.get('target') != 'fatjet_btagdeepb_all'):
            raise ValueError('Incomplete saved early-era DeepCSV association')
    metadata['topmixed_inputs'] = marker
    names = metadata['branch_schema']['vector_float32']
    if 'fatjet_btagdeepb_all' not in names:
        names.append('fatjet_btagdeepb_all')
    write_json(output.with_suffix('.json'), metadata)
    return True


def recover_candidate_metadata(output, era, metadata):
    """Finish the receipt of an already committed candidate ROOT."""
    era = analysis_era(era)
    with uproot.open(output) as root:
        if 'TROTA_metadata' not in root or 'TopMixedResults_metadata' not in root:
            return False
        resolved = json.loads(str(root['TROTA_metadata']))
        mixed = json.loads(str(root['TopMixedResults_metadata']))
        entries = root['Events'].num_entries
        if (metadata.get('analysis_era') != era or entries != metadata['events_written']
                or root['TopWTruth'].num_entries != entries
                or resolved.get('status') != 'complete' or mixed.get('status') != 'complete'
                or resolved.get('analysis_era') != era or mixed.get('analysis_era') != era
                or resolved.get('events_entries') != entries or mixed.get('entries') != entries
                or resolved.get('resolved_passing') != root['TROTA'].num_entries
                or mixed.get('threshold') != tagging_definition(era)['mixed_wp']
                or 'nTopMixed1pct' not in root['Events']):
            raise ValueError('Incomplete saved signal candidate ROOT')
    metadata.update(status='complete', campaign_year=era, trota=resolved, topmixed_results=mixed)
    write_json(output.with_suffix('.json'), metadata)
    return True


def complete_metadata(output, era):
    """Repair missing completion objects without recalculating or changing events."""
    output = Path(output)
    era = analysis_era(era)
    metadata = json.loads(output.with_suffix('.json').read_text())
    with uproot.open(output) as root:
        if 'TopMixedResults_metadata' in root and metadata.get('campaign_year') == era:
            return
        counts = metadata['trota']
        if (metadata['status'] != 'complete' or metadata['analysis_era'] != era
                or counts['events'] != root['Events'].num_entries
                or counts['resolved_passing'] != root['TROTA'].num_entries
                or metadata['topmixed_results']['status'] != 'complete'
                or 'nTopMixed1pct' not in root['Events']):
            raise ValueError('Cannot finalize incomplete candidate results')
        resolved, mixed = candidate_markers(era, counts)
    with uproot.update(output) as root:
        root['TROTA_metadata'] = json.dumps(resolved)
        root['TopMixedResults_metadata'] = json.dumps(mixed)
    metadata.update(campaign_year=era, trota=resolved, topmixed_results=mixed)
    write_json(output.with_suffix('.json'), metadata)
    receipt = ROOT / 'production/collected' / era / (output.stem + '.json')
    if receipt.exists():
        record = json.loads(receipt.read_text())
        record.setdefault('transferred_bytes', record['bytes'])
        record.update(bytes=output.stat().st_size, completion_metadata_repaired=True)
        write_json(receipt, record)


def complete_trota(output, era):
    """Add resolved and mixed candidates to a new ROOT in bounded chunks."""
    from .trota_inference import load_models, bounded_slices, infer_chunk, MIXED_INTS, MIXED_FLOATS
    from .trota_friend import MIXED, _candidate_support, _friend_rows, _identity
    output = Path(output)
    temporary = output.with_name(output.stem + '.candidates.root')
    events.configure_era(era)
    models = load_models(era)
    integers = ('file_id', 'entry', 'TopResolved1pct_candidateIndex',
                'TopResolved1pct_sourceJetIdx0', 'TopResolved1pct_sourceJetIdx1',
                'TopResolved1pct_sourceJetIdx2')
    floats = ('TopResolved1pct_QCDDiscriminant', 'TopResolved1pct_eta', 'TopResolved1pct_mass')
    trota_types = {**{k: np.int64 for k in integers}, **{k: np.float32 for k in floats}}
    counts = dict(events=0, resolved_passing=0, mixed_passing=0)
    with uproot.open(output) as original, uproot.recreate(temporary) as target:
        target.mktree('TROTA', trota_types)
        event_types = events.branch_types()
        event_types['nTopMixed1pct'] = np.int32
        event_types.update({'TopMixed1pct_' + k: 'var * int32' for k in MIXED_INTS})
        event_types.update({'TopMixed1pct_' + k: 'var * float32' for k in MIXED_FLOATS})
        if analysis_era(era) in EARLY_ERAS:
            event_types['fatjet_btagdeepb_all'] = 'var * float32'
        target.mktree('Events', event_types)
        for chunk in original['Events'].iterate(step_size=2000, library='ak'):
            for sl in bounded_slices(chunk, year=era):
                selected = chunk[sl]
                positions = np.flatnonzero(_candidate_support(selected, era))
                mixed = {}
                if len(positions):
                    compact = selected if len(positions) == len(selected) else selected[positions]
                    updated, _, audit, state = infer_chunk(compact, models, materialize_resolved=False)
                    mixed = {k: updated[k] for k in MIXED}
                    for k in ('mixed_passing', 'resolved_passing'):
                        counts[k] += audit[k]
                    rec = state['resolved']
                    keep = rec['passing']
                    if np.any(keep):
                        rows = rec['event'][keep]
                        source = rec['sources'][keep]
                        values = (np.asarray(compact['file_id'])[rows], np.asarray(compact['entry'])[rows],
                                  rec['index'][keep], *source.T, rec['discriminant'][keep],
                                  rec['p4'][keep, 1], rec['p4'][keep, 3])
                        target['TROTA'].extend({name: value.astype(trota_types[name], copy=False)
                                               for name, value in zip((*integers, *floats), values)})
                columns = {name: selected[name] for name in event_types if name not in MIXED}
                columns.update(_friend_rows(_identity(selected), positions, mixed, layouts=(MIXED,)))
                target['Events'].extend(columns)
                counts['events'] += len(selected)
        truth = original['TopWTruth']
        target.mktree('TopWTruth', {**{k: np.int64 for k in events.TOPW_ID_FIELDS},
                                   'fatjet_source_index_all': 'var * int32',
                                   'fatjet_decay_flavor_all': 'var * int8'})
        names = [*events.TOPW_ID_FIELDS, 'fatjet_source_index_all', 'fatjet_decay_flavor_all']
        for chunk in truth.iterate(names, step_size=20000, library='ak'):
            target['TopWTruth'].extend({k: chunk[k] for k in names})
        target['TopWTruth_metadata'] = str(original['TopWTruth_metadata'])
        resolved_marker, mixed_marker = candidate_markers(era, counts)
        target['TROTA_metadata'] = json.dumps(resolved_marker)
        target['TopMixedResults_metadata'] = json.dumps(mixed_marker)
    os.replace(temporary, output)
    metadata = json.loads(output.with_suffix('.json').read_text())
    metadata.update(status='complete', campaign_year=analysis_era(era),
                    trota=resolved_marker, topmixed_results=mixed_marker)
    write_json(output.with_suffix('.json'), metadata)
    return metadata


def validate_output(path, era, source):
    """New-output structural checks, without exhaustive numerical readback."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (metadata['status'] != 'complete' or metadata['events_read'] != source['nevents']
            or metadata.get('analysis_era') != analysis_era(era)):
        raise ValueError('Incomplete signal production')
    with uproot.open(path) as root:
        if root['Events'].num_entries != metadata['events_written']:
            raise ValueError('Events count differs from metadata')
        if root['TopWTruth'].num_entries != root['Events'].num_entries:
            raise ValueError('TopWTruth count differs from Events')
        required = {'nTopMixed1pct', 'mStop', 'mLSP', 'file_id', 'entry', 'gen_weight'}
        if analysis_era(era) in EARLY_ERAS:
            required.add('fatjet_btagdeepb_all')
        if required - set(root['Events'].keys()) or 'TROTA' not in root:
            raise ValueError('Required intermediate objects are missing')
    return metadata

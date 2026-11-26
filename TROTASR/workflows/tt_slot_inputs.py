"""Collect the completed TT-slot detector replay into per-era likelihood inputs."""
import fcntl
from pathlib import Path

from TROTASR.utils.io import read_json, write_json
from TROTASR.utils.paths import ROOT, ERAS, internal_path
from TROTASR.utils.top_pt import ENDPOINTS, PRESCRIPTION
from TROTASR.workflows.merge_nominal import merge_histograms
from TROTASR.workflows.top_pt_inventory import RELATIVE, experimental_plans


def collect(era, output):
    base = ROOT / RELATIVE
    output = internal_path(output)
    if era not in ERAS or output != base/'card_inputs'/(era+'.json.gz'):
        raise ValueError('Use the canonical TT card input for the specified era')
    stages = list(experimental_plans(read_json(base/'nominal.json'), read_json(base/'remainder.json')))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = {}
        for stage, expected in stages:
            plan = read_json(base/(stage+'.json'))
            state = read_json(base/(stage+'.state.json'))
            tasks = plan['tasks']
            selected = [t for t in tasks if t['year'] == era]
            if (plan != expected or not selected or state.get('status') != 'complete'
                    or state.get('failed') != 0 or state.get('complete') != len(tasks)
                    or state.get('total') != len(tasks)):
                raise ValueError('Incomplete TT card inputs: '+stage)
            for t in selected:
                row = state['results'].get(t['input'], {})
                if (row.get('status') != 'complete' or row.get('input') != t['input']
                        or row.get('output') != t['output'] or not (ROOT/t['output']).is_file()):
                    raise ValueError('TT card input receipt mismatch: '+t['input'])
            receipts[stage] = dict(files=len(selected), target_entries=sum(t['target_entries'] for t in selected))
        header = dict(schema='trotasr_complete_tt_card_input_v1', scope='TT_slot_only', year=era,
                      sr_data_blinded=True, top_pt_prescription=PRESCRIPTION, coverage=receipts)
        if output.exists():
            prior = read_json(output)
            if not all(prior.get(k) == v for k,v in header.items()) or prior.get('status') != 'complete':
                raise ValueError('Existing TT-slot product differs: '+era)
            return output
        sections = ('histograms','physical_histograms','dy_rz','dy_rz_variations')
        merged = {k:{} for k in sections}
        for stage, plan in stages:
            jme = stage == 'existing_jme_weights_nominal_pass'
            options = plan['histogram_options']
            for t in plan['tasks']:
                if t['year'] != era:
                    continue
                p = read_json(ROOT/t['output'])
                c = p['contract']
                if (not p.get('sr_data_blinded') or str(c['year']) != era
                        or Path(c['input']) != ROOT/t['input'] or c['shape_mode'] != options['shape_mode']
                        or c.get('variations') != options['variations']
                        or set(c['dataset_replay']['primaries']) != set(options['dataset_primaries'])
                        or (jme and c.get('event_population') != 'corrected_nominal_pass_sr_cr_and_dy_control_v1')):
                    raise ValueError('TT card input population differs: '+t['input'])
                for k in sections[:2]:
                    for mode, variations in p[k].items():
                        dest = merged[k].setdefault(mode,{})
                        for name, regions in variations.items():
                            if not jme and name == 'nominal':
                                continue  # stored non-JME nominal never replaces corrected central
                            if any(set(samples)-{'TT'} for cats in regions.values() for samples in cats.values()):
                                raise ValueError('Non-TT contribution in scoped replay: '+t['input'])
                            merge_histograms(dest.setdefault(name,{}),regions)
                if jme:
                    merge_histograms(merged['dy_rz'],p['dy_rz'])
                merge_histograms(merged['dy_rz_variations'],p['dy_rz_variations'])
        from TROTASR.utils.corrections import NONJME_ENDPOINTS
        for mode in ('highdm','lowdm'):
            needed = {'nominal','jerUp','jerDown','jesTotalUp','jesTotalDown',*ENDPOINTS,*NONJME_ENDPOINTS}
            if not needed <= merged['histograms'][mode].keys():
                raise ValueError('Incomplete detector or top-pT endpoints: '+era+'/'+mode)
        write_json(output,dict(header,status='complete',**merged))
        return output

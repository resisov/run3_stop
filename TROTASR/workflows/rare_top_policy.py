"""Background sample exclusions and the common Top grouping."""
from TROTASR.utils.paths import ERAS

POLICY = dict(excluded_primaries=['TTWH', 'TTZH'],
    top_primaries=['TTW-WtoQQ-1Jets', 'TTZ-ZtoQQ-1Jets', 'TTWW', 'TTWZ', 'TTZZ', 'TTTT'],
    tttt='Top', years=list(ERAS))



def primary(record):
    return record['dataset'].lstrip('/').split('/')[0].split('_Tune')[0]


def excluded_ids(metadata):
    result = {str(k) for k, r in metadata.items() if primary(r) in POLICY['excluded_primaries']}
    if any(metadata[k].get('is_data') or metadata[k].get('is_signal') for k in result):
        raise ValueError('Excluded primary is not background MC')
    for r in metadata.values():
        if r.get('process') == 'TTV' and primary(r) not in POLICY['excluded_primaries'] + POLICY['top_primaries']:
            raise ValueError('Unknown TTV primary; explicit sample classification required')
    return result


def require_clean(payload):
    """No excluded primary or unnormalized rare-Top label reaches consumers."""
    metadata = payload.get('input_metadata', {})
    selected = excluded_ids(metadata)
    # Source inventories retain excluded datasets; receipts identify their removal.
    evidence = payload.get('background_policy', {})
    if selected:
        recorded = {str(k) for row in evidence.get('excluded_files', [])
                    for k in row.get('datasets', {})}
        excluded_primaries = evidence.get('excluded_primaries', evidence.get('policy', {}).get('excluded_primaries'))
        if set(excluded_primaries or ()) != set(POLICY['excluded_primaries']) or selected != recorded:
            raise ValueError('TTWH/TTZH require canonical sample exclusion before analysis')
    for section in ('histograms', 'physical_histograms'):
        for modes in payload.get(section, {}).values():
            for regions in modes.values():
                for categories in regions.values():
                    for samples in categories.values():
                        if {'TTV', 'TTTT', 'TTWH', 'TTZH'} & set(samples):
                            raise ValueError('Common Top policy not applied to histogram samples')
    return payload


def regroup(payload):
    from TROTASR.workflows.merge_nominal import merge_histograms
    for section in ('histograms', 'physical_histograms'):
        for modes in payload[section].values():
            for regions in modes.values():
                for categories in regions.values():
                    for samples in categories.values():
                        for name in ('TTV', 'TTTT'):
                            rare = samples.pop(name, None)
                            if rare is not None:
                                if 'TT' in samples:
                                    merge_histograms(samples['TT'], rare)
                                else:
                                    samples['TT'] = rare

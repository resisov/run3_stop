"""Resolve calibration, configuration and result paths inside TROTASR."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if any(parent.name == 'TROTASR' for parent in ROOT.parents):
    raise RuntimeError('Nested TROTASR analysis copies are retired; use the canonical package')
JSONS = ROOT / 'jsons'
ERA_ALIASES = {'2022pre': '2022', '2022post': '2022EE',
               '2023pre': '2023', '2023post': '2023BPix'}
EARLY_ERAS = ('2022', '2022EE', '2023', '2023BPix')
ERAS = (*EARLY_ERAS, '2024', '2025')


def analysis_era(year):
    """Return the target analysis era independently of the source sample year."""
    era = ERA_ALIASES.get(str(year), str(year))
    if era not in ERAS:
        raise ValueError('Unknown analysis era: ' + str(year))
    return era


def era_year(year):
    """Return the era key used in the event schema."""
    era = analysis_era(year)
    return int(era) if era in ('2024', '2025') else era


def muon_pt_min(year):
    """Minimum loose/medium muon pT in GeV for the target analysis era."""
    return 15.0 if analysis_era(year) in EARLY_ERAS else 10.0


def luminosity_fb(year):
    """Use the same target-era luminosity as the event normalization."""
    import math
    from .io import read_json
    value = float(read_json(ROOT / 'estimations' / ('normalization_' + analysis_era(year) + '.json.gz'))['luminosity_fb'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid normalization luminosity')
    return value


def tagging_definition(year):
    era = analysis_era(year)
    early = era in EARLY_ERAS
    loose, medium, top, w = {
        '2022': (.047, .245, .6265, .9473),
        '2022EE': (.0499, .2605, .6416, .9478),
        '2023': (.0358, .1917, .5938, .9473),
        '2023BPix': (.0359, .1919, .5947, .9482),
        '2024': (.0246, .1272, .5078, .9385),
        '2025': (.0246, .1272, .5078, .9385),
    }[era]
    return dict(era=era, btag_branch='jet_btag_pnet_all' if early else 'jet_btag_upart_all',
                btag='particleNet' if early else 'UParTAK4', loose=loose, medium=medium,
                topw='ParticleNetWithMass' if early else 'GlobalParT3', top=top, w=w,
                resolved_wp=.973943293094635 if early else .9433798789978027,
                mixed_wp=.9420878291130066 if early else .9027690887451172)


def internal_path(path):
    """Reject external paths and retired execution namespaces before I/O.

    Resolve symlinks and parent-directory components before checking scope.
    """
    resolved = Path(path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("Path escapes TROTASR: " + str(path))
    relative=resolved.relative_to(ROOT)
    if relative.parts and relative.parts[0] in ('campaigns','validation'):
        raise ValueError('Retired execution namespace; use current canonical products: '+str(path))
    if relative.parts and relative.parts[0]=='history' and resolved.suffix in ('.py','.sh'):
        raise ValueError('Historical source is not a runtime implementation')
    return resolved






def payload_path(relative, root=ROOT):
    """Resolve a current payload path, normalizing only analysis-era names."""
    if Path(root).resolve() != ROOT:
        raise ValueError('Payload root differs from the analysis package')
    path = Path(relative)
    if path.is_absolute():
        return internal_path(path)
    parts = list(path.parts)
    if len(parts) >= 3 and parts[0] == 'scales':
        parts[2] = ERA_ALIASES.get(parts[2], parts[2])
    return internal_path(ROOT.joinpath(*parts))

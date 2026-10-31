"""One payload resolver. Paths are relative to TROTASR, never a campaign."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if any(parent.name == 'TROTASR' for parent in ROOT.parents):
    raise RuntimeError('Nested TROTASR analysis copies are retired; use the canonical package')
for canonical in (Path('/hep2-scratch/twkim/postprocessing/TROTASR'),
                  Path('/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR')):
    if canonical.is_dir() and ROOT != canonical.resolve():
        raise RuntimeError('Only the canonical TROTASR package is allowed: '+str(canonical))
JSONS = ROOT / 'jsons'
CONFIG_FILENAMES = frozenset({
    'an_figure_inventory.json', 'assets.json', 'background_sources.json',
    'completion_contract.json', 'diagnostic_sources.json', 'highdm_sr_binning.json',
    'nominal_execution_sources.json', 'nominal_plot_sources.json',
    'normalization_recovery_sources.json', 'prediction_plot_sources.json',
    'region_schematic_sources.json', 'selection_contract.json', 'sources.json',
    'sr_category_merge.json', 'statistics_sources.json', 'systematics_plan.json',
})
POG_NAMES = {"JMESF": "JME", "BTVSF": "BTV", "EGammaSF": "EGM",
             "MuonSF": "MUO", "PUweight": "LUM", "AnalysisSF": "Private",
             "lumiMask": "LUM"}
ERA_ALIASES = {'2022pre': '2022', '2022post': '2022EE',
               '2023pre': '2023', '2023post': '2023BPix'}
EARLY_ERAS = ('2022', '2022EE', '2023', '2023BPix')
ERAS = (*EARLY_ERAS, '2024', '2025')


def analysis_era(year):
    """Use the target campaign, never the NanoAOD production year."""
    era = ERA_ALIASES.get(str(year), str(year))
    if era not in ERAS:
        raise ValueError('Unknown analysis era: ' + str(year))
    return era


def era_year(year):
    """Retain the existing integer contract for the 2024/25 campaigns."""
    era = analysis_era(year)
    return int(era) if era in ('2024', '2025') else era


def muon_pt_min(year):
    """User-adopted loose/medium muon threshold, 2026-10-20."""
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
                mixed_wp=.9420878291130066 if early else .9027690887451172,
                resolved_sha256=('a5f59357291b178b4c7486b0781e2edbd78005a02d921ac3e18b6897f7f060a2'
                    if early else 'ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3'),
                mixed_sha256=('3208e7455892f608b6dde745ba591632f519035f56bb17d5656e9d882aadbbf1'
                    if early else '40469a1a62f64b08b4225682a2df6ac2538f49a1e7eef5ef4bdff857d0b875a4'))


def internal_path(path):
    """Reject external paths, including symlink/.. escapes, before any I/O.

    No external-input exceptions are currently registered. Approval to inspect
    a legacy source during development does not grant runtime access to it.
    """
    resolved = Path(path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("Path escapes TROTASR; explicit approval required: " + str(path))
    relative=resolved.relative_to(ROOT)
    if relative.parts and relative.parts[0] in ('campaigns','validation'):
        raise ValueError('Retired execution namespace; use current canonical products: '+str(path))
    if relative.parts and relative.parts[0]=='history' and resolved.suffix in ('.py','.sh'):
        raise ValueError('Historical source is not a runtime implementation')
    # Frozen manifests retain their original strings and content hashes. Only
    # these explicitly relocated root-level configuration names have an alias;
    # arbitrary missing files, campaign states and outputs never do.
    if resolved.parent == ROOT and resolved.name in CONFIG_FILENAMES:
        if resolved.exists():
            raise ValueError('Root-level configuration must be moved to jsons: ' + str(resolved))
        relocated = (JSONS / resolved.name).resolve()
        if JSONS.resolve() != JSONS or relocated.parent != JSONS:
            raise ValueError('Configuration relocation escapes jsons: ' + str(path))
        return relocated
    return resolved


def internal_modules(*names):
    """Do not let legacy renderer aliases bind to an external analysis package."""
    import sys
    for name, module in tuple(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + '.') for prefix in names):
            origin = getattr(module, '__file__', None)
            if origin is None:
                raise ValueError('Analysis module has no verifiable internal origin: ' + name)
            internal_path(origin)
            for folder in getattr(module, '__path__', ()):
                internal_path(folder)


def payload_path(relative, root=ROOT):
    """Resolve a legacy physics payload key to the new layout, without symlinks."""
    root, p = Path(root).resolve(), Path(relative)
    if root != ROOT:
        raise ValueError("Payload root must be this TROTASR")
    parts = list(p.parts)
    if p.is_absolute():
        try:
            parts = list(p.relative_to(root).parts)
        except ValueError:
            raise ValueError("Payload escapes TROTASR: " + str(p))
    if parts and parts[0] == "analysis":
        parts.pop(0)
    if parts and parts[0] == "data":
        parts.pop(0)
        if parts and parts[0] in POG_NAMES:
            parts = ["scales", POG_NAMES[parts[0]], *parts[1:]]
        else:
            parts = ["utils", "compiled", *parts]
    elif parts and parts[0] == "hists":
        parts = ["estimations", *parts[1:]]
    if len(parts) >= 3 and parts[0] == 'scales':
        parts[2] = ERA_ALIASES.get(parts[2], parts[2])
    return internal_path(root.joinpath(*parts))

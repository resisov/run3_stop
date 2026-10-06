"""One payload resolver. Paths are relative to TROTASR, never a campaign."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def internal_path(path):
    """Reject external paths, including symlink/.. escapes, before any I/O.

    No external-input exceptions are currently registered. Approval to inspect
    a legacy source during development does not grant runtime access to it.
    """
    resolved = Path(path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("Path escapes TROTASR; explicit approval required: " + str(path))
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
    return internal_path(root.joinpath(*parts))

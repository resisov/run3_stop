"""Target-era model routing, independent of the input production campaign."""
from TROTASR.utils.paths import ROOT, analysis_era, EARLY_ERAS
from TROTASR.utils.io import read_json, sha256


def configuration_path(year):
    name = 'early_run3_2022/config.json' if analysis_era(year) in EARLY_ERAS else 'config.json'
    return ROOT / 'gnn4lowdm' / name


def configuration(year):
    return read_json(configuration_path(year))


def portable_model(year):
    if analysis_era(year) not in EARLY_ERAS:
        return ROOT / 'gnn4lowdm/diagonal_v3_numpy.npz'
    config = configuration(year)
    path = ROOT / config['model']['portable_numpy']
    if sha256(path) != config['model']['portable_numpy_sha256']:
        raise ValueError('Early-era portable model differs from the adopted checkpoint export')
    return path


def global_feature_names(year):
    from .features import DIAGONAL_V3_GLOBAL_FEATURE_NAMES
    return tuple(n for n in DIAGONAL_V3_GLOBAL_FEATURE_NAMES
                 if analysis_era(year) not in EARLY_ERAS or n != 'lowdm_isr_subjet_btag_max')

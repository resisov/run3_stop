"""Canonical high-dM control likelihood; diagnostic tail bins are not extra data."""

FIT_GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6, 7))
DIAGNOSTIC_GROUPS = ((0,), (1,), (2,), (3,), (4, 5), (6, 7))
N_HIGH_CR = 2 * len(FIT_GROUPS)
N_BINS = 118 + 30 + 3 * (N_HIGH_CR + 10)
# Relative to the frozen model only ll and qcd lose a tail normalization per Nb.
# The photon transfer already uses the same >=500 parameter.
N_RATES = 44

def fit_group(native):
    if not 0 <= native < 8:
        raise ValueError('Unknown native high-dM recoil bin')
    return next(i for i, group in enumerate(FIT_GROUPS) if native in group)

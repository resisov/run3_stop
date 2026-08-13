# 2024 Top/W-split lost-lepton closure

- Top definition: `TT + ST`
- W definition: `WtoLNu`
- Selection authority: `real_subset_worker.py`
- Nominal intermediates modified: `false`

## Control-region scale factors

| Regime | Top | W+jets | Correlation |
|---|---:|---:|---:|
| highdm | 0.8823 ± 0.0101 | 1.0504 ± 0.0175 | -0.659 |
| lowdm | 0.9514 ± 0.0178 | 0.7300 ± 0.0086 | -0.083 |

## Data residual closure

| Validation region | Combined ratio | Top/W ratio | Top/W p-value | max abs. pull |
|---|---:|---:|---:|---:|
| highdm_nb0 | 0.6526 | 0.6539 | 1.6e-75 | 14.510 |
| highdm_njet3to4_nb1plus | 0.6757 | 0.6794 | 4.001e-268 | 19.645 |
| lowdm_isr200to300 | 0.8966 | 0.8001 | 4.234e-42 | 10.227 |
| lowdm_met250to300 | 0.8417 | 0.7763 | 1.436e-36 | 10.577 |
| lowdm_significance7to10 | 1.0197 | 0.9911 | 0.969 | 0.425 |

## Independent CR diagnostic

| High-dM control category | Integrated ratio | p-value | max abs. pull |
|---|---:|---:|---:|
| highdm_nb0 | 1.0000 | 0.0007951 | 3.554 |
| highdm_njet3to4_nb1plus | 1.0000 | 0.4832 | 1.470 |

The low-dM `Nb=1` one-lepton category was excluded from the two-component normalization fit.

- Predicted / residual data in the excluded `Nb=1` category: `0.7878`

Only statistical covariance is propagated here. Detector/model systematics and a possible adopted nonclosure nuisance are not included.

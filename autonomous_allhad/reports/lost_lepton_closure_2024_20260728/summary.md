# 2024 Lost-lepton closure

- Input status: `complete`
- Duplicate audit: `complete_no_duplicates`
- ROOT files: 1153
- Events scanned: 361,054,245
- Events retained for closure/VR: 34,735,958
- Target: `TT + WtoLNu + ST`
- Fold split: event-level stable SplitMix64 hash; file/shard splitting is not used.
- Selection authority: `real_subset_worker.py`.

## Independent MC closure

| Scheme | Valid bins | diagonal chi2/ndf | p-value | max |pull| |
|---|---:|---:|---:|---:|
| highdm_met | 6 | 0.00/6 | 1 | 0.0536 |
| highdm_search60 | 29 | 0.09/29 | 1 | 0.173 |
| lowdm_search42 | 36 | 0.14/36 | 1 | 0.206 |

## Full-mixture MC pseudodata closure

- `highdm_met`: 6 valid bins, p=0.9999999999700366, max |pull|=0.033455846125660486
- `highdm_search60`: 29 valid bins, p=1.0, max |pull|=0.12459422705633237
- `lowdm_search42`: 36 valid bins, p=1.0, max |pull|=0.1801965260169732

## Data validation regions

| Validation region | Valid bins | MC LL purity (1l / 0l) | p-value | max |pull| |
|---|---:|---:|---:|---:|
| highdm_nb0 | 6 | 0.928 / 0.425 | 2.0164599103853907e-77 | 12.211347893576535 |
| highdm_njet3to4_nb1plus | 7 | 0.967 / 0.630 | 7.183137686740301e-264 | 19.710901295269018 |
| lowdm_met250to300 | 2 | 0.927 / 0.451 | 4.1910622016112095e-21 | 7.965706968063002 |
| lowdm_isr200to300 | 4 | 0.926 / 0.411 | 6.97973226297385e-14 | 5.640021701528244 |
| lowdm_significance7to10 | 3 | 0.937 / 0.422 | 0.9449775319538594 | 0.5318583799756609 |

Statistical covariance is diagonal because one original event contributes to one bin in each displayed categorization. Detector/model systematics and any adopted nonclosure nuisance are separate from this statistical covariance.

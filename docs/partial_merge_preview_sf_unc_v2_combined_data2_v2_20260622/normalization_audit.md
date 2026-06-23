# Normalization Audit: sf_unc_v2_combined_data2_v2_20260622

Created UTC: `2026-06-23T09:06:05Z`

## Verdict
- MC split-record normalization now passes formula checks at physical-dataset level.
- The remaining large discrepancy came from DATA stream aggregation: EGamma/Muon/JetMET were all counted in every region.
- Region totals, plots, and search-bin data now use the stop-processor-style primary data stream per region.

## Region Data/MC
| Region | DATA stream | DATA used | Background | Data/MC | Excluded DATA streams | Unfiltered Data/MC before fix |
|---|---:|---:|---:|---:|---:|---:|
| LLCR | JetMET | 31980.000 | 41000.902 | 0.780 | EGamma:9937, Muon:16053 | 1.414 |
| QCDCR | JetMET | 30377.000 | 33923.240 | 0.895 | EGamma:6655, Muon:12748 | 1.467 |
| GCR | EGamma | 9907.000 | 10183.035 | 0.973 | JetMET:6772, Muon:541 | 1.691 |
| DY2E | EGamma | 351.000 | 412.325 | 0.851 | JetMET:287, Muon:31 | 1.623 |
| DY2M | Muon | 372.000 | 587.196 | 0.634 | EGamma:27, JetMET:457 | 1.458 |
| SR | JetMET | 36664.000 | 34859.747 | 1.052 | EGamma:3625, Muon:8028 | 1.386 |

## MC Factor Checks
- Physical MC datasets: `65`
- Per-split records: `5170`
- Blocked datasets: `0`
- Max factor formula relative error: `0.0`
- Xsec conflicts: `0`
- Sumw sources: `{'Runs.genEventSumw': 35500}`

## Search-Bin DATA Processes
- `ak8_kinematics_no_tag_scores`: `['JetMET']`
- `isr_sensitive`: `['JetMET']`
- `minimal_njet_nb_met`: `['JetMET']`
- `optimized_hybrid_no_tags`: `['JetMET']`
- `resolved_kinematics`: `['JetMET']`

## Top SR MC Physical Datasets
| Process | SR yield | Splits | Dataset |
|---|---:|---:|---|
| TT | 22936.931 | 45 | `TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| Zto2Nu | 2968.060 | 8 | `Zto2Nu-2Jets_Bin-2J-PTNuNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| WtoLNu | 2238.380 | 10 | `WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| TT | 1655.955 | 57 | `TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3` |
| QCD | 715.452 | 68 | `QCD_Bin-PT-170to300_TuneCP5_13p6TeV_pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| Zto2Nu | 658.148 | 6 | `Zto2Nu-2Jets_Bin-2J-PTNuNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| ST | 581.699 | 26 | `TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| ST | 581.236 | 26 | `TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| WtoLNu | 420.616 | 5 | `WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| QCD | 330.068 | 63 | `QCD_Bin-PT-300to470_TuneCP5_13p6TeV_pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| Zto2Nu | 254.151 | 24 | `Zto2Nu-2Jets_Bin-2J-PTNuNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` |
| WtoLNu | 180.109 | 327 | `WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3` |

## Files
- JSON: `normalization_audit.json`
- Source payload: `partial_normalized_yields.json`

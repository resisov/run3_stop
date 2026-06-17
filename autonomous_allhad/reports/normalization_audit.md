# Normalization Audit

Feature-side MC yields are currently raw genWeight sums, not luminosity/xsec normalized.

Luminosity used: 109.82 fb^-1 (109820.0 pb^-1).

Legacy stop_processor_v4.py validation is external/manual. No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad unless explicitly provided by the user.

## Answers

- Data yields are left unscaled.
- Negative weights are handled through signed genWeight sums.
- Signal samples are normalized per SMS dataset/mass point when xs and processed sumw are available.
- Bad/skipped files are excluded from the feature-subset denominator because only processed feature-table rows are used.
- Full dataset sumw is missing from metadata, so full production normalization is not claimed.

## Dataset Factors Preview

| Dataset | Process | xs pb | processed sumw | status | factor |
|---|---|---:|---:|---|---:|
| `DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3____1_` | DY | 0.08108 | 8488.58 | normalized_from_processed_feature_sumw | 1.048962999978074 |
| `EGamma0-Run2024C-MINIv6NANOv15-v1____1_` | EGamma | -1 | 24000 | data_unscaled | 1.0 |
| `GJ_Bin-PTG-600_TuneCP5_13p6TeV_amcatnlo-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | GJ | 0.576 | 19984.9 | normalized_from_processed_feature_sumw | 3.165202056444699 |
| `JetMET0-Run2024C-MINIv6NANOv15-v1____1_` | JetMET | -1 | 24000 | data_unscaled | 1.0 |
| `Muon0-Run2024C-MINIv6NANOv15-v1____1_` | Muon | -1 | 24000 | data_unscaled | 1.0 |
| `QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV_pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | QCD | 9.306 | 24000 | normalized_from_processed_feature_sumw | 42.582705 |
| `SMS-2Stop_Par-mStop-1000_TuneCP5_13p6TeV_madgraph-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | SMS | 0.009123 | 1985 | normalized_from_processed_feature_sumw | 0.5047294005037783 |
| `SMS-2Stop_Par-mStop-1500_TuneCP5_13p6TeV_madgraph-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` | SMS | 0.0003912 | 24000 | normalized_from_processed_feature_sumw | 0.001790066 |
| `SMS-2Stop_Par-mStop-600_TuneCP5_13p6TeV_madgraph-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2` | SMS | 0.256 | 21 | normalized_from_processed_feature_sumw | 1338.7580952380954 |
| `TBbarQto2Q-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | ST | 77.26 | 1.85764e+06 | normalized_from_processed_feature_sumw | 4.567457576759456 |
| `TTTT_TuneCP5_13p6TeV_amcatnlo-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | TT | 0.009652 | 231.349 | normalized_from_processed_feature_sumw | 4.581738540746771 |
| `WW_TuneCP5_13p6TeV_pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | VV | 80.23 | 24000.2 | normalized_from_processed_feature_sumw | 367.11679942078723 |
| `WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | WtoLNu | 0.07753 | 21227 | normalized_from_processed_feature_sumw | 0.4011089019838839 |
| `Zto2Nu-2Jets_Bin-1J-PTNuNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8-RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2____1_` | Zto2Nu | 0.01895 | 5253.9 | normalized_from_processed_feature_sumw | 0.396103533510644 |

## Missing

- full dataset sumw in metadata
- per-file Runs sumw bookkeeping for all files
- pileup/btag/lepton/photon/trigger correction weights
- systematic shifted weights
- full dataset processing rather than deterministic subset chunks

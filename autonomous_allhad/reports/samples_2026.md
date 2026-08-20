# 2026 data-skim readiness

Machine-readable source: `autonomous_allhad/configs/samples_2026.json`.

## Frozen data input

The available 2026 PromptReco NanoAOD data currently comprise 32 datasets:
`JetMET0/1`, `EGamma0/1/2/3`, and `Muon0/1` for Run2026A--D, all
`PromptReco-v1`.

- 14,295 unique files;
- 11.559 TB of NanoAOD input;
- 9,318,982,316 DBS-reported events across the split primary datasets;
- no duplicate LFNs;
- input-list SHA256:
  `2f171733ba3b4684139fd870db212124f90a4ff569fe161475b38cefa005fdae`.

The event total is processing volume, not a unique collision-event count,
because the split primary datasets contain overlapping collision content by
design.

The exact data GlobalTag was recovered from NanoAOD provenance:
`160X_dataRun3_Prompt_v1`.

## Schema check

A representative Run2026A JetMET0 NanoAOD contains all branches required by
the current full-selection data feature schema. The MET HLT paths and the
standard MET-filter branches are present. No schema blocker was found.

## Available official corrections

The current correction era is `Run3-26Prompt-Summer24-NanoAODv15`.

- Golden JSON: available through run 403937.
- AK4/AK8 PUPPI JEC: `Summer24Prompt26_V1`, including data residual compounds.
- JER: separate `RunBD` and low-pileup `RunC` tags. JER is MC-only and does not
  block the data skim.
- Jet ID: AK4/AK8 PUPPI Tight and TightLeptonVeto.
- Jet veto map: `Summer24Prompt26_RunBCD_V1`.
- Electron and photon scale/smearing: available with 2026 correction keys.

## Missing official corrections and SFs

The 2026 release is not yet complete:

- no MUO momentum scale/smearing payload;
- no electron ID/reco SF payload;
- no photon ID/CSEV/pixel-veto SF payload;
- no BTV SF payload;
- no LUM pileup payload;
- no tau payload;
- no MET XY payload;
- no analysis-specific MET, photon, electron, or muon trigger SFs;
- no analysis-specific 5--10 GeV electron/muon SFs;
- no 2026 b-tag efficiency payload.

Only the muon momentum correction is a direct blocker for a fully corrected
**data** skim. Multiplicative SFs and pileup are MC histogram weights and do
not belong in data skimming. A separate 2026 b-tag-efficiency measurement will
be needed after the compatible 2026 MC sample policy is frozen.

## Production decision

The data list, GlobalTag, Golden JSON, branch schema, jet corrections, and EGM
scale corrections are prepared. Condor production is intentionally not
prepared yet: using an implicit unity muon correction would be an unreviewed
physics choice. When the official MUO payload appears, the campaign can be
completed without rediscovering the inputs.

Sources: [CMS correction documentation](https://cms-analysis-corrections.docs.cern.ch/),
[CMS Data Quality](https://twiki.cern.ch/twiki/bin/view/CMSPublic/DataQuality),
CMS DAS/DBS, CMS Rucio, and the live CMS correction CVMFS tree.

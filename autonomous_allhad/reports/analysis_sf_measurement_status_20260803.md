# Analysis-owned SF measurement and integration status — 2026-08-04

## Physics definitions frozen from AN2019-016

- MET trigger: reference-trigger efficiency with the adopted single-electron
  reference measurement, applied to every MET-triggered MC process.
- Photon trigger: `Photon175 || Photon200` measured against an independent
  JetMET/PFHT denominator.  This is not a lepton tag-and-probe measurement.
- Low-pT veto electron: resonance tag-and-probe for the full analysis target
  `cutBased >= Veto && miniPFRelIso_all < 0.1` in `5 < pT < 10 GeV`.
- Low-pT loose muon: J/psi tag-and-probe for mini-isolation conditional on
  LooseID, multiplied by the official 2024 J/psi LooseID SF.  The installed
  payload is the combined analysis loose-muon SF.

## Workflow layout

- `workflow/met_trigger_measurement/`
- `workflow/photon_trigger_measurement/`
- `workflow/lowpt_electron_measurement/`
- `workflow/lowpt_muon_measurement/`

Each directory contains its own frozen 2024 config, measurement or fit entry
points, adoption gates, and correctionlib exporter.  Shared deterministic
counting, pass/fail fit, and payload code is under `workflow/`.

## Adopted measurements

All four final results are now explicitly adopted.  Their correctionlib v2
payloads are installed at:

- `analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz`
- `analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz`
- `analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz`
- `analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz`

## Trigger integration status

The post-skim histogram weight bundle has four analysis-owned correction
components and Up/Down variations:

- `met_trigger`
- `photon_trigger`
- `veto_electron_5to10`
- `loose_muon_5to10`

Histogram production now requires all four components by default, so any
missing or invalid adopted payload fails closed.  Nominal weights contain the
four measured SFs.  Histogram and search-bin outputs contain
`met_triggerUp/Down`, `photon_triggerUp/Down`,
`veto_electron_5to10Up/Down`, and `loose_muon_5to10Up/Down`.

The MET payload is the single-electron-reference measurement requested for the
2024 campaign and is applied to every MET-triggered MC process.  QCD no longer
requests a nonexistent QCD-only payload and therefore no longer silently falls
back to unity.  Photon-trigger weights cover both high- and low-DeltaM GCRs.

Representative histogram builds completed for Zto2Nu SR, QCD QCDCR, and GJ
GCR, with distinct trigger Up/Down integrals.  Exact audit numbers and payload
hashes are in `workflow/trigger_sf_integration_validation/summary.json`.

For selected objects with `5 < pT < 10 GeV`, the official electron-veto and
loose-muon 10 GeV edge-clipped contributions are replaced by unity before the
new low-pT SF is multiplied.  Therefore the official edge value and the new
measurement are never double counted.  At exactly 10 GeV the analysis-owned
low-pT component is unity and the official payload takes over.

Correctionlib exporters install only results whose status is exactly
`adopted`; preliminary and validation-pending results are rejected.

## Remaining production work

The four measurements and code integration are complete.  Existing nominal
histogram campaigns produced before this integration must not be reused:
the chunk execution contract now hashes `analysis_scale_factors.py` and all
four payloads, forcing regeneration when any of them changes.

## Validation performed

- correctionlib v2 round-trip and deterministic gzip writing;
- refusal to install non-adopted results;
- MET/photon one- and two-dimensional count flattening;
- analysis payload evaluation for nominal/Up/Down;
- exact low-pT electron and muon target selections on synthetic probes;
- simultaneous pass/fail fit recovery on a synthetic J/psi spectrum;
- official J/psi LooseID multiplication and uncertainty propagation.
- explicit 5.001, 9.999, and 10.000 GeV boundary tests proving removal of the
  old edge contribution and the exact official-payload handoff;
- production histogram-leaf filling for all eight new Up/Down variations;
- seven analysis-SF tests and seven shape-histogram tests in the EOS Python 3.8
  runtime, all passing;
- a 59,458-event real flat-shard histogram build with all four components
  applied, no weight failures, and all eight variation leaves present.

The complete machine-readable integration audit is
`workflow/analysis_sf_integration_validation/summary.json`.

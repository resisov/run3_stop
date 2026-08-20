# Analysis-owned SF measurement and integration status — 2026-08-18

## Physics definitions frozen from AN2019-016

- MET trigger: reference-trigger efficiency with the adopted single-electron
  reference measurement, applied to every MET-triggered MC process.
- Photon trigger: `Photon175 || Photon200` measured against an independent
  JetMET/PFHT denominator.  This is not a lepton tag-and-probe measurement.
- Low-pT veto electron: J/psi tag-and-probe for `cutBased >= Veto` only in
  `5 < pT < 10 GeV`; mini-isolation is excluded from pass/fail.
- Low-pT loose muon: J/psi tag-and-probe for LooseID relative to tracker
  muons in `5 < pT < 10 GeV`; mini-isolation is excluded from pass/fail.

## Workflow layout

- `workflow/met_trigger_measurement/`
- `workflow/photon_trigger_measurement/`
- `workflow/lowpt_electron_measurement/`
- `workflow/lowpt_muon_measurement/`

Each directory contains its own frozen 2024 config, measurement or fit entry
points, adoption gates, and correctionlib exporter.  Shared deterministic
counting, pass/fail fit, and payload code is under `workflow/`.

## Installed measurements

The trigger results and the latest ID-only low-pT lepton results are installed
as correctionlib v2 payloads at:

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

The installed low-pT payload correction keys are
`veto_electron_id_5to10_sf` and `loose_muon_id_5to10_sf`.  The older combined
ID-plus-mini-isolation correction contents have been replaced.

## Remaining production work

The four measurements and code integration are complete.  Existing nominal
histogram campaigns produced before the ID-only payload replacement must not be reused:
the chunk execution contract now hashes `analysis_scale_factors.py` and all
four payloads, forcing regeneration when any of them changes.

## Validation performed

- correctionlib v2 round-trip and deterministic gzip writing;
- refusal to install non-adopted results;
- MET/photon one- and two-dimensional count flattening;
- analysis payload evaluation for nominal/Up/Down;
- exact low-pT electron and muon ID-only selections on synthetic probes;
- simultaneous pass/fail fit recovery on a synthetic J/psi spectrum;
- installed correction-name and description checks proving isolation is excluded;
- explicit 5.001, 9.999, and 10.000 GeV boundary tests proving removal of the
  old edge contribution and the exact official-payload handoff;
- production histogram-leaf filling for all eight new Up/Down variations;
- eight analysis-SF tests and seven ID-only payload tests, all passing;
- the earlier 59,458-event representative histogram predates the ID-only
  replacement and is marked stale; the changed execution hash forces it to be rebuilt.

The complete machine-readable integration audit is
`workflow/analysis_sf_integration_validation/summary.json`.

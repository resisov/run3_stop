# 2024 MET trigger efficiency measurement preparation

Status: **prepared; physics result not yet measured**. This is a physics proposal and does not alter the baseline processor or event selection.

## Measurement definition

Measure the OR of the six `PFMET[NoMu]{120,130,140}_PFMHT[NoMu]{120,130,140}_IDTight` paths in an electron-triggered data sample. The denominator is selected without the probe trigger. Apply the golden JSON, MET filters, exactly one medium electron, zero loose muons, at least two good AK4 jets, HT above 300 GeV, and the established angular/quality selection. The primary observable is corrected PuppiMET for 2024.

The independent reference OR is `Ele{30,32,35,38,40}_WPTight_Gsf`. Before production, verify from run-dependent HLT/prescale information that accepted reference paths are present and unprescaled. Missing requested HLT branches are a hard measurement failure, not a silently skipped path. Deduplicate EGamma0/EGamma1 with `(run, luminosityBlock, event)`.

The frozen configuration is
`autonomous_allhad/workflow/met_trigger_measurement/config_2024.json`.

## Unified executable

All MET/photon trigger-measurement operations use exactly one executable:
`autonomous_allhad/workflow/measure_trigger.py`.  The available subcommands are
`build-records`, `prepare`, `count`, `recover`, `reduce`, and `export`.

Example reduction:

```bash
python autonomous_allhad/workflow/measure_trigger.py reduce \
  --measurement met_genuine \
  --input-dir /eos/user/t/taiwoo/.../shard_outputs \
  --config autonomous_allhad/workflow/met_trigger_measurement/config_2024.json \
  --output autonomous_allhad/outputs/trigger_efficiency/2024/result.json
```

Data intervals are exact Clopper–Pearson intervals. Weighted MC uses denominator effective entries and a Wilson interval; bins with non-positive total weight or an efficiency outside `[0,1]` are invalidated. Data/MC scale-factor uncertainties are propagated independently. Negative-weight samples require a bootstrap or dedicated covariance cross-check before adoption.

## Acceptance checklist

- Produce inclusive, per-era, and per-run efficiencies and check stability.
- Repeat versus electron eta, Njet, Nb, NPV, HT, and probe-path components.
- Demonstrate reference-trigger independence/exclusivity and prescale handling.
- Require at least 100 denominator data events per plateau bin and a 98% minimum efficiency with its lower interval no more than 2% below the central value.
- Compare corrected PuppiMET, PF MET, and no-muon recoil definitions; freeze the variable used by the SR.
- Validate data/MC closure and assign residual shape, era, reference-trigger, and limited-statistics uncertainties.
- Export machine-readable counts, efficiencies, covariance/uncertainties, plots, input manifest, config hash, and git commit.
- Do not apply a scale factor to FastSim or the nominal analysis until plateau and closure checks pass.

# 2024 MET trigger efficiency measurement preparation

Status: **prepared; physics result not yet measured**. This is a physics proposal and does not alter the baseline processor or event selection.

## Measurement definition

Measure the OR of the six `PFMET[NoMu]{120,130,140}_PFMHT[NoMu]{120,130,140}_IDTight` paths in an electron-triggered data sample. The denominator is selected without the probe trigger. Apply the golden JSON, MET filters, exactly one medium electron, zero loose muons, at least two good AK4 jets, HT above 300 GeV, and the established angular/quality selection. The primary observable is corrected PuppiMET for 2024.

The independent reference OR is `Ele{30,32,35,38,40}_WPTight_Gsf`. Before production, verify from run-dependent HLT/prescale information that accepted reference paths are present and unprescaled. Missing requested HLT branches are a hard measurement failure, not a silently skipped path. Deduplicate EGamma0/EGamma1 with `(run, luminosityBlock, event)`.

The frozen configuration is `autonomous_allhad/configs/trigger_efficiency_2024.yaml`.

## Required event-stage output

The reducer accepts a JSON object with `measurement`, `bin_edges_gev`, and equal-length `data` and `mc` arrays. Each data bin contains integer `total` and `passed`; each MC bin contains `sumw_total`, `sumw2_total`, and `sumw_passed`. Example:

```json
{"measurement":"met_or_2024","bin_edges_gev":[250,300,400],
 "data":[{"total":100,"passed":98},{"total":80,"passed":80}],
 "mc":[{"sumw_total":95,"sumw2_total":100,"sumw_passed":92},
       {"sumw_total":78,"sumw2_total":82,"sumw_passed":77}]}
```

Run:

```bash
python autonomous_allhad/workflow/measure_trigger_efficiency.py counts.json \
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

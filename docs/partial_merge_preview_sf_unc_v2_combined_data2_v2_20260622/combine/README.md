# Partial Combine Inputs: sf_unc_v2_combined_data2_v2_20260622

Generated from `docs/partial_merge_preview_sf_unc_v2_combined_data2_v2_20260622/fit_template_summary.json` and FastSim signal histograms/yields.

## Contents

- `templates_stop_2024_partial.root`: Combine shape template ROOT file.
- `datacards/datacard_<mass>.txt`: 352 per-mass datacards.
- `run_combine_expected.sh`: runs `combine -M AsymptoticLimits --run blind` for every mass point.
- `combine_input_manifest.json`: machine-readable provenance and model policy.
- `expected_limits.json`: collected Combine outputs; currently empty until Combine is run.

## Model Policy

- Channels: cat2 LLCR, cat3 QCDCR, cat4 GCR, cat5 DY2E, cat6 DY2M, cat7 SR.
- Background is an aggregate template per channel.
- Signal is included in LLCR, QCDCR, and SR where FastSim signal histograms are available; GCR/DY control regions are background-only.
- `BkgSyst` is an aggregate non-lumi background shape nuisance from `fit_template_summary.json`.
- `Lumi_2024 lnN 1.016` is applied to MC signal and background.
- `SignalTheory lnN` stores the signal cross-section uncertainty from `signal_yields_by_mass.json`.
- `autoMCStats 10` is enabled.
- `data_obs` is Asimov background because this is an expected-limit setup.

## Run Limits

Set up a Combine environment so that `combine` is in `PATH`, then run:

```bash
analysis/combine/partial_merge_sf_unc_v2_combined_data2_v2_20260622/run_combine_expected.sh
```

After Combine finishes, collect the limits and draw the contour with:

```bash
python3 autonomous_allhad/workflow/build_combine_inputs_from_preview.py   --output-dir analysis/combine/partial_merge_sf_unc_v2_combined_data2_v2_20260622   --collect-only
```

At generation time, `combine` and `text2workspace.py` were not available in `PATH`, so no real Combine limit or contour is claimed yet.

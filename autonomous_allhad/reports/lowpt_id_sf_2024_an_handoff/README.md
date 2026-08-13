# 2024 low-pT ID scale-factor plot handoff

Local plot copies for the analysis-note workflow.

- `electron/`: veto-electron ID-only efficiency, scale-factor, heatmap, and six pass/fail mass-fit figures.
- `muon/`: loose-muon ID-only efficiency, scale-factor, heatmap, and three pass/fail mass-fit figures.
- Both PNG and PDF formats are provided with each channel's `plot_manifest.json`.
- Figure label: `CMS Work in progress`, `2024 (13.6 TeV)`.
- Measurement status is `validation_pending`; the report records the need for future POG validation.
- `lowpt_id_sf_measurement_report_2024.pdf`: AN-ready physics report covering the ParkingSingleMuon data, double-J/psi simulation, trigger-bias strategy, tag-and-probe definitions, fit likelihood, uncertainties, results, and POG-validation outlook.
- `lowpt_id_sf_measurement_report_2024.tex`: editable report source for the AN workflow.
- `payloads/veto_electron_id_5to10_sf_candidate.json.gz`: validation-pending electron ID-only correctionlib candidate.  Its highest-$|\eta|$ bins return nominal 1.0 with measurement-derived symmetric uncertainties.
- `payloads/loose_muon_id_5to10_sf_candidate.json.gz`: validation-pending muon ID-only correctionlib candidate.

Neither candidate has been installed in `analysis/data`; both await the independent validation step.

Canonical plotting implementation: `autonomous_allhad/workflow/plot_measurement.py`.

# Final canonical TROTASR

User-approved final baseline: 2026-10-31, Thoiry/Europe-Paris.
All current inputs, results, analysis/plotting/fitting code are FROZEN.
No new changes, production, refits or rerenders without an explicit instruction.
Monitoring remains stopped. Review preparation reuses the final artifacts.

## One implementation

Remote R=/hep2-scratch/twkim/postprocessing/TROTASR
Local /Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR
P=R/stats/run3; G=P/grid; L=P/limits
Configuration R/jsons/analysis.json.
Use python -m TROTASR <stage>, on hep2 through:
bash TROTASR/workflows/run_inference.sh -m TROTASR <stage>

Existing numerical dependencies only, under /hep2-scratch/twkim/envs:
combine_20260914/CMSSW_14_1_0_pre4 and trotasr_inference_py39.
Do not recreate any alternate environment, old model, source snapshot or sidecar.
The user explicitly retained R/../Stop_Object_Review as review material,
not a second analysis implementation. Source NanoAOD/MiniAOD stays read-only.

Local/remote maintained Python/shell sources match exactly (131 files).
Source identity: SHA256 over sorted relative-path -> file-SHA256 JSON,
compact separators, in utils/workflows/gnn4lowdm and top-level *.py;
exclude __pycache__, history and macOS ._ metadata.
dc863a126a248d9ace0f27829c96d0db0d3b583123eb5f657cf96ba21b7011a2
analysis.json SHA256:
db2a5adccf19135ea8c75827970db4ccf0c4e45c56d1d753ef8254c03e853cdc

Current input/calibration/model/compiled/normalization/histogram/measurement
payloads stay at R/inputs,scales,models,utils/compiled,estimations,results.
Four adopted late-era efficiency files in estimations were matched byte-for-byte
before retiring the old local analysis tree. No large ROOT rehash/readback,
upstream reconstruction, physics changes or reprocessing was performed.

## Adopted model and completed results

Six-era era-merged model: 2022/2022EE/2023/2023BPix/2024/2025.
48channels/1284bins/288freeCRnorms ->8/214/48.
eraScale formulas, era templates/GNN boundaries and nuisance functions retained.
Histogram rate=-1. SR observations remain blinded.
Early normalization61.8971; LUM revision39 and independent5% for2025.
TTWH/TTZH excluded; present TTTT belongs to Top. Adopted fit-only exclusions
and all actual fit-quality qualifications remain unchanged.
The existing partial-systematic scope is preserved; this finalization neither
adds uncertainties nor certifies an all-source systematic model. Do not infer
a missing nuisance or authorize new work from the scope label alone.

L:895/895 accepted, T2tt205/T2bW345/T2tb345, active0.
Actual five expected quantiles and existing numerical acceptance are preserved.
The other60 points remain excluded. Use actual point receipts/workflow_state.
P/impacts: T2tt1200/500, mu1 S+B Asimov,r[0,20],753/753 valid.
publication.json complete; collection/impacts_mStop1200_mLSP500.json.
P/cronly: observed CR-only B-only r0,noAsimov,SR0,156/156 constrained profiles,
306totalparameters; publication.json complete. Actual covQual2 remains2,
global_covariance_accepted=false. User owns scientific approval/responsibility.
No diagnostic is a substitute for an expected-limit fit.
Current scientific completion receipts, accepted workspaces and required
same-model input records remain part of the frozen baseline, not old runtimes.

## Final figures

Remote P/plots/limits; local output/pdf/run3_limits_current_20261023/
T2tt.png/pdf,T2bW.png/pdf,T2tb.png/pdf,index.html,delivery_manifest.json.
Actual final limit snapshot2026-10-30 13:11:55CET,full895coverage.
Reuse these figures; a status report does not authorize a refresh.

Remote P/plots/cronly; local output/pdf/cronly/
cronly_prefit_postfit.png/pdf,prediction_payload.json,plot_manifest.json.
One66-bin comparison: high-dM36 LEFT,low-dM30 RIGHT; Prefit above/Postfit below.
Both uncertainty bands and Data/MC,17pt category boxes inside main ax at its
x axis,24pt bold titles,one-row legends,no yield counts,full-height dividers.
Base24x11in, horizontal crop19.89x11in,0.10in outer padding.
Ymax11.5x actual display peak; prefit2.615e6/postfit2.571e6.
Existing canonical draw_flat_blocks style/colors/labels unchanged.
Prefit:242Gaussian+16Poisson priors. Postfit:full saved correlated covariance,
500shared draws,seed12345. Cached66bins; no fit/ROOT re-export/resampling.
Actual PDF QA2026-10-30 16:39:21CET; source output checksums matched.

Local impacts: output/pdf/impacts/impacts_T2tt_1200_500.pdf(26pages),
_part1.png/_part2.png(top60),delivery_manifest.json; all actual pages reviewed.
Local CR pulls: output/pdf/era_merged_diagnostics_20261029/cronly/
cronly_nuisance_pulls_ylim6_split2.pdf,_part1.png,_part2.png;78+78,2pages reviewed.
Retain current six-era distribution/systematic figure deliveries in output/pdf,
reference diagrams and GNN figures; their dates are delivery identities only.

AN chat Review Overleaf AN draft:019f7a16-19ab-76f1-8953-a737c5b0dd68.
Latest confirmed GitLab/Overleaf sync410a466: section11.4 uses the cropped
prefit/postfit figure before nuisance pulls; PDF placement was checked.
No further message or AN change is implied by this baseline freeze.

## Retirement completed

Local and hep2 obsolete analysis/test trees, older statistical models/cards/
templates, source/layout backups, intermediate contour snapshots and obsolete
operational records were permanently removed; no Trash/restoration archive.
Unused py38 removed with explicit additional user approval.
Hep2 two measured deletion batches reclaimed111.544+50.987=162.531GiB;
free space at final check502.639GiB. Stop_Object_Review preserved by exception.
Only stats/run3,runtime,references remain as hep2 statistics directories.
Only TROTASR and the retained review tree remain under hep2 postprocessing.
Current numerical source/config identities, inputs and completion receipts
were preserved. No new fit, workspace build, rerender or smoke campaign ran.
No Git-history rewriting or conversation-database changes.

# 10 GeV / 250 GeV background-estimation handoff

## Scope and provenance

The user confirmed strict electron/loose-muon veto **pT > 10 GeV** and the
**250 GeV Low-dM double-ratio lower boundary**. Both years were remeasured
from the new Top/W-SF histograms promoted on 2026-09-11. The final two-year
audit is complete. The already-completed 2025 calculation was not repeated.
No old 5-GeV factors were reused.

- Local: `/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/background_estimation_lepton_veto10_20260908`
- EOS: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/reports/background_estimation_lepton_veto10_20260908`
- Canonical manifest SHA256: `868acb7888b77fb2a218df4a63f16c2801f84ab8683b56cd7d6386e4f3b8718c`.
- `campaign_state.json` records the final audit, every per-year artifact hash,
  input hashes, and source hashes. `calculation_state.json` preserves the
  separate completed calculation and command records for each year.

## Inputs for the existing card machinery

Use the following names under **`2024/` and `2025/`**. Both years use the
same interface and passed the final input, product and covariance audit.

| Product | File |
|---|---|
| Full histogram-derived TF input, including GNN systematic templates | `tf_inputs.json` |
| Reported High-dM and Low-dM TF values/covariance | `tf/transfer_factors_YEAR_nb_recoil.json` |
| Full joint ee/μμ on/off-Z measurement | `dy_measurement.json` |
| High-dM RZ export | `rz_high.json` |
| Low-dM RZ export | `rz_low.json` |
| Combined RZ covariance and per-channel RZ–RT covariance | `rz_covariance.json` |
| Sgamma factors and covariance | `sgamma/sgamma_ut.json` |
| Z/gamma double ratio | `zgamma/zgamma_double_ratio.json` |
| Frozen GNN mapping, TFs, parent fractions, Z projections | `lowdm_gnn_backgrounds.json` |
| UT-component double-ratio up/down responses in GNN bins | `lowdm_gnn_double_ratio.json` |

Do not replace `tf_inputs.json` with the smaller GNN export when complete
systematic histograms are needed: the small export intentionally omits the
large `histograms` member. Generated `tf_inputs.json` is retained on EOS and
excluded from Git; its recorded checksum is authoritative.

## Existing conventions retained

- The on/off-Z fit uses `DATA = RZ * zll + RT * other` in each window.
  `zll` is the producer's dataset-level group: process `DY`, plus names
  containing TTZ, WZ, ZZ, WWZ, WZZ, ZZZ or WZG. Other MC is `other`.
  This is not event-level generator matching and is not pure DY versus
  Top: `RT` scales the complete `other` component, not only TT + ST.
  The implementation is `_dy_rz_component` in
  `autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py`.
  On-Z is 71 < mll < 111 GeV; off-Z is 50 < mll < 71 or mll > 111 GeV.
- High-dM retains 73 bins after dropping the first six of 79 source bins.
- Low-dM final templates remain frozen six SR categories × five GNN bins.
  Nb1/Nb2plus are factor-measurement groups, not replacement SR categories.
- Top = TT + ST. Top/W sharing, CR-parent mapping, and nuisance correlations
  are unchanged. This task did not construct cards or run limits/impacts.
- GNN TF is SR score-bin MC divided by mapped CR-parent MC integral.
  Shared-denominator covariance and parent score fractions are retained.
- Z prediction is `RZ(Nb) * sum_UT[H_Zinv(GNN,UT) * Sgamma(Nb,UT)]`.
  Q is not transferred. No extra Z normalization is applied after projection.
- Sgamma and double ratio use the merged 500–1500 displayed tail. This is
  open-ended through the already-folded upstream overflow.
- GNN double-ratio responses use `delta = abs(D - 1)`, with multipliers
  `1 + delta` and `1 / (1 + delta)`. The plotted reporting band
  `max(abs(D - 1), statistical error)` must not replace this card convention.
- RZ combined cross-group diagonal covariance is the existing downstream
  assumption; these marginal measurements do not determine cross-region
  covariance. The complete per-channel RZ–RT fitted covariance is exported.
- SR data remain blinded. No VR likelihood channels were introduced.

## Plots and interpretation

Each updated year has 32 figures, available in both PNG and PDF (64 files):

- `dy_report/highdm/` and `dy_report/lowdm/`: RZ, RT and mll before/after scaling.
- `sgamma/`: Q diagnostic and Sgamma by Nb group.
- `zgamma/`: double ratios for both regimes.
- `tf/`: High-dM Top, W, QCD TFs; `tf/gnn/`: Low-dM Top, W, QCD and raw Z/GCR TFs.

Existing plotting code was reused, including the Photon HLT-style TF layout.
Each Low-dM background is displayed with its categories overlaid, without
changing bin edges or values. Every measured factor JSON retained its new
calculation hash after plotting. The existing presentation bundle
`../background_estimation_plots_2024_2025_20260908/` contains these updated
plots for both years, with 64 figures / 128 PNG and PDF files in total.

RZ post-scaling plots are fitted-data displays, not independent closure tests.
TF reconstruction and Q/Sgamma identities are mechanical checks, not physics
closure. A near-unity shape double ratio does not resolve the absolute GCR
normalization difference; the Q diagnostic is not an extra Z normalization.

## Limitations and retained records

The canonical 2025 source retains 8390 inputs but records 11 source bad files,
including one data file; **data luminosity coverage remains incomplete**.
The 2024 source has 5954 retained inputs, zero bad files and complete recorded
data luminosity coverage. No event input was reopened to change source
coverage accounting.

Regression result: **40 passed in 25.49 seconds**. See `RUN.md` for reproduction and batch job
identifiers, and `README.md` for the machine-derived numerical summary.
Nothing in this result package has been published to the web.

Calculation **1115059.0** (2024) and **1115053.0** (2025) completed with
return value 0 (132 and 142 seconds worker wall time). The final two-year
audit **1115061.0** terminated normally with return value 0 and 165 seconds
execution time. All **84 artifacts per year / 168 total** match locally
and on EOS, and the previous 84 completed 2025 artifacts are unchanged.
Campaign state is `complete`, with no pending years; its SHA256 is
`26701007010260ac29d5c655781567d78262b669758b4bd95715cec3596a3ef8`.
QCD has three zero nominal GNN numerator bins in 2024 and four in 2025,
retained without nonzero floors; neither year has signed nominal TF bins.
All 30 GNN double-ratio responses per year reconstruct, with maximum
fractional up responses 0.00899736 (2024) and 0.0572099 (2025).
These are mechanical checks, not physics closure.

The main task must next generate/check the full SR/CR histogram plots locally
before creating cards. The background-estimation plots here do not substitute
for that separate full-histogram plotting step. No cards or limits were run.

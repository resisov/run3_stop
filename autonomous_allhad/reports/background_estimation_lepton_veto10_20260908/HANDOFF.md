# 10 GeV / 250 GeV background-estimation handoff

## Scope and provenance

The user confirmed strict electron/loose-muon veto **pT > 10 GeV** and the
**250 GeV Low-dM double-ratio lower boundary**. Both years were remeasured
from the promoted histogram-only inputs; no old 5-GeV factors were reused.

- Local: `/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/background_estimation_lepton_veto10_20260908`
- EOS: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/reports/background_estimation_lepton_veto10_20260908`
- Canonical manifest SHA256: `92f800ec26da259e96b9e72ba14e4aeff35e6453f741b237e8678784245945cc`.
- `campaign_state.json` records the final audit, every per-year artifact hash,
  input hashes, and source hashes. `calculation_state.json` preserves the
  separate completed calculation and command records for each year.

## Inputs for the existing card machinery

Use the following names under **each of `2024/` and `2025/`**:

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
large `histograms` member.

## Existing conventions retained

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

Each year has 52 figures, available in both PNG and PDF (104 files/year):

- `dy_report/highdm/` and `dy_report/lowdm/`: RZ, RT and mll before/after scaling.
- `sgamma/`: Q diagnostic and Sgamma by Nb group.
- `zgamma/`: double ratios for both regimes.
- `tf/`: High-dM Top, W, QCD TFs; `tf/gnn/`: Low-dM Top, W, QCD and raw Z/GCR TFs.

Existing plotting code was reused. Only rendering controls were added to
separate batch calculation from local plotting; the existing mll y-axis upper
limit was extended where the Z peak would otherwise be clipped. Every measured
factor JSON retained its original calculation hash after plotting.

RZ post-scaling plots are fitted-data displays, not independent closure tests.
TF reconstruction and Q/Sgamma identities are mechanical checks, not physics
closure. A near-unity shape double ratio does not resolve the absolute GCR
normalization difference; the Q diagnostic is not an extra Z normalization.

## Limitations and retained records

The canonical 2025 source retains 8390 inputs but records 11 source bad files,
including one data file; **data luminosity coverage remains incomplete**.
The 2024 source retains 5954 inputs, with no source bad files and complete
recorded coverage. No event input was reopened to change this accounting.

Previous `background_estimation_histonly_20260907` results and rollback records
are preserved. Its 254 recorded per-year artifact checksums still match, and
no tracked file in that result directory changed.

Regression result: **38 passed**. See `RUN.md` for reproduction and batch job
identifiers, and `README.md` for the machine-derived numerical summary.
Nothing in this result package has been published to the web.

Final audit **1111099.0** completed with return value 0. All 248 recorded
per-year artifacts and 11 canonical source hashes also match locally after
retrieval. QCD has 3 (2024) and 4 (2025) zero nominal GNN numerator bins;
these were kept as measured, not assigned a nonzero floor. The signed-bin
lists are empty. Each year contains 30 reconstructed GNN double-ratio
component responses.

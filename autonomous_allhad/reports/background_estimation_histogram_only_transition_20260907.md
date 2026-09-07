# Background-estimation histogram-only transition

Date: 2026-09-07

## Adopted boundary

The nominal histogram production is now the only stage allowed to read the
decorated intermediate ROOT files. While normalized event weights and region
objects are already in memory, it records a compact
`background_estimation_histograms_v1` product. The chunk merger writes this as

```text
<merged-histogram-stem>_background_estimation.json
```

Every background-estimation step after that boundary is JSON/histogram-only.
No TF, Sgamma, Z/gamma double-ratio, or RZ command accepts ROOT or NanoAOD
inputs.

The boundary contains:

- SR, LLCR, QCDCR, GCR, DY2E, and DY2M yields versus native U_T;
- nominal data and all normalized MC components;
- all available weight variations needed by Top, W, and QCD transfer factors;
- disjoint on-Z/off-Z data, Z-like, and other-background counts;
- mll histograms used by the existing RZ pre/post-fit visualization code;
- dataset and normalization provenance.

Signal-region data are never recorded in the compact product.

## Category policy

High-dM and Low-dM use the same background-estimation categories:

```text
Nb1      : Nb = 1
Nb2plus  : Nb >= 2
```

U_T remains a factor-measurement axis. It is not an additional event category.
The final Low-dM SR template is **frozen GNN30**, not U_T or legacy Low-dM34.
The process-separated GNN-score × U_T histogram product supplies the actual
component mapping. Its six SR categories and four existing CR parents remain
unchanged; the Nb-only policy does not replace this frozen final layout.
The former Low-dM Njet, b-jet-pT, ISR, and search-bin-family subdivisions are
not present in the compact estimator input.

## DY mass windows

The RZ matrix follows the current nominal DYCR definition:

```text
on-Z  : 71 < mll < 111 GeV
off-Z : 50 < mll < 71 GeV or mll > 111 GeV
```

The two channels remain DY2E and DY2M, and both High-dM and Low-dM are derived
from the same merged histogram artifact.

## Retired direct-read workflows

The following event-level background-estimation programs were removed:

- `workflow/build_nb_recoil_transfer_inputs_2024.py`;
- `dy_estimation/prepare_features.py`;
- `dy_estimation/feature_stage.py`;
- `dy_estimation/merge_features.py`;
- `dy_estimation/prepare_lowdm.py`;
- `dy_estimation/run_lowdm_partition.py`;
- `dy_estimation/lowdm_recovery.py`;
- `dy_estimation/sparse.py`;
- `dy_estimation/merge_lowdm.py`;
- `dy_estimation/validate.py`.

The remaining ROOT readers under `workflow/` are upstream production,
integrity validation, object/selection studies, or Combine-output readers.
They are not accepted inputs to the active background-estimation entrypoints.
Standalone historical closure/diagnostic scripts are likewise outside the
recalculation sequence and must not be used to build the new factor products.

## Recalculation order

For each year, after the canonical nominal histogram merge finishes:

```bash
python workflow/build_histogram_tf_inputs_2024.py \
  --hist-input <hists-stem>_background_estimation.json \
  --campaign-year <YEAR> \
  --output <work>/tf_inputs.json

python workflow/build_sgamma_ut_report_2024.py \
  --hist-input <hists-stem>_background_estimation.json \
  --campaign-year <YEAR> \
  --output-dir <work>/sgamma

python workflow/build_zgamma_double_ratio_2024.py \
  --hist-input <hists-stem>_background_estimation.json \
  --campaign-year <YEAR> \
  --output-dir <work>/zgamma

python -m autonomous_allhad.dy_estimation build-measurement \
  --hist-input <hists-stem>_background_estimation.json \
  --campaign-year <YEAR> \
  --output <work>/dy_measurement.json

python -m autonomous_allhad.dy_estimation report \
  --measurement <work>/dy_measurement.json \
  --campaign-year <YEAR> \
  --selection both \
  --output-dir <work>/dy_report
```

The existing plotting implementations are reused. No plot was regenerated as
part of this code transition.

For the completed 2024/2025 histogram-only recalculation and GNN propagation,
use `reports/background_estimation_histonly_20260907/RUN.md`. That sequence adds
the frozen GNN histogram and configuration to the existing TF builder, retains
parent-normalization semantics despite different SR/CR score edges, and
exports RZ × Sgamma Zinv projections without transferring Q. The Low-dM
double-ratio 250-GeV domain was approved on 2026-09-07T22:29:50Z and is now the
default for both years. Previous 300-start products and checksums are preserved
in that report's `YEAR/zgamma_300_previous/` and `low250_adoption_baseline.json`.
Only central abs(D−1) is transmitted through the existing GNN×UT mapping;
RZ, TF, Sgamma and nominal GNN predictions are unchanged by this adoption.

## Regression checks

`tests/test_background_estimation_histogram_only.py` enforces that active
estimator entrypoints contain no uproot access and expose no exact/sparse ROOT
arguments. A synthetic compact histogram product is then propagated through
the TF, Sgamma, Z/gamma, and RZ builders. The test requires complete outputs,
the two-category policy in both regimes, the 71--111 GeV on-Z definition, and
`intermediate_root_reread: false` provenance.

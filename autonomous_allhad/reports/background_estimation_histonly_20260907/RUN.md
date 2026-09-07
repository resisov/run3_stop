# Reproduce the histogram-only background products

Run from the repository root using the existing local Python environment with
NumPy, SciPy, Matplotlib and mplhep. No environment rebuild, ROOT access,
NanoAOD access, job submission or histogram reweighting is involved.

The authoritative input list is
`autonomous_allhad/reports/dy_window20_canonical_20260907.json`.
`campaign_state.json` records the checked source and product SHA256 values.
Do not reuse these products with a different histogram hash.

For each `YEAR` in `2024 2025`, use this order. Existing valid products may be
kept; the measurement stages need not repeat for a plotting-only update.

```bash
export PYTHONPATH=autonomous_allhad
export MPLCONFIGDIR=autonomous_allhad/workflow/.mplconfig
YEAR=2024
RESULT=autonomous_allhad/reports/background_estimation_histonly_20260907/$YEAR
HIST=autonomous_allhad/workflow/plot$YEAR/hists_background_estimation.json
GNN=autonomous_allhad/workflow/plot$YEAR/hists_lowdm_gnn30.json

python3 -m autonomous_allhad.dy_estimation build-measurement \
  --hist-input "$HIST" --campaign-year "$YEAR" \
  --output "$RESULT/dy_measurement.json"

python3 -m autonomous_allhad.dy_estimation report \
  --measurement "$RESULT/dy_measurement.json" --campaign-year "$YEAR" \
  --selection both --output-dir "$RESULT/dy_report"

python3 autonomous_allhad/workflow/build_sgamma_ut_report_2024.py \
  --hist-input "$HIST" --campaign-year "$YEAR" --output-dir "$RESULT/sgamma"

python3 autonomous_allhad/workflow/build_zgamma_double_ratio_2024.py \
  --hist-input "$HIST" --campaign-year "$YEAR" --low-ut-min 300 \
  --output-dir "$RESULT/zgamma"

# Comparison only: no automatic adoption of the 250-GeV normalization domain.
python3 autonomous_allhad/workflow/build_zgamma_double_ratio_2024.py \
  --hist-input "$HIST" --campaign-year "$YEAR" --low-ut-min 250 \
  --output-dir "$RESULT/zgamma_250_proposal"

python3 autonomous_allhad/workflow/build_histogram_tf_inputs_2024.py \
  --hist-input "$HIST" --campaign-year "$YEAR" \
  --gnn-input "$GNN" --gnn-config autonomous_allhad/gnn_lowdm/config.json \
  --sgamma-input "$RESULT/sgamma/sgamma_ut.json" \
  --dy-measurement "$RESULT/dy_measurement.json" \
  --output "$RESULT/tf_inputs.json"

python3 autonomous_allhad/workflow/plot_recoil_transfer_factors_2024.py \
  --input "$RESULT/tf_inputs.json" --campaign-year "$YEAR" \
  --output-dir "$RESULT/tf"
```

After both years:

```bash
python3 autonomous_allhad/workflow/audit_background_histogram_products.py \
  --manifest autonomous_allhad/reports/dy_window20_canonical_20260907.json \
  --output-root autonomous_allhad/reports/background_estimation_histonly_20260907

python3 -m pytest -q \
  autonomous_allhad/tests/test_background_estimation_histogram_only.py \
  autonomous_allhad/tests/test_gnn_background_histograms.py \
  autonomous_allhad/tests/test_dycr_mass_window.py \
  autonomous_allhad/tests/test_highdm_orthogonality_projection.py
```

The audit exports small `rz_high.json`, `rz_low.json`, `rz_covariance.json` and
`lowdm_gnn_backgrounds.json` handoff files without modifying canonical inputs.
The full `tf_inputs.json` also retains process-separated GNN joint histograms
and available weight variations. Neither legacy Low-dM34 nor SR observations
are used as final templates. Only the main agent owns card integration.

The 250-GeV double-ratio proposal remains separate. Use the explicit
`downstream_central_abs_deviation` field for existing card semantics, not the
historical plot's `systematic=max(abs(D-1),stat)` reporting band.

# Strict 10-GeV lepton-veto background estimation: 2024 and 2025

## Top/W input update — 2026-09-11

2024 was subsequently promoted with canonical manifest SHA256
`868acb7888b77fb2a218df4a63f16c2801f84ab8683b56cd7d6386e4f3b8718c`.
Validation 1115056.0 completed with ExitCode=0. The same `calculation.sub`
submitted **only 2024** as **1115059.0** with py38/workday on the EOS schedd.
The completed 2025 calculation is not resubmitted. The final audit will check
both years and refresh the combined campaign state without refitting 2025.

2025 was promoted by the main task with canonical manifest SHA256
`3a9784dd2e648300f0e4790f41c96ec99fcc174454c90b50c3802c15c49d55ca`.
The existing `calculation.sub` submitted only 2025 as **1115053.0** on the
EOS schedd `bigbird24.cern.ch`, using the existing py38 runtime and workday.
2024 was not included in that 2025 job; its later submission is recorded above.

Calculation 1115053.0 completed with ExitCode=0 (142 seconds worker wall time).
All five derived products matched the new recorded hashes; four calculation
commands exited successfully. Existing local plot commands completed, with
TF mechanical residual 1.36e-16. The obsolete 48 per-category Low-dM TF PNG/PDF
files for 2025 were removed after verifying the new combined TF figures.
They remain recoverable from Git history. Large generated `tf_inputs.json`
files are no longer tracked in Git; the actual EOS/local products remain.
Final 2025 audit **1115054.0** completed with ExitCode=0 (96 seconds worker
wall time), verified 84 artifacts and exported all RZ covariance/GNN response
products. The campaign state is `partial`, with `audited_years=[2025]` and
`pending_years=[2024]`; this is not a failed 2025 calculation. Forty regression
tests passed. The full SR/CR histogram plotting step remains with the main task.

The existing runner's `--replace-stale` option retains the previous products
while recalculating in Condor worker scratch. It retains final relative
provenance paths, checks the completed products and copies, then replaces the
five measured products in the same campaign. No new campaign or sidecar is
created. The frozen numerical sources/configuration are unchanged. Existing
input SHA checks are retained; no old/new histogram bitwise protection is used.
`audit_background_histogram_products.py --years 2025` audits this ready year
without treating pending 2024 as an updated result. After both years are ready,
omit `--years` for the complete audit. ROOT inputs and large histogram JSON
remain on EOS; only compact measured products are copied for local plotting.

The execution records below describe the original 2026-09-08 measurement.

This campaign uses the user-approved canonical manifest in `input_manifest.json`.
It does not reuse factors measured with the old 5-GeV selection. The Low-dM
double-ratio lower boundary is the already-approved 250 GeV. All downstream
input is JSON histograms; ROOT and NanoAOD access are prohibited.

## 1. Calculation on EOS Condor

`batch/run.py` is orchestration only. Per year it verifies the main, compact,
and GNN histogram SHA256 values and canonical aliases, then calls:

1. `python -m autonomous_allhad.dy_estimation build-measurement`
2. `build_sgamma_ut_report_2024.py --no-plots`
3. `build_zgamma_double_ratio_2024.py --low-ut-min 250 --no-plots`
4. `build_histogram_tf_inputs_2024.py` with the new RZ/Sgamma and GNN inputs.

Both years use identical source. `batch/code_manifest.json` fixes the small
source/config snapshot; `YEAR/calculation_state.json` records all commands,
input hashes, completed steps and final product hashes. Successful states are
reused only after checksum validation. Do not resubmit a valid calculation.

The batch runtime is the existing transferred `py38.tgz`, with one CPU,
6000 MB memory and 5000 MB disk per year. The two calculations were submitted
to `bigbird24.cern.ch` as cluster **1111095**, ProcId 0=2024 and 1=2025.
Submission-time proxy/VOMS lifetimes were 645253 / 616453 seconds. Runtime
caches remain inside Condor scratch, never AFS or an lxplus `/tmp` workflow.
Both calculation jobs terminated normally with return value 0. The schedd
allocated 2 CPUs per calculation despite the submit file requesting 1;
logged execution times were 156 seconds (2024) and 115 seconds (2025).

## 2. Existing local plotting

Copy only the small result products, not the multi-GB nominal/GNN inputs.
Run these commands from the local repository root for each year:

```bash
export PYTHONPATH=autonomous_allhad
export MPLCONFIGDIR="$PWD/autonomous_allhad/workflow/.mplconfig"
YEAR=2024
RESULT=autonomous_allhad/reports/background_estimation_lepton_veto10_20260908/$YEAR
HIST=autonomous_allhad/workflow/histograms/lepton_veto10_20260908/$YEAR/hists_background_estimation.json
python3 -m autonomous_allhad.dy_estimation report --measurement "$RESULT/dy_measurement.json" --campaign-year "$YEAR" --selection both --output-dir "$RESULT/dy_report"
python3 autonomous_allhad/workflow/build_sgamma_ut_report_2024.py --hist-input "$HIST" --campaign-year "$YEAR" --output-dir "$RESULT/sgamma" --plot-only
python3 autonomous_allhad/workflow/build_zgamma_double_ratio_2024.py --hist-input "$HIST" --campaign-year "$YEAR" --low-ut-min 250 --output-dir "$RESULT/zgamma" --plot-only
python3 autonomous_allhad/workflow/plot_recoil_transfer_factors_2024.py --input "$RESULT/tf_inputs.json" --campaign-year "$YEAR" --output-dir "$RESULT/tf"
```

Plot-only modes reuse existing figure functions and write `plot_manifest.json`;
they do not change measured factor JSONs or their source hashes.
The existing DY `report.py` now extends its logarithmic y-axis upper limit
when needed so that a populated Z peak is not cut off at 1000 events/bin.
This rendering-only fix does not change yields, fits, or error propagation.

## 3. Final audit and export

After synchronizing the local figures to the same relative EOS result path,
run the existing `audit_background_histogram_products.py` in a Condor worker:

```bash
python autonomous_allhad/workflow/audit_background_histogram_products.py \
  --manifest autonomous_allhad/reports/lepton_veto10_canonical_20260908.json \
  --output-root autonomous_allhad/reports/background_estimation_lepton_veto10_20260908
```

It exports `rz_high.json`, `rz_low.json`, `rz_covariance.json`,
`lowdm_gnn_backgrounds.json`, and `lowdm_gnn_double_ratio.json`, plus the
machine-readable campaign state and report. Input hash and alias mismatches,
stale factor inputs, invalid fit covariance, missing plots or missing GNN
mappings stop the audit. Signed MC bins are retained and reported, not clipped.

No old 300-GeV comparison or old 5-GeV factor is required by a fresh campaign.
The previous 20260907 results and rollback files remain untouched. Cards,
limits, impacts and web publication are outside this campaign's scope.

The final audit job is **1111099.0**, submitted with 1 CPU and 2000 MB memory.
It was held before execution while the DY display range was fixed, then
released without resubmitting the completed factor calculations.
It completed normally with return value 0, with 95 seconds of execution;
the schedd allocated 1 CPU and 3000 MB memory. Both years passed all recorded
input, source, artifact, covariance and GNN-projection checks.

## 4. Local regression tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=autonomous_allhad python3 -m pytest -q \
  autonomous_allhad/tests/test_background_estimation_histogram_only.py \
  autonomous_allhad/tests/test_gnn_background_histograms.py \
  autonomous_allhad/tests/test_dycr_mass_window.py \
  autonomous_allhad/tests/test_highdm_orthogonality_projection.py
```

Result: 38 passed in 25.52 seconds, including plot-only factor immutability,
strict-10-GeV/250-GeV manifest policy, and large-Z-peak visibility.

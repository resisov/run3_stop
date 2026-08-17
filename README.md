# Run-3 All-Hadronic Stop Analysis

CMS Run-3 all-hadronic stop search workspace. The active analysis is the 2024
workflow under `autonomous_allhad`; `analysis` supplies the Coffea processor,
object definitions, corrections, metadata, and adopted scale-factor payloads.

## Active entry points

- `autonomous_allhad/analysisctl`: top-level workflow command
- `autonomous_allhad/configs/run3_2024.yaml`: active 2024 configuration
- `analysis/processors/stop_processor_v4.py`: current Coffea processor reference
- `analysis/utils/ids.py`: object definitions
- `analysis/utils/corrections.py`: corrections and scale factors
- `autonomous_allhad/workflow/plot_control_search_bins_style.py`: main plotter
- `autonomous_allhad/workflow/build_an_zinv_combine_inputs_2024.py`: adopted control-region likelihood builder
- `autonomous_allhad/workflow/build_free_background_combine_inputs_2024.py`: free-background comparison model

The Run-2 physics reference `AN2019_016_v9.pdf` is intentionally local-only and
is excluded from Git.

## Runtime

The production working copy and large runtime products live on EOS:

```text
/eos/user/t/taiwoo/run3_stop/decaf
```

The standard Python environment is:

```text
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
```

Run the active command surface with:

```bash
PYTHONPATH=/eos/user/t/taiwoo/run3_stop/decaf:/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad \
  ./autonomous_allhad/analysisctl --help
```

Large ROOT files, Condor products, merged Coffea payloads, logs, temporary
histograms, and Combine internals are not stored in Git. Reproducible source,
compact validation records, adopted correction payloads, and final public
artifacts are retained here.

## Retained outputs

- `analysis/data/AnalysisSF/2024/`: adopted MET/photon trigger and low-pT lepton SFs
- `autonomous_allhad/reports/`: final measurement, closure, and datacard-strategy reports
- `autonomous_allhad/validation/`: compact validation records
- `autonomous_allhad/signals/`: signal discovery, mass grid, and stop cross sections
- `docs/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/`: current public 2024 plots, limits, and impacts

The current public page is:

https://resisov.github.io/run3_stop/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/

Historical web snapshots and superseded diagnostic outputs remain recoverable
from Git history but are not kept in the working tree.

## Validation

Run the repository tests from the project root:

```bash
python3 -m pytest autonomous_allhad/tests
```

Before declaring a production result complete, validate file coverage,
normalization, required systematic variations, plot inputs, datacards, fit
outputs, and public-file checksums.

## Remaining analysis work

- add the remaining Run-3 years;
- propagate and validate the full systematic model across all years;
- regenerate combined cards, limits, and impacts only after those inputs pass validation.

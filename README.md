# Run-3 all-hadronic stop analysis

The current analysis package is **[TROTASR](TROTASR/README.md)** on the
`run3_full_analysis` branch.

For setup, required inputs, exact commands and output locations, start with
the **[TROTASR execution guide](TROTASR/README.md)**.

```text
Integrated Events + TROTA ROOT files
  → high-/low-dM histograms with corrections and systematic variations
  → yearly merge and background measurements
  → templates and datacards
  → expected limits, impacts, CR-only fit and pulls
  → SR/CR distributions and exclusion contours
```

The 2024+2025 statistical model uses 118 high-dM and 30 low-dM SR bins per
year, high-dM CRs with 12 bins each, and low-dM CRs with 10 bins each.
The combined model has 428 bins in 16 channels. Observed SR data remain blinded.

| Resource | Location |
|---|---|
| Execution guide | [TROTASR/README.md](TROTASR/README.md) |
| Object definitions | [ids.adl](TROTASR/utils/ids.adl) |
| Region and bin definitions | [event_selections.adl](TROTASR/utils/event_selections.adl) |
| Correction keys, SFs and variations | [corrections.adl](TROTASR/utils/corrections.adl) |
| Combined model configuration | [combined_systematics_config.json](TROTASR/jsons/combined_systematics_config.json) |
| Final high-dM category merge | [sr_category_merge.json](TROTASR/jsons/sr_category_merge.json) |
| Early Run-3 payload inventory | [early_run3_assets.json](TROTASR/jsons/early_run3_assets.json) |

2022, 2022EE, 2023 and 2023BPix calibration payloads are staged in TROTASR.
Their histogram/statistical year interfaces are not yet connected; payload
availability alone does not make those eras executable through the 2024/2025
commands.

`autonomous_allhad/` contains upstream production tools and earlier analysis
implementations. Its historical histogram/datacard drivers are not the current
TROTASR execution path.

Event ROOT files, HDF5 models, campaign outputs, fit scratch and generated
figures are not included in Git. The execution guide lists the external
inputs required for reproduction. Internal test/validation products and
PNG/PDF files are excluded from publication unless explicitly requested.

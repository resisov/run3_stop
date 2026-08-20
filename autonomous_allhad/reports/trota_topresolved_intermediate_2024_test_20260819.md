# TROTA TopResolved on the 2024 intermediate ROOT

Status: **passed** on 2026-08-19.

The official 2024 TROTA TopResolved HDF5 model was evaluated directly from the existing intermediate ROOT branches. A NanoAOD-to-NanoAOD rewrite is not required.

## Selected working point

The selected 1% QCD-mistag training working point is

`TTScore / (TTScore + QCDScore) >= 0.9433798789978027`.

This is a TROTA training-study threshold. The 2024 data/MC scale factor is not applied.

## Test results

| Sample | Events tested | Candidates | Candidate pass | Events with >=1 pass |
|---|---:|---:|---:|---:|
| Data | 3,594 (full file) | 16,290 | 377 (2.31%) | 184 (5.12%) |
| Background MC | 5,000 | 13,464 | 84 (0.624%) | 58 (1.16%) |
| Signal | 5,000 | 93,224 | 6,490 (6.96%) | 2,337 (46.74%) |

All scores were finite and all three-class probability sums agreed with one to better than `2.24e-7`.

## Important constraints

- Jet selection uses stored `jet_id_all`, `jet_nanoaod_pt > 25 GeV`, and `abs(jet_eta_all) < 2.5`.
- Official 2024 JetID is not recomputed because its full composition inputs are absent from the intermediate schema.
- The pass fractions above are sample observations, not calibrated efficiencies.
- TensorFlow 2.12.0 loads the legacy model; TensorFlow 2.18.0 did not load its `SlicingOpLambda` layer.
- TROTA scores remain excluded from the primary event categorization.

The complete machine-readable result is in `trota_topresolved_intermediate_2024_test_20260819.json`.

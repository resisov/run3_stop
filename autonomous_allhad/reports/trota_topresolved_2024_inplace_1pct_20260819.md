# TROTA TopResolved 2024 in-place production

Status: **complete** on 2026-08-19.

TROTA TopResolved was added in place to all 5,489 existing 2024 intermediate ROOT files. The original filenames and `Events` trees were preserved. No invalid files or temporary/backup artifacts remain.

## Production result

| Quantity | Result |
|---|---:|
| Files | 5,489 / 5,489 valid |
| Events | 230,830,776 |
| Candidates evaluated | 1,531,057,643 |
| Candidates persisted | 99,051,747 |
| Added storage | 3,833,428,617 bytes |
| Invalid files | 0 |

Only passing candidates are stored in the sparse `TROTA` tree. Scores and kinematics are `float32`; completion and provenance are stored in `TROTA_metadata`.

## Physics definition

The 1% QCD-mistag working point is

`TTScore / (TTScore + QCDScore) >= 0.9433798789978027`.

Jets use stored `jet_id_all`, `jet_nanoaod_pt > 25 GeV`, and `abs(jet_eta_all) < 2.5`. Official 2024 JetID is not recomputed because the required full NanoAOD composition inputs are not present. The not-yet-available 2024 TROTA data/MC scale factor is not applied.

## Validation

Every output was opened directly through XRootD. The audit checked the preserved `Events` schema and counts, exact TROTA branch schema and dtypes, provenance marker, row counts, finite scores, probability sums, the 1% threshold, and discriminator recomputation for all 99,051,747 stored rows.

- Valid files: 5,489
- Invalid files: 0
- Maximum probability-sum deviation: `2.980232238769531e-07`
- Maximum discriminator recomputation difference: `0.0`
- Temporary or backup artifacts: 0

The machine-readable local report is `trota_topresolved_2024_inplace_1pct_20260819.json`. The authoritative EOS audit is `full_campaign/xrootd_validation_summary.json` under the campaign directory.

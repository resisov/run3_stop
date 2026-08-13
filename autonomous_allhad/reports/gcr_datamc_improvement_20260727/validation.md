# Validation record

Status: PASS with one documented local-environment limitation.

## Syntax and JSON integrity

- `python3 -m py_compile` passed for:
  - `study_gcr_datamc_improvement_2024.py`
  - `audit_gcr_photon_strata_2024.py`
  - `build_gcr_prompt_constrained_candidate_2024.py`
  - `plot_gcr_datamc_improvement_2024.py`
  - `scan_gcr_prompt_overlap_2024.py`
- all report and audit JSON files pass JSON parsing;
- report summary status is `conditional_r_and_d_not_adopted`;
- report summary records `nominal_mutated: false`;
- normalization audit status is `pass`;
- representative overlap audit status is `complete`;
- representative overlap audit records
  `scope.nominal_and_sidecar_untouched: true`;
- GenPart lookup completed for 172/172 sampled exact nominal GCR rows.

## Photon-fake probe-assignment unit tests

The local system Python cannot import Coffea because its installed Numba requires
NumPy 2.0 or older while the local NumPy is 2.1. This is an environment
incompatibility, not a test failure.

The worker and test files were verified byte-for-byte identical between the
local workspace and the CERN repository:

```text
photon_fake_2024_worker.py
76c0ac15051a605eeaefae5346c01d786d63e909818c3f2821399d79259aa9d9

test_photon_fake_2024_probe_assignment.py
22707f979450b9b000b55e9f2244edbf94b7a002374518b94606f4eb1929d03b
```

The two test functions were then executed directly in the requested CERN
miniconda Python 3.8 environment:

```text
PASS test_probe_assignment_matches_nominal_target_multiplicity
PASS test_selected_probe_values_uses_only_assigned_probe
```

The Python 3.8 environment does not contain the `pytest` package, so the
functions were invoked through `runpy` without changing the test source.

## Nominal immutability

The candidate payload recorded the source nominal SHA-256 as

```text
b51abfd2562e0e8667c0834dc4b6153dad5d4fb3751cf254c79f4a336d240eab
```

A fresh SHA-256 calculation after candidate construction returned the same
value.


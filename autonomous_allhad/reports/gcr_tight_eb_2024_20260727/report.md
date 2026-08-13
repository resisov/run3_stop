# 2024 GCR Tight-EB photon candidate

This is a read-only comparison. No nominal intermediate ROOT file was modified.

Candidate definition: trusted nominal GCR selection, additionally requiring exactly one photon with corrected $p_T>220$ GeV, $|\eta|<1.4442$, `cutBased>=3`, and `electronVeto`. The Tight photon ID scale factor is used for MC.

| Region | Policy | Data | MC | Data/MC | Shape-only p-value |
|---|---:|---:|---:|---:|---:|
| high-dM | Medium, EB+EE | 14495.0 | 10239.7 | 1.416 | 0.683 |
| high-dM | Tight, EB only | 9678.0 | 6939.2 | 1.395 | 0.507 |
| low-dM | Medium, EB+EE | 15483.0 | 13501.6 | 1.147 | 1.55e-05 |
| low-dM | Tight, EB only | 10792.0 | 9311.1 | 1.159 | 2.78e-05 |

## Interpretation

- Execution: both payloads are complete, use the same 1285 ROOT files, and have matching normalization and code hashes. No missing input, sidecar, zero-entry, weight-fallback, or weight-rejection warning was recorded.
- High-dM: Data/MC changes from 1.416 to 1.395. This removes only 5.0% of the excess above unity, while retaining 66.8% of the data events. The absolute ratio shift (0.021) is smaller than the per-selection statistical uncertainty (0.077 nominal, 0.078 Tight-EB).
- High-dM shape-only compatibility changes from p=0.683 to p=0.507; both are compatible, but the Tight-EB candidate does not improve the shape metric.
- Low-dM: Data/MC changes from 1.147 to 1.159, so the integrated agreement becomes slightly worse. The shape-only p-value remains poor (2.78e-05).
- Decision: Tight-EB alone is not a sufficient GCR Data/MC remedy and should not replace the nominal photon definition on the basis of this test. It is useful as a diagnostic cross-check because it preserves the qualitative high-dM shape while reducing statistics by about one third.

Primary plots:

- `highdm_gcr_ut_nominal.png` and `highdm_gcr_ut_tight_eb.png`
- `highdm_gcr_ut_data_mc_comparison.png`
- `lowdm_gcr_ut_nominal.png` and `lowdm_gcr_ut_tight_eb.png`
- `lowdm_gcr_ut_data_mc_comparison.png`

The normalization comparison and the shape-only test answer different questions. The integrated Data/MC value measures rate agreement; the shape-only p-value first normalizes MC to data and tests the remaining binned shape disagreement.

# Nominal 2024 high-\(\Delta M\) GCR Data/MC audit

Artifact status: read-only audit complete.

## Scope and trust boundary

This is a read-only audit. The GCR definition is taken only from:

`/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py`

Legacy `stop_processor` and `ids.py` implementations were not used. No shared code or nominal output was modified.

## Bottom line

The GCR discrepancy is **not** consistent with a global luminosity or universal MC-normalization error.

For the primary \(U_T\) distribution:

- Data: 14,495
- MC: 10,239.67
- Data/MC: \(1.4156 \pm 0.0772\)
- Difference from unity: approximately \(5.4\sigma\)
- Shape-only test after one common normalization: \(\chi^2/\mathrm{dof}=4.81/7\), \(p=0.683\)

A normalization measured in \(250<U_T<500\) GeV predicts every bin in \(500<U_T<1500\) GeV within \(0.9\sigma\). Thus the primary \(U_T\) distribution looks like a photon-channel rate deficit, not a wrong \(U_T\) shape.

However, b-jet \(p_T\), leading-jet \(p_T\), fatjet \(p_T\), \(H_T\), \(N_b\), and \(N_W\) have highly significant residual shape differences. The model therefore cannot be repaired by blindly multiplying all GCR MC by 1.416.

## Main diagnosis

Nominal GCR MC is approximately:

- Gamma+jets: 5,533.0, or 54.0%
- QCD: 4,306.6, or 42.1%
- \(t\bar t\): 322.4, or 3.1%
- single top: 52.2, or 0.5%

The truth-origin audit shows that 92.6% of the QCD-labelled target yield is actually prompt photon, while only 7.4% is truth fake. The QCD prompt component has only about 55 effective events and is sensitive to a small number of large weights.

Across populated histogram bins, the Data/MC ratio is strongly anti-correlated with the nominal QCD fraction:

| Variable | Pearson correlation |
|---|---:|
| b-jet \(p_T\) | -0.947 |
| fatjet \(p_T\) | -0.990 |
| \(H_T\) | -0.983 |
| leading-jet \(p_T\) | -0.974 |
| \(N_b\) | -0.964 |
| \(N_W\) | -0.964 |
| \(U_T\) | -0.057 |

This points to QCD/Gamma+jets prompt-photon composition, overlap, or stitching as the first issue to resolve. It does not point to a defective \(U_T\) selection.

## Shape summary after a common normalization

| Variable | Integral Data/MC | Shape \(p\)-value |
|---|---:|---:|
| \(U_T\) | 1.416 | 0.683 |
| \(N_\mathrm{jet}\) | 1.415 | 0.769 |
| \(N_\mathrm{fatjet}\) | 1.415 | 0.551 |
| \(N_\mathrm{top}\) | 1.415 | 0.510 |
| b-jet \(p_T\) | 1.416 | \(3.27\times10^{-18}\) |
| fatjet \(p_T\) | 1.386 | \(7.87\times10^{-18}\) |
| \(H_T\) | 1.416 | \(6.55\times10^{-18}\) |
| leading-jet \(p_T\) | 1.416 | \(2.18\times10^{-22}\) |
| \(N_b\) | 1.415 | \(6.42\times10^{-9}\) |
| \(N_W\) | 1.415 | \(2.21\times10^{-6}\) |

The current high-\(\Delta M\) GCR MET histogram is structurally empty: GCR requires \(p_T^\mathrm{miss}<250\) GeV, whereas that histogram starts at 250 GeV. It must be replaced by a dedicated 0–250 GeV view for validation.

## Integrity and normalization checks

- GCR data event keys: 14,497 unique, zero duplicates.
- GCR uses EGamma only; JetMET and Muon entries are excluded.
- One GCR-relevant EGamma shard is missing. Its expected impact is only about 29 events and recovering it would slightly increase Data/MC.
- The MC retained-sum-of-weights normalization audit passed for 65 datasets, agreeing with recomputation at about \(10^{-13}\) relative precision.
- Other regions do not show the same scale: QCDCR 1.035, LLCR 0.878, DY2M 0.846, SR 1.148. A universal 1.416 MC multiplier is therefore excluded.

## Recommended candidate

1. Make prompt-photon ownership between QCD and Gamma+jets generator-level exclusive, without changing `real_subset_worker.py` event selection.
2. Use the data-driven fake estimate to replace only the truth-fake component, never all QCD.
3. After overlap removal, use one prompt-pool normalization constraint in GCR. The current conditional diagnostic value is approximately 1.39.
4. Determine the factor in low \(U_T\), then validate untouched high-\(U_T\) bins.
5. Require closure in EB/EE, photon \(p_T\), Nt0/Nt1, \(N_b\), \(N_\mathrm{jet}\), and \(H_T\).

The fitted factor is a control-region constraint, not proof that the prefit MC is correct.

## Explicitly rejected actions

- Replacing all QCD with fake photons.
- Applying the GCR factor to every process and region.
- Applying an empirical per-\(U_T\)-bin reweight.
- Using the strongly anti-correlated Gamma+jets/QCD two-template fit as a physics correction.
- Claiming independent improvement using the same bins that determined the normalization.

## Required validation plots

- \(U_T\): low-range fit and high-range holdout.
- \(U_T\) split by EB/EE and photon \(p_T\).
- Photon \(p_T\), \(|\eta|\), \(\phi\), R9, and exact `cutBased`.
- Trigger-path and run-era yields normalized by luminosity.
- Generator-origin and generator-bin composition for QCD and Gamma+jets, including effective event counts.
- Two-dimensional \(U_T\) versus \(H_T\), \(N_b\), \(N_\mathrm{jet}\), photon \(p_T\), and photon \(\eta\).
- Photon-jet \(\Delta R\) before cleaning, \(U_T/p_T^\gamma\), and photon–MET \(\Delta\phi\).
- Dedicated \(0<p_T^\mathrm{miss}<250\) GeV GCR histogram.
- Orthogonal \(N_\mathrm{jet}=4\) or \(N_b=0\) validation sidebands.

## Primary evidence paths

- Nominal payload: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/nominal_plots_2024_fullselection_v3_20260725/render_payloads/categories_2024.json`
- Event-key audit: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_2024_local2400k_v3_20260726/diagnostics/gcr_event_key_audit_all_examples.json`
- Truth-origin audit: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_2024_snapshot_complete_20260726T1917Z/origin_audit_qcd_gj.json`
- Normalization audit: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_2024_snapshot_complete_20260726T1917Z/normalization_audit_all_background_contamination.json`
- Normalization input: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/intermediate_2024_fullselection_v3_lowdm_relaxed_20260724/final_nominal_inputs_20260725/normalization.json`
- Data merge manifest: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/intermediate_2024_fullselection_v3_lowdm_relaxed_20260724/merged/data_balanced20/manifest.json`

The accompanying JSON report contains the complete machine-readable metrics, recommendations, adoption gates, caveats, and evidence paths.

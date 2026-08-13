# 2024 high-\(\Delta m\) GCR normalization and correction audit

Status: complete with documented follow-ups  
Selection authority: `autonomous_allhad/autonomous_allhad/real_subset_worker.py`  
Legacy `stop_processor`/`ids` used: no  
Nominal outputs modified: no

## Result

The nominal GCR has 14,495 data events and \(10,239.674\) predicted MC events:

\[
\mathrm{Data/MC}=1.41557.
\]

The deficit is not caused by an incorrect luminosity, signed-sumw denominator, split-dataset normalization, or a duplicated normalization factor. It is also not explained by the routine photon, pileup, b-tag, or top corrections.

GJ contributes \(5,532.986\) events (54.03%) and QCD contributes \(4,306.643\) events (42.06%). Together they are 96.1% of the prediction. The selected QCD sample is statistically fragile: its \(4,306.6\)-event weighted yield has only \(N_\text{eff}=61.1\).

## Luminosity, sumw, and normalization

The campaign uses \(109.82~\mathrm{fb}^{-1}=109,820~\mathrm{pb}^{-1}\), consistent with the 2024 certified-data scale. For all 65 background physical datasets:

- the denominator is `Runs.genEventSumw`;
- numerator event weights use signed `genWeight`;
- no background sumw is nonpositive;
- no cross-section conflict or blocked normalization factor is recorded;
- all 3,946 split-dataset factors reproduce
  \[
  \sigma\,\mathcal L/\sum w_\text{physical dataset};
  \]
- the shard bundle contains 3,341 unique GJ files and no duplicated GJ file path.

The skim explicitly stores raw `genWeight` and defers final normalization in `flat_ntuple_worker.py:820-832`. The histogram builder constructs the factor once and multiplies it once in `build_flat_boosted_recoil_hists.py:435-451,977-1023`.

Five MC input files failed: one GJ 100–200 file (1/2751), one GJ 200–400 file (1/397), one QCD 300–470 file (1/631), one single-top file, and one \(Z\to\nu\nu\) file. The retained-file sumw is used consistently in both numerator and denominator. These fractions cannot create a 41.6% deficit, although a missing QCD file contributes to the already large rare-event variance.

## Independent GJ cross-section provenance

The four registry values were checked against authenticated McM production metadata and the actual CVMFS gridpack pilot integrations.

| GJ bin | McM prep-ID | Gridpack pilot xsec [pb] | McM LHE filter | Product [pb] | Used [pb] | Difference |
|---|---|---:|---:|---:|---:|---:|
| 100–200 | `GEN-RunIII2024Summer24wmLHEGS-00078` | \(5444\pm20\) | 0.26 | 1415.44 | 1391 | -1.73% |
| 200–400 | `GEN-RunIII2024Summer24wmLHEGS-00079` | \(174.1\pm0.4\) | 0.50 | 87.05 | 88.24 | +1.37% |
| 400–600 | `GEN-RunIII2024Summer24wmLHEGS-00080` | \(5.601\pm0.015\) | 0.68 | 3.8087 | 3.77 | -1.02% |
| \(>600\) | `GEN-RunIII2024Summer24wmLHEGS-00081` | \(0.7158\pm0.0015\) | 0.82 | 0.5870 | 0.576 | -1.87% |

The used values are therefore post-filter effective cross sections. The agreement within 2% excludes an order-one missing or double-applied filter efficiency. It cannot explain the GJ-only diagnostic scale of about 1.77.

The McM `cross_section` field itself is a 1 pb placeholder for these requests. A fresh full GenXSecAnalyzer pass was not possible because the active X509 proxy had expired, and the legacy XSDB hostname did not resolve. Those are percent-level follow-ups, not blockers to rejecting a factor-of-two error.

## Photon trigger, ID, and electron-veto corrections

The GCR requires one medium photon with \(p_T>220\) GeV, `cutBased >= 2`, and `electronVeto == 1` (`real_subset_worker.py:1066-1073`).

The Medium photon-ID SF is applied once and only in GCR (`real_subset_worker.py:559-568`). Its central effect is about +2.47% on the total MC; its up/down shifts are \(\pm4.74\%\). It is not duplicated and cannot close the deficit.

The `HLT_Photon175 || HLT_Photon200` requirement is present (`real_subset_worker.py:95,1482`), but no 2024 photon-trigger SF is supplied (`object_corrections_2024.py:180-186`). It must be measured versus photon \(p_T\), \(\eta\), and run era using an orthogonal trigger. Above the 220 GeV offline threshold it should be treated as a plateau-scale correction unless data prove otherwise; assigning a 42% correction would be unjustified.

Data receive unit event weight and only the raw trigger-bit OR is retained. If a trigger were prescaled without inverse-prescale weighting, it would lower the data yield, so prescale omission has the wrong sign to produce the observed data excess. The 2024 menu still needs a run-by-run check that Photon200 is unprescaled. The current flat skim lacks path-specific bits and prescale columns, so that check cannot be done from the skim alone.

The photon CSEV SF exists in the 2024 payload but is not applied. For Medium photons it spans 0.948–0.978 in EB and 0.892–0.920 in EE depending on \(R_9\). With the observed EB/EE mixture it would lower total MC by about 3.6–6.6% and worsen Data/MC to approximately 1.47–1.52. It is a required correctness fix, not an agreement fix. `Photon_r9` must be retained, and the SF should be applied to prompt photons rather than the data-driven fake component.

## Other corrections

The current wrapper applies Summer24Prompt24_V5 JEC, Summer24Prompt24_JRV2 JER, lepton/photon momentum corrections, and propagated PuppiMET before calling `real_subset_worker.py`. The old JEC/MET hooks are replaced by passthrough functions, so no duplicate correction was found (`intermediate_2024_worker.py:327-352,382-385`; `object_corrections_2024.py:602-735`).

Representative total-yield shifts are:

- pileup up/down: -2.18% / -0.73%;
- photon ID up/down: +4.74% / -4.74%;
- largest listed b-tag shift: about 1.14%.

Both pileup directions lying below nominal should be inspected, but this is not an order-one effect.

There is one GCR-specific bug: the selection uses photon-cleaned jets (`real_subset_worker.py:1424,1452-1455,1482`), while the b-tag event weight is computed with all `good_j` jets (`real_subset_worker.py:1613-1618`). The GCR b-tag weight should use `good_j & photon_clean_j`. Existing variations imply a percent-level, not 42%, effect.

## What should be done

1. Make the GJ and QCD prompt-photon samples mutually exclusive and validate their stitching. QCD is about 92.6% prompt-photon origin after the GCR selection, so treating all QCD as fake is wrong.
2. Replace only the hadron-fake component with the data-driven fake estimate. Then constrain the mutually exclusive prompt pool in GCR and demonstrate transfer-factor closure before propagating it.
3. Measure and apply the Photon175-or-Photon200 trigger efficiency/SF.
4. Fix the photon-cleaned b-tag weight scope.
5. Add the prompt-photon CSEV SF with \(R_9\), knowing it worsens the raw ratio.
6. Inspect the pileup variation asymmetry and add a top-\(p_T\) systematic.

The prompt-pool diagnostic fit gives a scale near 1.3895 when the data-driven fake component is used; keeping nominal QCD and scaling GJ alone gives about 1.771. These are control-region diagnostics, not permission to alter generator cross sections. No luminosity or global cross-section fudge should be made.

Machine-readable details are in `autonomous_allhad/validation/gcr_normalization_corrections_audit_20260726.json`.

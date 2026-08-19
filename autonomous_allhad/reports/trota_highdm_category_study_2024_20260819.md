# 2024 TROTA TopResolved impact on the High-dM categories

Status: **complete** (5,489/5,489 intermediate ROOT files, 0 failures).

## Executive conclusion

The 2024 intermediate format now supports a physically meaningful resolved-top split in the current High-dM `Nt=0, Nw=0, Nb>=1` block. The adopted baseline is the **55-bin tail-merged scheme**. The recommended next validation target is an **exploratory 61-bin scheme**: replace each of the six inclusive `Nres` recoil bins retained in category 1 by `Nres=0` and `Nres>=1`. This is a physics proposal, not an adopted category definition.

Do not yet split boosted-tag categories by `Nres`: the intermediate format does not retain the AK8-subjet identity needed to reproduce the Run-2 AK4/AK8 cross-cleaning. Also do not adopt the separate `Nres>=2` **67-bin** alternative before TROTA data/MC scale factors, systematic variations, control-region closure, and a full expected-limit comparison exist.

## Provenance and scope

| Quantity | Value |
|---|---|
| Intermediate ROOT files | 5,489 |
| Events scanned | 230,830,776 |
| Events in the studied High-dM regions/block | 1,863,285 |
| Sparse TROTA rows scanned | 82,658,502 |
| TROTA rows attached to studied events | 3,838,226 |
| Files failed | 0 |
| Input manifest SHA256 | `6ebea828c8bce53d21dc16f0af6dc6f83cd2c54872628b9ce1849a7d518453b9` |
| Normalization SHA256 | `712950966065f1cba16ba5a5f5a0a14a69c42ab06e99569bf9140e6188049e9c` |

The study is restricted to events already assigned to a current High-dM region with `Nb>=1`, `Nt=0`, `Nw=0`, and `nboosted_total=0`. Event weights are generator weights times the validated cross-section/luminosity/sum-of-weights normalization. They do **not** include post-skim AnalysisSF weights or a TROTA SF.

## Adopted 55-bin baseline and proposed mapping

The adopted 55-bin layout is obtained from the 60-bin precursor by merging the last two recoil bins in five categories: zero-based source-bin pairs `(22,23)`, `(34,35)`, `(40,41)`, `(52,53)`, and `(58,59)`. Categories 1, 2, 3, 5, and 8 retain six recoil bins. Category 1 is `NT0_Nb1plus_T0_W0`, so it is untouched by the five tail merges.

Therefore the two-way resolved-top proposal is `55 - 6 + 12 = 61` bins. The three-way diagnostic is `55 - 6 + 18 = 67` bins. The full-file TROTA scan studied exactly the six unmerged category-1 recoil bins, so correcting the baseline from the obsolete 60-bin label to the adopted 55-bin label does not require reprocessing events and does not change any yield or sensitivity number below.

## Full-AnalysisSF 61-bin visual validation

A dedicated refill of the category-1 events applies the same full 2024 weight bundle used by the canonical histogram. Its two Nres blocks recombine to the canonical first six bins for every stored variation; the other 49 adopted bins are copied unchanged. This closes the event-weight/provenance gap in the earlier normalized-genweight diagnostic, but it does not supply a TROTA efficiency SF.

| Check | Result |
|---|---|
| Output search bins | 61 |
| Canonical first-six-bin recombination | passed for all samples/variations |
| Unchanged adopted bins | 49 |
| Finite histogram contents | passed |
| Canonical hists SHA256 | `2ad64b5236a23c03fbe0c21ec7be7e1a51cb035bf9868e73ef959e7510c178df` |
| Full-weight Nres component SHA256 | `28e0ecdb01b9906abcbc7c391914271c66ca67c88ad3955475104d382c3b2026` |
| PNG | `highdm_sr_trota_nres61_from_adopted55_bins.png` |
| PDF | `highdm_sr_trota_nres61_from_adopted55_bins.pdf` |
| TROTA SF | unavailable; not applied |

## Run-2 mapping and candidate arbitration

The Run-2 analysis used a resolved trijet top tag (`Nres`) together with boosted-top and boosted-W multiplicities. Candidate triplets were ranked by discriminator and overlapping candidates were removed. This study mirrors that candidate-candidate arbitration by sorting on the TROTA QCD discriminator and greedily rejecting candidates that share an AK4 jet. The primary definition is the TROTA-native 1% WP plus this overlap removal. A separate robustness variant additionally requires `100 <= m(trijet) <= 250 GeV` and `|eta(trijet)| < 2`; those legacy DeepResolved cuts are not silently made part of the new TROTA definition.

| Candidate definition | Events with >=1 candidate | Fraction of raw |
|---|---|---|
| Raw sparse passing triplets | 1,010,272 | 100.0% |
| Jet-disjoint TROTA candidates | 1,010,272 | 100.0% |
| Jet-disjoint + Run-2 mass/eta | 925,791 | 91.6% |

## High-dM SR total background versus Nres

| Nres | Background yield | Fraction | Effective MC events |
|---|---|---|---|
| 0 | 7,008.0 | 30.7% | 14,756 |
| 1 | 14,687 | 64.3% | 17,533 |
| >=2 | 1,141.3 | 5.0% | 5,337.9 |
| Total | 22,837 | 100.0% | 32,849 |

The validated TROTA production uses `TTScore/(TTScore+QCDScore) >= 0.9433798789978027`, input jets with stored JetID, `pT>25 GeV`, and `|eta|<2.5`. The model SHA256 is `ce673e6497860cc67fcdfb30017301fb476e32a0a33a60e8b51a31ba109f7ef3`.

### Definition robustness in the High-dM SR

| Definition | Nres=0 | Nres=1 | Nres>=2 | Nres>=1 fraction |
|---|---|---|---|---|
| Raw sparse triplets | 7,008.0 | 4,685.6 | 11,143 | 69.3% |
| Jet-disjoint TROTA (primary) | 7,008.0 | 14,687 | 1,141.3 | 69.3% |
| Jet-disjoint + Run-2 mass/eta | 8,324.6 | 13,655 | 857.4 | 63.5% |

## High-dM SR background-process composition

| Process | Nres=0 | Nres=1 | Nres>=2 | Nres>=1 fraction |
|---|---|---|---|---|
| TT | 2,956.9 | 12,095 | 1,079.7 | 81.7% |
| ST | 311.1 | 644.8 | 29.24 | 68.4% |
| WtoLNu | 1,266.9 | 653.9 | 6.82 | 34.3% |
| Zto2Nu | 1,781.8 | 946.7 | 12.22 | 35.0% |
| QCD | 568.2 | 261.5 | 12.99 | 32.6% |
| DY | 33.79 | 26.96 | 0 | 44.4% |
| GJ | 43.13 | 26.11 | 0.0822 | 37.8% |
| VV | 46.17 | 31.88 | 0.332 | 41.1% |

## Recoil dependence

| Recoil (GeV) | Nres=0 | Nres=1 | Nres>=2 | Nres>=1 fraction |
|---|---|---|---|---|
| 250-300 | 3,489.3 | 9,045.7 | 752.0 | 73.7% |
| 300-350 | 1,629.0 | 3,353.2 | 250.7 | 68.9% |
| 350-400 | 812.3 | 1,270.6 | 86.49 | 62.6% |
| 400-500 | 661.9 | 778.3 | 43.68 | 55.4% |
| 500-800 | 376.5 | 226.1 | 8.24 | 38.4% |
| 800-1500 | 38.99 | 13.38 | 0.182 | 25.8% |

## Signal separation diagnostic

The table below is a normalized-yield, statistical-only Asimov diagnostic for this one six-bin block. It is **not** an expected limit: AnalysisSF weights, TROTA SF/uncertainty, other High-dM categories, control-region constraints, and systematics are absent.

| Signal | Current category-1 Z | 61-bin-block Z | Gain | 67-bin-block Z | Gain | 61-bin Run-2-cut gain |
|---|---|---|---|---|---|---|
| T2bW_mStop1200_mLSP500 | 0.394 | 0.401 | 1.019x | 0.404 | 1.026x | 1.015x |
| T2bW_mStop1600_mLSP1 | 0.0883 | 0.0893 | 1.011x | 0.0894 | 1.013x | 1.011x |
| T2tb_mStop1200_mLSP500 | 0.436 | 0.481 | 1.105x | 0.502 | 1.153x | 1.112x |
| T2tb_mStop1600_mLSP1 | 0.0684 | 0.0693 | 1.013x | 0.0703 | 1.027x | 1.014x |
| T2tt_mStop1200_mLSP500 | 0.446 | 0.562 | 1.258x | 0.655 | 1.468x | 1.263x |
| T2tt_mStop1600_mLSP1 | 0.0487 | 0.0535 | 1.099x | 0.0567 | 1.165x | 1.088x |

## Nb and jet-multiplicity diagnostics

| Nb | Njet | Nres=0 | Nres>=1 | Nres>=1 fraction |
|---|---|---|---|---|
| Nb1 | Nj5to6 | 4,301.6 | 6,028.8 | 58.4% |
| Nb1 | Nj7plus | 357.7 | 980.4 | 73.3% |
| Nb2 | Nj5to6 | 1,838.5 | 6,156.6 | 77.0% |
| Nb2 | Nj7plus | 194.4 | 1,465.0 | 88.3% |
| Nb3plus | Nj5to6 | 272.5 | 810.1 | 74.8% |
| Nb3plus | Nj7plus | 43.28 | 387.6 | 90.0% |

## Recommendation and required validation

1. Implement an **exploratory `highdm61`** definition on top of the adopted 55-bin tail-merged baseline by replacing its six category-1 `NT0_Nb1plus_T0_W0` recoil bins with twelve bins: the same recoil edges crossed with `Nres=0` and `Nres>=1`, where `Nres` is the jet-disjoint TROTA 1% WP multiplicity.
2. Keep `Nres=1` and `Nres>=2` merged initially. The three-way split is only a diagnostic until per-bin MC effective statistics and background closure are demonstrated.
3. Do not add `Nres` to categories with a selected boosted top/W until exact AK4/AK8 subjet cross-cleaning can be reconstructed and validated.
4. The SR category-1 component has now been rebuilt with the full AnalysisSF weight bundle and checked against the canonical histogram. Before adoption, still obtain/apply the TROTA data/MC SF and uncertainty, validate LLCR/QCDCR/GCR/DY closure and transfer factors, and compare the full nuisance-aware expected limit against the adopted 55-bin baseline.
5. Preserve both the adopted 55-bin baseline and the exploratory 61-bin output so the category change remains auditable and reversible. The obsolete 60-bin precursor must not be used as the comparison baseline.

## Machine-readable source

- Study schema: `trota_highdm_category_study_2024_v1`
- Study finished/updated: `2026-08-19T15:14:55.950707Z`
- Wall time: `1572.1 s`

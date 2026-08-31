# TROTA resolved-top reconstruction in the Run-3 all-hadronic stop search

## Purpose of this note

This note provides analysis-note-ready physics text for the resolved hadronic top reconstruction used in the 2024 and 2025 all-hadronic stop search. It is intended to supply the object-definition, event-selection, and signal-categorization discussion that is currently missing from the AN. Production details, computing workflow, and calibration bookkeeping are deliberately outside its scope.

The Run-2 analysis provides the physics motivation: a hadronically decaying top quark is not always sufficiently boosted for all three quarks to be captured by one AK8 jet. A complementary reconstruction from three AK4 jets recovers the intermediate-boost regime. The present Run-3 implementation follows that logic but uses the TROTA classifier and the Run-3 object definitions described below; Run-2 numerical preselection thresholds must therefore not be copied into the new AN.

## AN-ready text

### Resolved hadronic top candidates

Hadronic top quark decays are reconstructed in two complementary regimes. At high Lorentz boost, the three-prong decay can be contained in a single AK8 jet and is identified by the GloParT top tagger. At more moderate boost, the decay products are commonly reconstructed as three separate AK4 jets. These decays are identified with the TROTA resolved-top classifier. The two reconstructions extend the top-quark acceptance across the range of boosts expected as the stop and neutralino masses are varied.

Resolved-top candidates are formed from every combination of three AK4 jets satisfying the jet identification, with each constituent jet required to have $p_{\mathrm{T}}>25\,\mathrm{GeV}$ and $|\eta|<2.5$. The candidate four-momentum is the sum of the three constituent-jet four-momenta. For each jet, the classifier uses the jet transverse momentum, mass, area, pseudorapidity, azimuthal angle, UParTAK4 b-tagging information, and its angular coordinates relative to the trijet axis. This combines the kinematic compatibility with a three-body top decay and the heavy-flavor content expected from $t\to bW\to bq\bar q'$.

The classifier returns top-like and multijet-like scores. The discriminant used in this analysis is

\[
D_{\mathrm{TROTA}}
=\frac{S_{\mathrm{top}}}{S_{\mathrm{top}}+S_{\mathrm{QCD}}}.
\]

A candidate is tagged when $D_{\mathrm{TROTA}}\geq0.94338$, the threshold denoted as the 1% QCD-misidentification working point in the classifier study. This label describes the reference working point of that study and must not be presented as a measurement of the multijet misidentification probability after the full analysis selection. Tagged candidates are additionally required to satisfy $100\leq m_{jjj}\leq250\,\mathrm{GeV}$ and $|\eta_{jjj}|<2.0$. The same definition is used for the 2024 and 2025 selections.

### Exclusive reconstruction with AK8 top and W candidates

A physical hadronic decay must not be counted simultaneously as a merged AK8 object and as one or more resolved trijet candidates. Resolved candidates are therefore cleaned against every AK8 jet that passes either the analysis top-tag or W-tag requirement. When two valid AK8 subjets are available, an AK4 jet is considered part of the boosted object if it lies within $\Delta R<0.4$ of either subjet. If fewer than two valid subjets are available, the boosted-object footprint is defined by $\Delta R<0.8$ around the AK8 jet axis. Any resolved candidate containing one of these AK4 jets is removed.

The remaining resolved candidates are ordered by decreasing $D_{\mathrm{TROTA}}$. Starting from the highest-scoring candidate, a candidate is accepted only if none of its three AK4 jets has already been assigned to an accepted resolved top. The multiplicity $N_{\mathrm{res}}$ is the number of candidates left after both the boosted--resolved and resolved--resolved overlap removal. Consequently, the triplets counted by $N_{\mathrm{res}}$ are mutually exclusive in their AK4 constituents and are also disjoint from the selected boosted top and W objects.

This arbitration is essential to the physics interpretation of the categories. Without it, the same shower could increase more than one of $N_{\mathrm{t}}$, $N_{\mathrm{W}}$, and $N_{\mathrm{res}}$, and a single AK4 jet could seed several nominal resolved tops. After arbitration, these multiplicities represent distinct reconstructed objects and can be used to define disjoint event classes.

### Role in the high-$\Delta m$ selection

For stop decays with a mass splitting above or near the top-quark mass, the visible system contains energetic hadronic top decays. The AK8 and resolved reconstructions cover complementary boost regimes: $N_{\mathrm{t}}$ and $N_{\mathrm{W}}$ identify merged heavy-particle decays, while $N_{\mathrm{res}}$ recovers three-jet top decays that are not captured as a tagged AK8 object. The resolved multiplicity is therefore used only after the standard high-$\Delta m$ signal selection, including $N_{\mathrm{j}}\geq5$, $N_{\mathrm{b}}\geq1$, $p_{\mathrm{T}}^{\mathrm{miss}}>250\,\mathrm{GeV}$, and $m_{\mathrm{T}}^{\mathrm{b}}\geq175\,\mathrm{GeV}$ for the categorized search population.

The original $(N_{\mathrm{b}},N_{\mathrm{t}},N_{\mathrm{W}})$ classes are retained for events with $N_{\mathrm{res}}=0$. Events containing a resolved top are removed from those classes and assigned to exactly one of the following topology groups:

| Topology group | Multiplicity requirement | Physics interpretation |
|---|---|---|
| one resolved top only | $N_{\mathrm{t}}=0,\ N_{\mathrm{W}}=0,\ N_{\mathrm{res}}=1$ | one moderately boosted three-jet top candidate |
| at least two resolved tops only | $N_{\mathrm{t}}=0,\ N_{\mathrm{W}}=0,\ N_{\mathrm{res}}\geq2$ | two resolved hadronic-top candidates in the same event |
| boosted top plus resolved top | $N_{\mathrm{t}}\geq1,\ N_{\mathrm{W}}=0,\ N_{\mathrm{res}}\geq1$ | simultaneous merged and resolved top reconstruction |
| boosted W plus resolved top | $N_{\mathrm{t}}=0,\ N_{\mathrm{W}}\geq1,\ N_{\mathrm{res}}\geq1$ | a merged W candidate accompanied by a resolved top |

All four groups inherit the high-$\Delta m$ requirement $N_{\mathrm{b}}\geq1$ and are subdivided using the same missing-momentum intervals as the baseline search. The topology with $N_{\mathrm{t}}\geq1$, $N_{\mathrm{W}}\geq1$, and $N_{\mathrm{res}}\geq1$ is not retained as an independent search category because its population is too small to support a stable bin. Low-statistics adjacent bins are merged before the statistical interpretation; these mergers do not change the object definition or the exclusivity prescription. With the adopted mergers, the high-$\Delta m$ search contains 73 bins.

The category structure is deliberately coarser than the complete Run-2 $(N_{\mathrm{t}},N_{\mathrm{W}},N_{\mathrm{res}},N_{\mathrm{b}})$ lattice. It preserves the topologies that have usable Run-3 event populations while avoiding categories whose expected background and control-sample support would be too small. Its purpose is not simply to increase the bin count: it separates events according to whether the hadronic top system is merged, resolved, or reconstructed in both regimes.

### Role in the low-$\Delta m$ selection

In compressed stop spectra, the decay products of the stop are soft and the event is selected primarily through recoil against an energetic ISR jet. The low-$\Delta m$ selection therefore rejects events containing any tagged AK8 top or W candidate and additionally requires $N_{\mathrm{res}}=0$. The resolved-top veto uses the same fiducial candidate definition and the same overlap arbitration as the high-$\Delta m$ categorization.

The $N_{\mathrm{res}}=0$ requirement completes the intended heavy-object veto. A veto based only on AK8 tags would still accept moderately boosted hadronic top decays reconstructed as three AK4 jets, which are characteristic of top-quark backgrounds and of the noncompressed signal regime. Requiring $N_{\mathrm{t}}=N_{\mathrm{W}}=N_{\mathrm{res}}=0$ produces a cleaner ISR-driven topology and reduces migration between the physical interpretations of the high- and low-$\Delta m$ selections. Because every low-$\Delta m$ category inherits this veto, $N_{\mathrm{res}}=0$ need not be repeated in each plotted category label, but it must be stated explicitly in the selection text and tables.

## Compact object-definition entry for the AN summary table

| Object | Selection | Analysis use |
|---|---|---|
| TROTA resolved top | three AK4 jets with jet ID, each $p_{\mathrm{T}}>25\,\mathrm{GeV}$ and $|\eta|<2.5$; $D_{\mathrm{TROTA}}\geq0.94338$; $100\leq m_{jjj}\leq250\,\mathrm{GeV}$; $|\eta_{jjj}|<2.0$; exclusive with selected AK8 top/W objects and with other resolved candidates | $N_{\mathrm{res}}$ categories in high-$\Delta m$; resolved-top veto in low-$\Delta m$ |

## Short physics summary for the AN introduction or strategy section

The high-$\Delta m$ search reconstructs hadronic top decays in both the merged and resolved regimes. Tagged AK8 jets identify collimated top or W decays, while the TROTA classifier identifies moderately boosted top decays reconstructed from three AK4 jets. An explicit overlap-removal procedure makes the boosted and resolved objects exclusive. The resulting resolved-top multiplicity extends the high-$\Delta m$ categories and is vetoed in the ISR-based low-$\Delta m$ selection.

## Required edits in the current AN

The AN agent should make the following physics-content updates.

1. Add a Resolved top tagging subsection immediately after the AK8 top/W-tagging discussion and before ISR tagging. The first two AN-ready subsections above can be used with only citation and notation adjustments.
2. Add the compact resolved-top row to the object-definition summary table.
3. Amend every low-$\Delta m$ baseline, region-summary, and signal-selection statement from $N_{\mathrm{t}}=N_{\mathrm{W}}=0$ to $N_{\mathrm{t}}=N_{\mathrm{W}}=N_{\mathrm{res}}=0$. The category labels may omit the redundant $N_{\mathrm{res}}=0$ text.
4. Replace the obsolete 55-bin high-$\Delta m$ categorization with the exclusive $N_{\mathrm{res}}=0$ baseline plus the four retained resolved-top topology groups. State the adopted 73-bin total after the low-statistics mergers.
5. Remove the version-history statement that the resolved-object tagger has not yet been applied.
6. Update downstream references to the high-$\Delta m$ bin count, including statistical-model and systematic-uncertainty text, so they agree with the adopted category definition.

## Wording and physics guardrails

- Do not copy the Run-2 40/30/20 GeV ordered-jet thresholds, the Run-2 explicit candidate b-tag multiplicity requirement, or its candidate angular preselection. They are not part of the present TROTA object definition.
- Describe the 0.94338 threshold as the classifier's nominal 1% QCD-misidentification working point, not as an in-situ measurement of a 1% background rate.
- Do not state that resolved-top candidates are independent of the boosted objects merely because different jet radii are used; their exclusivity follows from the explicit constituent/subjet overlap removal.
- Do not describe $N_{\mathrm{res}}$ as the number of all passing triplets. It is the number remaining after boosted-object cleaning and greedy removal of shared AK4 jets.
- Do not imply that the high- and low-$\Delta m$ branches are globally orthogonal solely because of the resolved-top veto. The veto defines the low-$\Delta m$ heavy-object topology; the full relationship between the branches is set by all event selections.

## Source basis

- Run-2 analysis note AN2019_016_v9.pdf, Secs. 4.9 and 5.3: physical motivation for complementary merged and resolved hadronic-top reconstruction, object overlap removal, and use of $N_{\mathrm{res}}$ in signal categorization.
- Current Run-3 TROTA candidate definition: autonomous_allhad/autonomous_allhad/trota_resolved_2024.py.
- Current exclusive overlap and multiplicity definition: autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py.
- Current high-$\Delta m$ category mapping and bin configuration: autonomous_allhad/autonomous_allhad/search_bin_categorization.py and autonomous_allhad/configs/search_bins_2024.json (identical physics definition for 2025).
- Integrated high-/low-$\Delta m$ selection use: autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py.

# TROTASR AN replacement handoff - 2026-09-24

The user explicitly authorizes the AN agent to replace all related plots and
text with the current TROTASR analysis, including the new TopMixed object.
This supersedes prior instructions withholding delivery to the AN; it does not
authorize inventing missing results or presenting partial fits as validated.

All relative paths below are under
`/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR`.

## Object, selections, categories and statistical model

- Read `NOMINAL_CONTINUATION.md`, `selection_contract.json`,
  `utils/event_selections.py`, `utils/constituents.py`,
  `highdm_sr_binning.json`, `sr_category_merge.json`, `gnn4lowdm/config.json`,
  `background_sources.json`, and the matching workflow/model manifests.
- Historical draft-status fields in `selection_contract.json` are not current
  campaign progress: the completed 14,474-file campaign uses this implementation.
  Do not silently rewrite the frozen contract or use the old 162-bin count as
  the final likelihood count.
- TopMixed nominal candidates are integrated in the Events tree as
  `nTopMixed1pct` and `TopMixed1pct_*`; not a sidecar or re-inference. The
  TT/(TT+QCD) threshold is >=0.9027690887451172 in both 2024 and2025.
  Raw multiplicity is not Nmix: Nmix is after region cleaning and arbitration.
- Current candidate priority: Boosted > Mixed > Resolved > independent W.
  Candidate and constituent identity, AK4 footprints and geometric cleaning
  must be described from the implementation, not guessed. Shared Mixed/W
  constituents are forbidden; spatially/constituently independent Mixed and W
  candidates can coexist. Do not describe Nw as forced to zero whenever Nmix>0.
- High-dM topology requires Nbst>=1 OR Nmix>=1 OR Nres>=1 OR Nw>=1;
  low-dM vetoes all four. Rebuild topology from non-topology cuts; do not
  intersect with old tag-restricted flags. Category order Nb,Nbst,Nmix,Nres,Nw.
- Original highSR162 is an immutable histogram source. Current explicit
  projection is 21 high-dM classes/118 bins after B<2 category merges and the
  user-approved final-bin merges in display classes2,15,20. Low-dM SR stays30.
  Combined 2024+2025 card:452 bins/16 channels. CR Nb streams remain1 and>=2;
  do not give them SR topology subdivisions. Low CRs have10 bins/year each,
  with NISR pooling and five unchanged GNN bins per Nb group.
- LLCR display is now also10 bins. This explicitly supersedes the AN agent's
  earlier user-approved five-bin inclusive display. All GNN CR figures use
  real score intervals, half-open except the inclusive last endpoint1.
- High TF uses same-Nb/native-recoil process-specific SR/CR links; low TF uses
  same-Nb whole-score CR integral/covariance. Sgamma uses GNN x recoil
  templates, not score-index matching. Q is not applied to Zinv a second time.
- No Mixed-specific SF/nuisance was supplied; do not invent one or claim full
  systematic coverage. This is the current nominal analysis, not the cancelled
  systematic campaign. Preserve SR blinding and autoMCStats10 1 1.

## Plot sources and mandatory replacement audit

- `an_figure_inventory.json`:223 reference entries (156 current-analysis and67
  reference assets). Build a per-figure replacement/status map and verify hashes.
- `results/background_estimation/manifest.json`:exact125 background/CR plots
  with250 PNG/PDF hashes and23 explicit missing products. Use the paths in this
  manifest, not a blanket copy from the old nominal folder. It includes the
  latest11 GNN replacements, whereas other114 unchanged results keep their
  existing production paths. Gallery/index and ZIP are in the same directory.
- GNN yearly: `stats/nominal_20260924_sr118/plots/{2024,2025}/gnn_controls`.
  Combined: `stats/nominal_20260924_sr118/plots/gnn_controls`.
  Shared renderer entry point: plotting_nominal_distributions.draw_distribution.
  Numerical QA:40 tests and200 real-array comparisons passed. PDF11pages were
  inspected; the common renderer's legend/separator overlap and fixed ratio
  clipping remain explicitly open. Do not silently restyle plots in the AN task.
- Current high SR: `stats/nominal_20260924_sr118/plots/predictions/` and its
  manifest. LowSR30 is unchanged; locate its verified existing source using
  inventory/continuation. Read `validation/sr_tail_merge_20260924` for SR QA.
- Current contours: `stats/nominal_20260924_sr118/plots/contours/` and its manifest.
  These are PARTIAL:893/924 valid,31 failed, respectively T2tt201/205,
  T2bW339/354,T2tb353/365. Existing display cuts omit20 below600GeV points each
  in T2bW/T2tb, so plotted counts differ. Do not claim a complete grid or full
  systematic exclusion. Older162/121-bin fits cannot stand in for SR118.
- The current impact is now916/916 valid, zero remaining fit failures. Use
  `stats/nominal_20260924_sr118/impacts/profile_validation/impacts_mStop1200_mLSP500.json`
  (SHA256 db92d0524e48491b133d3fbe14618912293345c0e4adc6d781c7c706f11eeab3).
  The two formerly failed low-dM Sgamma Nb>=2 native-recoil300-350GeV lower
  endpoints were validated at theta=0 using independent minimization paths;
  DeltaNLL=0.06807462/0.05991837 for2024/2025. Their upper crossings and returns
  to the common minimum passed, with no physical-model changes. These two are
  boundary-limited, not ordinary two-sided1-sigma intervals; there are16 such
  nuisances in total. Keep this explicit in the text and captions.
- The unchanged original plotImpacts.py generated the31-page impact PDF at
  `stats/nominal_20260924_sr118/impacts/impacts_mStop1200_mLSP500.pdf`
  (SHA256 acee04246820ee3eee6c28cc46d53aa3f8ab379c05976bd110f74d735d488ab2).
  All31pages were Poppler-rendered and visually inspected. This is NOT a final
  publication-QA pass: the scientific-notation multiplier clips at the right
  edge on pages7-31, and decimal ticks are crowded on pages4 and6. Do not
  silently restyle it or mark it final; use the valid numbers and track these
  presentation issues separately. Never substitute an old model's impact.
  Detailed evidence: `validation/sgamma_profiles_20260924/`.
- Missing23:12 complete TF panels were blocked by the stage-wide gate; two
  high-QCD TF panels have unsupported cells. Eight high-Njet figures are blocked
  because the old display excludes selected Njet4. CR-only fit failed covQual3
  requirement (both trials covQual2). Keep these explicit; no fabricated figures.
- Remaining mll shaded-window/fit-window and AN/frozen-code auxiliary RZ
  definition differences are documented in the continuation. Preserve the
  actual implemented baseline and report inconsistencies instead of inventing
  a reconciliation.

Update the object chapter, overlap rules, region selections/tables/diagrams,
SR category/bin tables, CR/TF/background-method descriptions, plot captions,
nominal uncertainty scope, fit/limit discussion, counts and cross-references.
Replace available relevant figures, and explicitly mark pending/missing results
without relabeling old model outputs. Preserve team edits. Compile and visually
check the AN using its own instructions/PDF skill. Report changed files, figure
coverage, unresolved items and actual compile/synchronization outcome.

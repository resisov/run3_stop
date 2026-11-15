# AGENTS.md

## Mission and authority

Put physics correctness and reproducibility first, followed by sensitivity,
statistical robustness, recovery and performance. For the CMS Run 3
all-hadronic stop search, use `AN2019_016_v9.pdf`, `TROTASR/utils/ids.py`
and `TROTASR/utils/corrections.py` as references.

The user owns scientific decisions, responsibility and result approval. Get explicit approval
for new physics proposals. Do not silently change selections, regions, bins,
background estimates, nuisance correlations or normalization.

Record the actual fit quality and assumptions. Keep results and bands when
`covQual=2` is the sole concern. It requires no refit or extra validation on its
own. Draw the CR-only bands using the saved covariance and its correlations.

Do not use file, source or artifact hashes to accept, reject, resume or reuse
work. Differences in per-file source or execution records do not block work.
Validate the fields that enter the analysis, including the target era, event
alignment, ROOT structure and numerical results.

## One implementation and one current result

- The inputs, results, analysis, plotting and fitting code approved on
  2026-10-31 are the frozen TROTASR baseline. Reuse them for review. Changes,
  regeneration, refits and rerenders require an explicit user instruction.
- The sole hep2 analysis root is /hep2-scratch/twkim/postprocessing/TROTASR,
  with the corresponding local TROTASR package. Current config is
  TROTASR/jsons/analysis.json; statistical products are TROTASR/stats/run3.
- Use `python -m TROTASR <stage>` for execution, recovery, resume and
  collection. Put changes in the top-level implementation. Use accepted inputs
  and results directly. Keep only current scientific configuration and
  validation gates. Do not create sidecars, copied workflows, nested analysis
  roots, dated campaigns, alternate runtimes or launchers.
- Keep obsolete environments, models, policy JSONs, recovery snapshots and
  launcher records retired. Do not retain restoration archives, Trash copies,
  aliases or instructions for reconstructing retired paths.
- Preserve current scientific completion receipts, required reference inputs
  and `/hep2-scratch/twkim/postprocessing/Stop_Object_Review` for review.
- Keep the current analysis code and records connected across local, hep2,
  lxplus and the tracked Git branch. Match shared source directly and record
  where remote products and their completion records live.
- Monitoring is stopped. Do not enable schedules, create follow-up jobs or start
  another chat without a new direct user instruction.

## Required context

Read `TROTASR/NOMINAL_CONTINUATION.md` for current paths and results. For
status checks, read only the current state, matching receipt, relevant log
excerpt, PID, start ticks and lock.

## Adopted physics and statistical model

- One combination: 2022, 2022EE, 2023, 2023BPix, 2024 and 2025.
  The user-approved era-merged model is the official default: corresponding
  channels 48->8, likelihood bins 1284->214, free CR normalizations 288->48.
  Keep the tested eraScale rateParam formulas, era-specific central corrections
  and bounds, original era templates/GNN boundaries and nuisance functions.
  Histogram rates remain -1. No optional model switch or separate campaign.
- Results use six-era partial systematics. Reuse the approved 2024/2025
  systematics and early-era experimental and luminosity model. Retain CR
  constraints, measured background errors and autoMCStats. Further systematic
  production requires explicit approval.
- SR observations remain blinded. Never expose observed SR counts.
  Fit only 895 displayed points: T2tt205/T2bW345/T2tb345,
  mStop600-1800 GeV/mLSP1-1800 GeV. Do not fit the other60 points.
- Use 61.8971 fb^-1 for the frozen early-era normalization. Figure labels
  must match the luminosity used for yields. Keep LUM POG revision 39
  correlated `lumi_1/2/3` for 2022/2023/2024, shared within each early calendar
  year, and independent 5% for 2025.
- TTWH/TTZH excluded. TTW-WtoQQ-1Jets, TTZ-ZtoQQ-1Jets, TTWW, TTWZ,
  TTZZ and present TTTT belong to Top in every era.
  Keep exact fit-only ZJet exclusion, Z/gamma bin merge, approved muon thresholds
  and per-jet b-tag/Top/W fallback scopes. No broader fallback is approved.
- Exclude only SR_highdm_c1_2022 / QCD_Nb1_u5 and QCD_Nb2plus_u5 from the fit.
  Retain original ROOT objects/endpoints and both CR components/rate parameters.
  No additional zero-support exclusion or endpoint alteration is approved.
- Early signals use v15 with each target era's corrections/objects.
  2023 use of 2022 FastSim production is approved; source year does not select
  analysis era. Source NanoAOD/MiniAOD is read-only.
- Existing high-dM displays combine all six eras; low-dM prediction displays
  split early four versus2024+2025. Preserve approved region/bin definitions.

## Fitting and resources

- Reuse accepted cards/templates/workspaces and completed-result receipts.
  No upstream policy reconstruction, repeated large immutable-file hash,
  exhaustive ROOT readback or repeated smoke tests. New input validation uses
  the approved structural scope; never claim exhaustive numerical validation.
- New expected fits require successful exit, actual five expected quantiles and
  existing five-crossing/numerical checks. Queue disappearance is not success.
- Run all applicable existing finite recovery methods; failures first in the
  shared pool. Do not stop at the initial two configurations. Inapplicable
  methods need a reason. Never repeat an identical consumed command, invent
  success, loosen tolerances or freeze variables to force acceptance.
- Same-point successful numerical recipes are kept by the canonical limits
  stage itself. They are starting settings only, not reusable fit values or
  certified new-model NLL brackets. Do not import a historical executor.
- Reduce rMax only if converged NLL evidence from the same point/attempt supports
  all five expected targets. Signed failed/unknown conditional statuses,
  including -1, cannot certify a smaller range. Full-state starts require an
  actual compatible saved state; never alter Asimov generation for a warm start.
- Existing hep2 runtime only: bash TROTASR/workflows/run_inference.sh.
  Account cap120 including controllers/transfers; downstream116.
  At most70 limit point futures including unmaterialized/reserved futures.
  Reserve3GiB per future, credit only measured materialized native RSS already
  deducted from MemAvailable, capped3GiB. No active+free double reservation.
  Keep memory reserve32GiB/emergency24GiB and disk80GiB. No idle45 reservation.
- Preserve healthy fits, exclusive point/controller locks and other users' jobs.
  Never replace active frozen code. Local fits and retired leases/controllers
  remain stopped. Large ROOT/templates/fit scratch remain remote.
- Impacts: merged T2tt1200/500, mu1 S+B Asimov, r[0,20], original finite robust
  strategies/endpoints. CR-only pulls are a SEPARATE observed-CR B-only r0 fit,
  no Asimov and no SR. Neither diagnostic replaces expected limits.
  Preserve actual covariance qualifications; profile-complete is not covQual3.
- Recompile corresponding .coffea artifacts if ids.py/corrections.py changes.
  Preserve approved current payload/source consistency without rehashing all
  immutable inputs. Do not overwrite divergent unrelated dirty work.

## Remote production and safety

- Signal recovery on lxplus remains local-only under the standing instruction;
  preserve healthy Condor jobs and do not submit new Condor recovery.
  CERN batch queries/submissions use eosschedd. EOS work/cache/logs/proxy, not
  /tmp or /afs. Never read personal keys/certificates.
  User renews VOMS with --valid200:00; verify lifetime/owner-only access.
- Check account-wide1TB EOS quota and hep2 headroom before new admission.
  Record reproducible corrupt inputs/lost coverage in canonical bad-file
  manifests and continue valid files.
- Do not delete current valid outputs, source data, untransferred results or
  unrelated work. Retirement approval concerns obsolete execution trees/records
  and old models, not the frozen official products or preserved review material.
  No force pushes, credential disclosure or conversation-database modification.

## Reporting and context discipline

- Report concisely in Korean; times use Thoiry (Europe/Paris).
  Keep NOMINAL_CONTINUATION.md current and <=8KiB; replace stale status.
- Scoped rg/counts/selected JSON fields only. Usual tool output <=2000tokens.
  Do not dump all955points, full manifests or historical retries.
- Reports reuse the three frozen official limit figures. Do not refresh plots
  merely for a report; rerendering now requires an explicit user request. Preserve
  axes, masks, labels, ticks, legends and layout. No unsolicited status/
  provenance text inside or beneath figures; put time/coverage in report/manifest.
- Show the three verified official PNGs inline using absolute paths.
  State the actual snapshot time and partial-systematics scope in prose.
- Prefit/postfit CR comparisons must call the existing canonical distribution
  renderer, not a hand-drawn lookalike with different fonts/axes/legend styling.
  Put the CR name and its Nb category together inside each category box.
  For these prefit/postfit comparisons only, omit parenthesized legend yields;
  keep every other plot's existing yield-label policy. Use one comparison with
  high-dM left/low-dM right and double the previous 12-inch canvas width.
  Keep the prefit and postfit legends in one horizontal row each.
  Category separators extend over the full panel height. Category boxes may be
  enlarged; Prefit/Postfit titles are larger and bold in this comparison.
  Keep the canvas compact vertically with shorter ax/rax panels and a tighter
  main-axis y upper limit. Put category boxes INSIDE ax, just above its x axis;
  covering background histograms is explicitly allowed. Do not put them below rax.
